#!/usr/bin/env python3
import subprocess
import time
import signal
import sys
import os
import atexit
import json
import tempfile

SCRIPTS = [
    {"name": "GPTP Sync", "path": "/home/ubuntu/work/linuxptp/run_gptp.sh", "delay": 2},
    {"name": "LiDAR Driver", "path": "/home/ubuntu/work/rs_lidar_ws/run_lidar_driver.sh", "delay": 3},
    {"name": "Camera Driver", "path": "/home/ubuntu/work/sr75_cluster_driver_ws/run_camera_driver_comp.sh", "delay": 3},
    {"name": "GPS Driver", "path": "/home/ubuntu/work/serial_ws/run_gps.sh", "delay": 3},
    {"name": "Data Recording", "path": "/home/ubuntu/work/bags/record.sh", "delay": 0},
]

processes = []
_killed_once = False

# Persist process group IDs so a watchdog can still clean up even if this process is killed
STATE_PATH = os.path.join(tempfile.gettempdir(), "one_click_record_state.json")

def _proc_desc(p):
    try:
        return f"pid={p.pid}"
    except Exception:
        return "pid=<unknown>"

def _save_state():
    """
    Save current process groups to a temp file so a watchdog can kill them later.
    """
    try:
        pgids = []
        for p in processes:
            try:
                pgids.append(int(os.getpgid(p.pid)))
            except Exception:
                pass
        state = {"pgids": sorted(list(set(pgids)))}
        with open(STATE_PATH, "w") as f:
            json.dump(state, f)
    except Exception:
        pass

def _load_state_pgids():
    try:
        with open(STATE_PATH, "r") as f:
            state = json.load(f)
        return [int(x) for x in state.get("pgids", [])]
    except Exception:
        return []

def _wait_ros2_bag_record_exit(max_wait_sec=30.0):
    """
    Wait until all `ros2 bag record` processes exit.

    This is important because ros2 bag writes/flushes files (e.g. metadata.yaml) on SIGINT.
    We wait for the process to disappear to maximize the chance that metadata is written.
    """
    start = time.time()
    while True:
        # pgrep returns 0 if found, 1 if not found
        r = subprocess.run(
            ["pgrep", "-f", "ros2 bag record"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if r.returncode != 0:
            return True

        if time.time() - start > float(max_wait_sec):
            return False

        time.sleep(0.2)

def _kill_pgids(pgids, sig):
    for pgid in pgids:
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass
        except Exception as e:
            print(f"[Console] Failed to signal pgid={pgid} sig={sig}: {e}")

def _pkill_chain(pattern, max_wait_sec=10.0, sigint_first=True):
    """
    Kill processes by pattern with escalation and optional wait.
    """
    if sigint_first:
        try:
            subprocess.run(["pkill", "-SIGINT", "-f", pattern], check=False)
        except Exception:
            pass

    # wait them exit
    start = time.time()
    while True:
        r = subprocess.run(
            ["pgrep", "-f", pattern],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if r.returncode != 0:
            break
        if time.time() - start > float(max_wait_sec):
            break
        time.sleep(0.2)

    # SIGTERM then SIGKILL if still alive
    r = subprocess.run(
        ["pgrep", "-f", pattern],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if r.returncode == 0:
        try:
            subprocess.run(["pkill", "-SIGTERM", "-f", pattern], check=False)
        except Exception:
            pass
        time.sleep(0.8)

    r = subprocess.run(
        ["pgrep", "-f", pattern],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if r.returncode == 0:
        try:
            subprocess.run(["pkill", "-SIGKILL", "-f", pattern], check=False)
        except Exception:
            pass

def _watchdog_main(parent_pid, poll_sec=0.5):
    """
    Watchdog process: if parent dies (e.g. terminal window closed and parent is killed),
    perform cleanup using persisted STATE_PATH + pkill for ros2 bag record.

    This makes closing the terminal window (X) reliable even when no signal reaches parent.
    """
    # Wait until parent process exits
    while True:
        try:
            os.kill(parent_pid, 0)  # check existence
        except ProcessLookupError:
            break
        except PermissionError:
            # If we can't check, keep trying
            pass
        time.sleep(poll_sec)

    print("[Watchdog] Parent exited, starting cleanup...")

    # 0) Stop ros2 bag record first and WAIT it exits to finish writing metadata.yaml
    _pkill_chain(r"ros2 bag record", max_wait_sec=60.0, sigint_first=True)

    # 1) Kill recorded process groups (best effort)
    pgids = _load_state_pgids()
    if pgids:
        _kill_pgids(pgids, signal.SIGINT)
        time.sleep(3.0)
        _kill_pgids(pgids, signal.SIGTERM)
        time.sleep(1.0)
        _kill_pgids(pgids, signal.SIGKILL)

    # 2) Fallback: kill common long-running child processes even if not captured in state
    # gPTP
    _pkill_chain(r"ptp4l(\s|$)", max_wait_sec=5.0, sigint_first=True)
    _pkill_chain(r"phc2sys(\s|$)", max_wait_sec=5.0, sigint_first=True)

    # ROS2 launch / containers / nodes
    _pkill_chain(r"/opt/ros/humble/bin/ros2 launch", max_wait_sec=5.0, sigint_first=True)
    _pkill_chain(r"rclcpp_components/component_container", max_wait_sec=5.0, sigint_first=True)
    _pkill_chain(r"rslidar_sdk_node", max_wait_sec=5.0, sigint_first=True)

    print("[Watchdog] Cleanup finished.")

def kill_all_processes(timeout_sec=5.0):
    """
    Best-effort cleanup of all spawned child processes.

    Strategy:
    0) First, mimic: `pkill -SIGINT -f "ros2 bag record"` (stop rosbag recording first)
    1) Send SIGINT to each process group (graceful shutdown for ROS/record scripts)
    2) Wait up to timeout_sec
    3) Escalate to SIGTERM for remaining
    4) Finally SIGKILL for remaining (force kill)

    Note: this only targets processes spawned by this script (tracked in `processes`),
    plus the extra pkill step for any lingering `ros2 bag record`.
    """
    global _killed_once
    if _killed_once:
        return
    _killed_once = True

    # Ensure state is saved before killing
    _save_state()

    print("\n[Console] Stopping all processes, please wait...")

    # Step 0: stop rosbag recording first (equivalent to: pkill -SIGINT -f "ros2 bag record")
    # Then WAIT until ros2 bag exits, so it has time to flush/write metadata.yaml.
    try:
        subprocess.run(
            ["pkill", "-SIGINT", "-f", "ros2 bag record"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        print('[Console] pkill -SIGINT -f "ros2 bag record" executed')
    except Exception as e:
        print(f'[Console] Failed to run pkill for "ros2 bag record": {e}')

    print('[Console] Waiting for "ros2 bag record" to exit (to finish writing metadata.yaml)...')
    ok = False
    try:
        ok = _wait_ros2_bag_record_exit(max_wait_sec=60.0)
    except Exception as e:
        print(f'[Console] Error while waiting for "ros2 bag record" to exit: {e}')

    if ok:
        print('[Console] "ros2 bag record" exited.')
    else:
        print('[Console] Timeout waiting for "ros2 bag record". Escalating to SIGTERM/SIGKILL...')

        try:
            subprocess.run(
                ["pkill", "-SIGTERM", "-f", "ros2 bag record"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception as e:
            print(f'[Console] Failed to run pkill -SIGTERM for "ros2 bag record": {e}')

        time.sleep(1.0)

        # If still alive, SIGKILL
        try:
            still = subprocess.run(
                ["pgrep", "-f", "ros2 bag record"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode == 0
            if still:
                subprocess.run(
                    ["pkill", "-SIGKILL", "-f", "ros2 bag record"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                print('[Console] pkill -SIGKILL -f "ros2 bag record" executed')
        except Exception as e:
            print(f'[Console] Failed to SIGKILL "ros2 bag record": {e}')

    # stop in reverse order: stop recording first, then drivers
    for p in reversed(processes):
        try:
            pgid = os.getpgid(p.pid)
            os.killpg(pgid, signal.SIGINT)
            print(f"SIGINT sent to process group {pgid} ({_proc_desc(p)})")
        except ProcessLookupError:
            # already exited
            pass
        except Exception as e:
            print(f"Failed to send SIGINT ({_proc_desc(p)}): {e}")

    # wait for graceful exit
    deadline = time.time() + float(timeout_sec)
    while time.time() < deadline:
        alive = [p for p in processes if p.poll() is None]
        if not alive:
            print("[Console] All processes exited gracefully.")
            return
        time.sleep(0.2)

    # escalate: SIGTERM
    alive = [p for p in processes if p.poll() is None]
    if alive:
        print(f"[Console] {len(alive)} processes still running, escalating to SIGTERM...")
    for p in alive:
        try:
            pgid = os.getpgid(p.pid)
            os.killpg(pgid, signal.SIGTERM)
            print(f"SIGTERM sent to process group {pgid} ({_proc_desc(p)})")
        except ProcessLookupError:
            pass
        except Exception as e:
            print(f"Failed to send SIGTERM ({_proc_desc(p)}): {e}")

    time.sleep(0.8)

    # final: SIGKILL
    alive = [p for p in processes if p.poll() is None]
    if alive:
        print(f"[Console] {len(alive)} processes still running, escalating to SIGKILL...")
    for p in alive:
        try:
            pgid = os.getpgid(p.pid)
            os.killpg(pgid, signal.SIGKILL)
            print(f"SIGKILL sent to process group {pgid} ({_proc_desc(p)})")
        except ProcessLookupError:
            pass
        except Exception as e:
            print(f"Failed to send SIGKILL ({_proc_desc(p)}): {e}")

    print("[Console] Cleanup sequence finished.")

def signal_handler(sig, frame):
    # Handle Ctrl+C (SIGINT), terminal close/hangup (SIGHUP), and termination (SIGTERM)
    kill_all_processes()
    sys.exit(0)

# Register common termination signals
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
try:
    signal.signal(signal.SIGHUP, signal_handler)
except AttributeError:
    # SIGHUP may not exist on some platforms
    pass
try:
    # Some terminal emulators may use SIGQUIT when closing the window
    signal.signal(signal.SIGQUIT, signal_handler)
except AttributeError:
    pass

# As a last resort, attempt cleanup on normal interpreter exit as well
atexit.register(kill_all_processes)

def main():
    print("=== One-click Auto Recording Launcher ===")

    for script in SCRIPTS:
        print(f"\n[Start] {script['name']}...")
        try:
            # Use a new process group so we can terminate all children later.
            p = subprocess.Popen(
                script["path"],
                shell=True,
                preexec_fn=os.setsid,
            )
            processes.append(p)
            _save_state()

            if script["delay"] > 0:
                print(f"Waiting {script['delay']} seconds...")
                time.sleep(script["delay"])
        except Exception as e:
            print(f"Failed to start {script['name']}: {e}")
            kill_all_processes()
            return

    print("\n" + "=" * 30)
    print("All processes started! Recording...")
    print("Press [Ctrl+C] or close this window to stop everything.")
    print("=" * 30)

    # Keep the main process alive until the recording script exits or user interrupts.
    try:
        # Monitor the recording script status (the last one in the list)
        processes[-1].wait()
    except KeyboardInterrupt:
        pass
    finally:
        kill_all_processes()

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--watchdog":
        _watchdog_main(int(sys.argv[2]))
        sys.exit(0)

    # Spawn a watchdog that will clean up if this process is killed by terminal close (X)
    try:
        subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--watchdog", str(os.getpid())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        pass

    main()
