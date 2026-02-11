import re
import matplotlib.pyplot as plt
import os
import sys

import matplotlib.pyplot as plt
import os

def analyze_log_file(file_path):
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return

    # 存储结果
    received_delays = []      # 延迟：Log系统时间 - Frame内部时间
    processing_durations = [] # 耗时：End Log时间 - Received Log时间
    
    # 用于记录 Received frame 出现的时刻，方便和 End process 对接
    # key: frame_id, value: log_system_time
    start_time_records = {}

    # only store Received frame log system time, used to analyze receive frequency and pattern
    rec_log_time_list = []
    rec_timestamp_list = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 1. 识别这行日志是否包含我们要的信息
            is_received = "Received frame" in line
            is_end = "End process frame" in line

            if not (is_received or is_end):
                continue

            # 2. 提取 Log 系统时间 (位于第二个方括号中)
            # 例子: [INFO] [1769908706.896759384] ...
            # 先按 ']' 分割
            parts_by_bracket = line.split(']')
            # parts_by_bracket[1] 会得到 " [1769908706.896759384"
            log_time_str = parts_by_bracket[1].replace('[', '').strip()
            log_system_time = float(log_time_str)

            # 3. 提取末尾的 Frame 时间戳
            # 例子: ... Received frame 1769908706.785644293
            # 按空格分割，取最后一个部分
            parts_by_space = line.split()
            frame_id_str = parts_by_space[-1]
            frame_time = float(frame_id_str)

            # 4. 根据类型进行逻辑计算
            if is_received:
                # 分析 1: Received frame 后的时间与 log info 时间的差距
                delay = log_system_time - frame_time - 0.1  # 100ms each frame duration
                received_delays.append(delay)
                if delay > 0.15:
                    # each frame duration is 100ms, true delay > 50ms
                    print(f"check lidar timestamp {frame_id_str}")

                
                # 记录下来，给后面的 End process 使用
                start_time_records[frame_id_str] = log_system_time
                rec_log_time_list.append(log_system_time)
                rec_timestamp_list.append(frame_time)
                
            elif is_end:
                # 分析 2: 相邻两个 Received 和 End 之间的 log info 时间间隔
                # 检查这个 frame_id 是否之前出现过 Received
                if frame_id_str in start_time_records:
                    start_log_time = start_time_records.pop(frame_id_str)
                    duration = log_system_time - start_log_time
                    processing_durations.append(duration)


    # --- 绘图部分 ---
    if not received_delays:
        print("未在日志中匹配到相关数据。")
        return

    ################### Received Delays
    plt.figure(figsize=(20,10))
    plt.vlines(range(len(received_delays)), ymin=0, ymax=received_delays, colors='green', linewidth=0.5)
    plt.axhline(sum(received_delays)/len(received_delays), color='orange', linestyle='--', label=f'Avg: {sum(received_delays)/len(received_delays):.3f}s')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title("Received Delay (receive log time - header timestamp - 0.1)")
    plt.xlabel("Frame Index")
    plt.ylabel("Time (Seconds)")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig("/home/weizh/data/1_received_delay.png")
    plt.close()

    ################### Processing Durations
    plt.figure(figsize=(20,10))
    plt.vlines(range(len(processing_durations)), ymin=0, ymax=processing_durations, colors='green', linewidth=0.5)
    plt.axhline(sum(processing_durations)/len(processing_durations), color='orange', linestyle='--', label=f'Avg: {sum(processing_durations)/len(processing_durations):.3f}s')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title("Processing Duration")
    plt.xlabel("Frame Index")
    plt.ylabel("Time (Seconds)")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig("/home/weizh/data/2_processing_duration.png")
    plt.close()

    ################### receive log time interval
    rec_log_time_intervals = []
    for i in range(1, len(rec_log_time_list)):
        interval = rec_log_time_list[i] - rec_log_time_list[i-1]
        rec_log_time_intervals.append(interval)
    
    plt.figure(figsize=(20,10))
    plt.vlines(range(len(rec_log_time_intervals)), ymin=0, ymax=rec_log_time_intervals, colors='green', linewidth=1)
    plt.axhline(sum(rec_log_time_intervals)/len(rec_log_time_intervals), color='orange', linestyle='--', label=f'Avg: {sum(rec_log_time_intervals)/len(rec_log_time_intervals):.3f}s')
    plt.title("Receiving log time interval (expect 0.1s)")
    plt.xlabel("Frame Index")
    plt.ylabel("Interval (Seconds)")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig("/home/weizh/data/5_receive_log_time_interval.png")
    plt.close()

    ################### receive timestamp interval
    rec_timestamp_intervals = []
    for i in range(1, len(rec_timestamp_list)):
        interval = rec_timestamp_list[i] - rec_timestamp_list[i-1]
        rec_timestamp_intervals.append(interval)
    
    plt.figure(figsize=(20,10))
    plt.vlines(range(len(rec_timestamp_intervals)), ymin=0, ymax=rec_timestamp_intervals, colors='green', linewidth=1)
    plt.axhline(sum(rec_timestamp_intervals)/len(rec_timestamp_intervals), color='orange', linestyle='--', label=f'Avg: {sum(rec_log_time_intervals)/len(rec_log_time_intervals):.3f}s')
    plt.title("Receiving timestamp interval (expect 0.1s)")
    plt.xlabel("Frame Index")
    plt.ylabel("Interval (Seconds)")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig("/home/weizh/data/6_receive_timestamp_interval.png")  # should be almost the same as 5_receive_log_time_interval
    plt.close()

    ################## summary
    print(f"接收统计分析完成：")
    print(f"平均接收间隔: {sum(rec_log_time_intervals)/len(rec_log_time_intervals):.4f} s")
    avg_freq = len(rec_log_time_list) / (rec_log_time_list[-1] - rec_log_time_list[0])
    print(f"平均接收频率: {avg_freq:.2f} Hz")

if __name__ == "__main__":
    file_path = '/home/weizh/data/obstacle_detector_node_5585_1770681952528.log'
    analyze_log_file(file_path)