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
    log_system_time_list = []

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
                delay = log_system_time - frame_time
                received_delays.append(delay)
                if delay > 0.15:
                    # each frame duration is 100ms, true delay > 50ms
                    print(f"check lidar timestamp {frame_id_str}")

                
                # 记录下来，给后面的 End process 使用
                start_time_records[frame_id_str] = log_system_time
                log_system_time_list.append(log_system_time)
                
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

    # 图 1: Received Delays - 垂直线图
    plt.figure(figsize=(20,10))
    plt.vlines(range(len(received_delays)), ymin=0, ymax=received_delays, colors='blue', linewidth=0.5)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title("Received Delay - Line Plot")
    plt.xlabel("Frame Index")
    plt.ylabel("Time (Seconds)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig("/home/weizh/data/1_received_delay_lines.png")
    plt.close()

    # 图 2: Processing Durations - 垂直线图
    plt.figure(figsize=(20,10))
    plt.vlines(range(len(processing_durations)), ymin=0, ymax=processing_durations, colors='red', linewidth=0.5)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title("Processing Duration - Line Plot")
    plt.xlabel("Frame Index")
    plt.ylabel("Time (Seconds)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig("/home/weizh/data/2_processing_duration_lines.png")
    plt.close()

    # 图 3: Received Delays - 直方图
    # 针对大部分 < 0.1s，少数 > 1s 的情况，我们重点展示 0-0.2s 区域，
    # 但 bins 设置得细致一些。
    plt.figure(figsize=(20,10))
    # range 设置为 (0, 0.2) 能够看清 100ms 以内的分布；
    # 如果你想看全量（包括1s以上的），可以删掉 range 参数
    plt.hist(received_delays, bins=100, range=(0, 0.2), color='blue', edgecolor='white', alpha=0.7)
    plt.title("Received Delay Distribution (Focus on 0-200ms)")
    plt.xlabel("Delay (Seconds)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.savefig("/home/weizh/data/3_received_delay_hist.png")
    plt.close()

    # 图 4: Processing Durations - 直方图
    # 针对大部分 < 60ms (0.06s)
    plt.figure(figsize=(20,10))
    plt.hist(processing_durations, bins=50, range=(0, 0.1), color='red', edgecolor='white', alpha=0.7)
    plt.title("Processing Duration Distribution (Focus on 0-100ms)")
    plt.xlabel("Duration (Seconds)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.savefig("/home/weizh/data/4_processing_duration_hist.png")
    plt.close()

    print(f"分析完成，已生成 4 张图片：")
    print("1. 1_received_delay_lines.png      (接收延迟时序图)")
    print("2. 2_processing_duration_lines.png (处理耗时时序图)")
    print("3. 3_received_delay_hist.png       (接收延迟直方图 - 聚焦200ms内)")
    print("4. 4_processing_duration_hist.png  (处理耗时直方图 - 聚焦100ms内)")

    ######################################## frequency


    # 1. 计算接收间隔 (Intervals)
    # 计算相邻两个 log 时间的差值
    intervals = []
    for i in range(1, len(log_system_time_list)):
        interval = log_system_time_list[i] - log_system_time_list[i-1]
        intervals.append(interval)

    # 2. 计算瞬时频率 (Instantaneous Frequency)
    # 频率 = 1 / 间隔
    frequencies = []
    for t in intervals:
        if t > 0:
            frequencies.append(1.0 / t)
        else:
            frequencies.append(0) # 防止除以0

    # --- 绘图 5: 接收间隔时序图 (垂直线) ---
    plt.figure(figsize=(20,10))
    plt.vlines(range(len(intervals)), ymin=0, ymax=intervals, colors='green', linewidth=1)
    plt.axhline(sum(intervals)/len(intervals), color='orange', linestyle='--', label=f'Avg: {sum(intervals)/len(intervals):.3f}s')
    plt.title("Receiving log system time interval")
    plt.xlabel("Frame Index")
    plt.ylabel("Interval (Seconds)")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig("/home/weizh/data/5_receiving_interval_lines.png")
    plt.close()

    # --- 绘图 6: 接收间隔直方图 ---
    plt.figure(figsize=(20,10))
    # 假设你的频率是 10Hz-20Hz，间隔大概在 0.05-0.1s，设置 range 方便观察
    plt.hist(intervals, bins=50, color='green', edgecolor='white', alpha=0.7)
    plt.title("Receiving Interval Distribution")
    plt.xlabel("Interval (Seconds)")
    plt.ylabel("Frequency (Count)")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.savefig("/home/weizh/data/6_receiving_interval_hist.png")
    plt.close()

    print(f"接收统计分析完成：")
    print(f"平均接收间隔: {sum(intervals)/len(intervals):.4f} s")
    avg_freq = len(frequencies) / (log_system_time_list[-1] - log_system_time_list[0])
    print(f"平均接收频率: {avg_freq:.2f} Hz")
    print("已生成图片：5_receiving_interval_lines.png, 6_receiving_interval_hist.png, 7_receiving_frequency_lines.png")

# 集成到你之前的逻辑中
# 在解析完 log 得到 received_delays 和 processing_durations 的同时，
# 确保你有一个列表存下了所有的 log_system_time
# 然后调用：analyze_receiving_stats(all_received_log_system_times)

if __name__ == "__main__":
    file_path = '/home/weizh/data/obstacle_detector_node_6179_1769908703986.log'
    analyze_log_file(file_path)