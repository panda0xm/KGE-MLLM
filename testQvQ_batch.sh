#!/bin/bash

TOTAL_QUESTIONS=107394
BATCH_SIZE=5000
MAX_PARALLEL=30
LOG_DIR="logs"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 计算批次数量（向上取整）
NUM_BATCHES=$(( (TOTAL_QUESTIONS + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "开始处理 $TOTAL_QUESTIONS 个问题，分 $NUM_BATCHES 批，每批 $BATCH_SIZE 个，最大并行 $MAX_PARALLEL 个任务..."

# 启动所有任务
for ((i=0; i<NUM_BATCHES; i++)); do
    START=$((i * BATCH_SIZE))
    END=$((START + BATCH_SIZE - 1))  # 闭区间 [START, END]
    
    # 确保最后一个批次不超过 TOTAL_QUESTIONS - 1
    if (( END >= TOTAL_QUESTIONS )); then
        END=$((TOTAL_QUESTIONS - 1))
    fi
    
    LOG_FILE="$LOG_DIR/batch_${START}_${END}.log"
    
    echo "启动任务: python testQvQ.py --start $START --end $END > $LOG_FILE 2>&1"
    python testQvQ.py --start $START --end $END > "$LOG_FILE" 2>&1 &
    
    # 控制并行数量
    if (( (i+1) % MAX_PARALLEL == 0 )); then
        echo "等待当前批次任务完成..."
        wait
    fi
done

# 等待剩余任务完成
wait

echo "所有任务已完成！日志文件保存在 $LOG_DIR/"