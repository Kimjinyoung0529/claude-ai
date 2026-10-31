#!/bin/bash
# 전체 파이프라인: (전처리는 이미 끝났다고 가정)
#   보코더(2 스레드)를 백그라운드로, TTS 사전학습 -> KSS 미세조정(2 스레드)을 순서대로.
# 사용: nohup bash run_all.sh > data/run_all.log 2>&1 &
set -e
cd "$(dirname "$0")"
. .venv/bin/activate

PRE_MIN=${PRE_MIN:-180}
FT_MIN=${FT_MIN:-90}
VOC_MIN=${VOC_MIN:-300}

nohup python -m tts.train_vocoder --minutes "$VOC_MIN" --threads 2 > data/voc.log 2>&1 &
VOC_PID=$!

python -m tts.train --stage pre --threads 2 --lr 1e-3 --minutes "$PRE_MIN" \
  --data "data/proc/data.pkl,data/proc_zeroth/shard*.pkl" > data/train_pre.log 2>&1

python -m tts.train --stage ft --threads 2 --lr 3e-4 --minutes "$FT_MIN" \
  --data data/proc/data.pkl --init ckpt/pre.pt > data/train_ft.log 2>&1

wait "$VOC_PID"
echo "ALL TRAINING DONE"
