#!/bin/bash
# 구간(chunk) 실행용 파이프라인. 각 단계는 체크포인트에서 이어서 학습한다.
#   보코더(2 스레드)는 백그라운드로, TTS 사전학습 -> KSS 미세조정(2 스레드)은 순서대로.
# 환경변수 PRE_MIN / FT_MIN / VOC_MIN (분). 0 이면 해당 단계를 건너뛴다.
# 주의: 하네스가 추적하는 백그라운드 작업으로 실행할 것(nohup 으로 떼어 내면 컨테이너가 회수될 수 있음).
cd "$(dirname "$0")"
. .venv/bin/activate

PRE_MIN=${PRE_MIN:-180}
FT_MIN=${FT_MIN:-90}
VOC_MIN=${VOC_MIN:-300}

if [ "$VOC_MIN" -gt 0 ]; then
  python -m tts.train_vocoder --minutes "$VOC_MIN" --threads 2 >> data/voc.log 2>&1 &
  VOC_PID=$!
fi

if [ "$PRE_MIN" -gt 0 ]; then
  python -m tts.train --stage pre --threads 2 --lr 1e-3 --minutes "$PRE_MIN" \
    --data "data/proc/data.pkl,data/proc_zeroth/shard*.pkl" >> data/train_pre.log 2>&1
fi

if [ "$FT_MIN" -gt 0 ]; then
  python -m tts.train --stage ft --threads 2 --lr 3e-4 --minutes "$FT_MIN" \
    --data data/proc/data.pkl --init ckpt/pre.pt >> data/train_ft.log 2>&1
fi

[ -n "$VOC_PID" ] && wait "$VOC_PID"
echo "CHUNK DONE PRE=$PRE_MIN FT=$FT_MIN VOC=$VOC_MIN"
