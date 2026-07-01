#!/usr/bin/env bash
# Watchdog: run GRPO 50-step, auto-relaunch on the verl-0.7.0 async-rollout wedge.
# Detects wedge = log mtime frozen >9min AND gpu util 0. Kills, relaunches (verl
# resume_mode=auto picks up from last checkpoint). Stops at 50 steps or 8h.
WORKDIR=${WORKDIR:-/path/to/dualkv-flash-attn-for-rl}   # repo checkout
RUN=$WORKDIR/experiments/run_gemma4_31b_longreason_dualkv_sp1_50step_mb4.sh
LOG=/tmp/grpo_wd.log
CKPT=$WORKDIR/ckpts/gemma4-31b-longreason-grpo
WDLOG=/tmp/grpo_watchdog.status
START=$(date +%s)
MAXSEC=$((8*3600))
cycle=0

launch() {
  cycle=$((cycle+1))
  echo "[$(date +%H:%M:%S)] CYCLE $cycle: launching (resume auto)" >> $WDLOG
  setsid bash $RUN > $LOG 2>&1 &
  echo $!
}

kill_run() {
  for p in $(pgrep -f "main_ppo|run_gemma4_31b|vLLMHttpServer|AgentLoopWorker|EngineCore"); do kill -9 "$p" 2>/dev/null; done
  sleep 8
}

echo "[$(date +%H:%M:%S)] watchdog start" > $WDLOG
launch
while true; do
  sleep 120
  # done?
  DONE=$(grep -cE "Training Progress: +100%|step:50 -" $LOG 2>/dev/null)
  if [ "$DONE" -gt 0 ]; then echo "[$(date +%H:%M:%S)] REACHED 50 STEPS - done" >> $WDLOG; break; fi
  # global timeout
  NOW=$(date +%s); [ $((NOW-START)) -gt $MAXSEC ] && { echo "[$(date +%H:%M:%S)] 8h timeout" >> $WDLOG; break; }
  # process alive?
  if ! pgrep -f main_ppo >/dev/null; then
    echo "[$(date +%H:%M:%S)] main_ppo gone (crash/exit) - relaunching" >> $WDLOG
    kill_run; launch; continue
  fi
  # wedge detection: log mtime frozen >9min AND util 0
  MT=$(stat -c %Y $LOG 2>/dev/null); AGE=$(( $(date +%s) - MT ))
  UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  STEP=$(grep -oE "training/global_step:[0-9]+" $LOG 2>/dev/null | tail -1)
  echo "[$(date +%H:%M:%S)] alive logage=${AGE}s util=${UTIL} $STEP" >> $WDLOG
  if [ "$AGE" -gt 540 ] && [ "${UTIL:-0}" -lt 5 ]; then
    echo "[$(date +%H:%M:%S)] WEDGE detected (logage ${AGE}s, util ${UTIL}) - kill+resume" >> $WDLOG
    kill_run; launch
  fi
done
echo "[$(date +%H:%M:%S)] watchdog exit" >> $WDLOG
