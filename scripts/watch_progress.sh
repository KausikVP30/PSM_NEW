#!/bin/bash
# Simple progress watcher - shows elapsed time and sample count every 10 minutes

set -e

INTERVAL=${1:-600}  # default 600 seconds (10 minutes)
PID=""

echo "🚀 Watching experiment progress (check every ${INTERVAL}s)..."
echo ""

while true; do
  # find running process
  PID=$(pgrep -f "run_experiment.py" || true)
  
  if [ -z "$PID" ]; then
    echo "❌ No running experiment found. Exiting."
    break
  fi
  
  # get elapsed time
  ELAPSED=$(ps -p "$PID" -o etimes= 2>/dev/null || echo "?")
  
  # convert to human-readable
  if [ "$ELAPSED" != "?" ]; then
    HOURS=$((ELAPSED / 3600))
    MINS=$(((ELAPSED % 3600) / 60))
    SECS=$((ELAPSED % 60))
    TIME_STR=$(printf "%02d:%02d:%02d" $HOURS $MINS $SECS)
  else
    TIME_STR="??:??:??"
  fi
  
  # count lines in predictions
  PROCESSED=0
  if [ -f "outputs/predictions/predictions.csv" ]; then
    PROCESSED=$(wc -l < outputs/predictions/predictions.csv)
    PROCESSED=$((PROCESSED - 1))  # subtract header
    [ "$PROCESSED" -lt 0 ] && PROCESSED=0
  fi
  
  # get total from dataset
  TOTAL=0
  for f in data/triviaqa_full.jsonl data/triviaqa.jsonl data/triviaqa_sample.jsonl; do
    if [ -f "$f" ]; then
      TOTAL=$(wc -l < "$f")
      break
    fi
  done
  
  # compute percent and ETA
  PERCENT=0
  ETA="??:??:??"
  if [ "$TOTAL" -gt 0 ]; then
    PERCENT=$((PROCESSED * 100 / TOTAL))
    if [ "$PROCESSED" -gt 0 ] && [ "$ELAPSED" != "?" ]; then
      AVG_PER_SAMPLE=$((ELAPSED / PROCESSED))
      REMAINING=$((TOTAL - PROCESSED))
      ETA_SECS=$((REMAINING * AVG_PER_SAMPLE))
      ETA_HOURS=$((ETA_SECS / 3600))
      ETA_MINS=$(((ETA_SECS % 3600) / 60))
      ETA_SECS=$((ETA_SECS % 60))
      ETA=$(printf "%02d:%02d:%02d" $ETA_HOURS $ETA_MINS $ETA_SECS)
    fi
  fi
  
  # show status
  TS=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[$TS] ⏱️  Elapsed: $TIME_STR | 📊 Progress: $PROCESSED/$TOTAL ($PERCENT%) | ⏳ ETA: $ETA"
  
  # check if done (predictions written by run_experiment_gpu.py wrapper)
  if ls outputs/metrics/metrics_gpu*_*.json >/dev/null 2>&1; then
    LATEST_METRICS=$(ls -t outputs/metrics/metrics_gpu*_*.json 2>/dev/null | head -1)
    if [ -n "$LATEST_METRICS" ]; then
      # check if modified in last 60s (still being written)
      MOD_TIME=$(stat -c %Y "$LATEST_METRICS" 2>/dev/null || stat -f %m "$LATEST_METRICS" 2>/dev/null || echo 0)
      NOW=$(date +%s)
      AGE=$((NOW - MOD_TIME))
      if [ "$AGE" -gt 60 ]; then
        echo "✅ EXPERIMENT COMPLETE!"
        echo "   Metrics: $LATEST_METRICS"
        break
      fi
    fi
  fi
  
  sleep "$INTERVAL"
done
