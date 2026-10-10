#!/bin/bash
# Usage: tools/snap.sh "<query string, e.g. m=2&skip&demo>" out.png [width] [height] [wait_ms]
# Loads the dev page in headless Chrome in REAL time (frames run at normal speed), waits wait_ms
# after load, saves a screenshot and prints page console lines and uncaught errors.
# Widths under 760 emulate a real phone viewport (touch, coarse pointer) at the exact size given.
DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${PORT:-8765}"
if ! curl -s -o /dev/null "http://127.0.0.1:$PORT/"; then
  (cd "$DIR" && nohup python3 tools/serve.py "$PORT" >/dev/null 2>&1 &)
  sleep 1
fi
node "$DIR/tools/snap.mjs" "$1" "$2" "${3:-1440}" "${4:-900}" "${5:-6000}"
