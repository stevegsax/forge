#!/usr/bin/env bash
#
# install.sh [--with-pbook | --with-ocr] [--uninstall]
#
# Generate launchd agents from com.saxcapital.forge.plist.in and load them
# into the gui domain of the current user:
#
#   com.saxcapital.forge-stack       RunAtLoad: podman machine + stack-up
#   com.saxcapital.forge-worker-1/2  KeepAlive: forge workers (host processes)
#   com.saxcapital.pbook-worker      KeepAlive: pbook worker (--with-pbook)
#   com.saxcapital.ocr-worker        KeepAlive: ocr worker (--with-ocr)
#
# Workers read $XDG_CONFIG_HOME/forge/forge.env (chmod 600) via run-worker.sh.
# Logs land under $XDG_STATE_HOME/forge/logs. Re-running is idempotent
# (bootout before bootstrap). Workers crash-loop politely (ThrottleInterval)
# until Temporal is reachable after a fresh boot.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEMPLATE="$SCRIPT_DIR/com.saxcapital.forge.plist.in"
AGENTS_DIR="$HOME/Library/LaunchAgents"
LOGS_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/forge/logs"
DOMAIN="gui/$(id -u)"

WITH_PBOOK=false
WITH_OCR=false
UNINSTALL=false
for arg in "$@"; do
  case "$arg" in
    --with-pbook) WITH_PBOOK=true ;;
    --with-ocr)   WITH_OCR=true ;;
    --uninstall)  UNINSTALL=true ;;
    *) echo "usage: install.sh [--with-pbook] [--with-ocr] [--uninstall]" >&2; exit 64 ;;
  esac
done

AGENTS=(forge-stack forge-worker-1 forge-worker-2)
$WITH_PBOOK && AGENTS+=(pbook-worker)
$WITH_OCR && AGENTS+=(ocr-worker)
$UNINSTALL && AGENTS=(forge-stack forge-worker-1 forge-worker-2 pbook-worker ocr-worker)

program_args() {  # program_args <name> -> <string> elements, \n-escaped for awk -v
  local w="$SCRIPT_DIR/run-worker.sh"
  case "$1" in
    forge-stack)    printf '        <string>%s</string>\\n' "$SCRIPT_DIR/start-stack.sh" ;;
    forge-worker-*) printf '        <string>%s</string>\\n' "$w" forge "desktop-${1}" ;;
    pbook-worker)   printf '        <string>%s</string>\\n' "$w" pbook ;;
    ocr-worker)     printf '        <string>%s</string>\\n' "$w" ocr ;;
  esac
}

keep_alive() {  # the stack agent is one-shot; workers are supervised
  case "$1" in
    forge-stack) echo '<false/>' ;;
    *)           echo '<true/>' ;;
  esac
}

mkdir -p "$AGENTS_DIR" "$LOGS_DIR"

for name in "${AGENTS[@]}"; do
  label="com.saxcapital.$name"
  plist="$AGENTS_DIR/$label.plist"

  launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
  # bootout of a running agent is asynchronous — bootstrapping again while
  # launchd is still tearing it down fails with EIO. Wait until it's gone.
  for _ in $(seq 1 20); do
    launchctl print "$DOMAIN/$label" >/dev/null 2>&1 || break
    sleep 0.5
  done

  if $UNINSTALL; then
    rm -f "$plist"
    echo "removed $label"
    continue
  fi

  args="$(program_args "$name")"
  awk -v label="$label" -v name="$name" -v repo="$REPO_ROOT" \
      -v logs="$LOGS_DIR" -v keep="$(keep_alive "$name")" -v args="$args" '
    { gsub(/@LABEL@/, label); gsub(/@NAME@/, name); gsub(/@REPO@/, repo);
      gsub(/@LOGS@/, logs);   gsub(/@KEEP_ALIVE@/, keep);
      if ($0 ~ /@PROGRAM_ARGS@/) { printf "%s", args } else { print } }
  ' "$TEMPLATE" > "$plist"

  launchctl bootstrap "$DOMAIN" "$plist"
  echo "installed + started $label"
done

$UNINSTALL || echo "logs: $LOGS_DIR — env: ${XDG_CONFIG_HOME:-$HOME/.config}/forge/forge.env"
