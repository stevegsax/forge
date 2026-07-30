#!/usr/bin/env bash
#
# install.sh [--with-pbook] [--with-ocr] [--with-backup] [--uninstall]
#
# Generate launchd agents from com.saxcapital.forge.plist.in (and the backup
# job from com.saxcapital.db-backup.plist.in) and load them into the gui domain
# of the current user:
#
#   com.saxcapital.forge-stack       RunAtLoad: podman machine + stack-up
#   com.saxcapital.forge-worker-1/2  KeepAlive: forge workers (host processes)
#   com.saxcapital.pbook-worker      KeepAlive: pbook worker (--with-pbook)
#   com.saxcapital.ocr-worker        KeepAlive: ocr worker (--with-ocr)
#   com.saxcapital.db-backup         daily 03:30: pg_dump forge+pbook -> S3
#                                    (--with-backup)
#
# The worker + backup agents declare their environment in launchd:
# EnvironmentVariables set FORGE_ENV=prod and FORGE_PROD_ACK=yes — production is
# an explicit act (T0.9). run-worker.sh / backup-app-dbs.sh then load the
# matching profile $XDG_CONFIG_HOME/forge/envs/prod.env (chmod 600) via
# load-env.sh. Logs land under $XDG_STATE_HOME/forge/logs. Re-running is
# idempotent (bootout before bootstrap). Workers crash-loop politely
# (ThrottleInterval) until Temporal is reachable after a fresh boot.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEMPLATE="$SCRIPT_DIR/com.saxcapital.forge.plist.in"
BACKUP_TEMPLATE="$SCRIPT_DIR/com.saxcapital.db-backup.plist.in"
BACKUP_SCRIPT="$REPO_ROOT/deploy/local-stack/backup-app-dbs.sh"
AGENTS_DIR="$HOME/Library/LaunchAgents"
LOGS_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/forge/logs"
DOMAIN="gui/$(id -u)"

WITH_PBOOK=false
WITH_OCR=false
WITH_BACKUP=false
UNINSTALL=false
for arg in "$@"; do
  case "$arg" in
    --with-pbook)  WITH_PBOOK=true ;;
    --with-ocr)    WITH_OCR=true ;;
    --with-backup) WITH_BACKUP=true ;;
    --uninstall)   UNINSTALL=true ;;
    *) echo "usage: install.sh [--with-pbook] [--with-ocr] [--with-backup] [--uninstall]" >&2; exit 64 ;;
  esac
done

AGENTS=(forge-stack forge-worker-1 forge-worker-2)
$WITH_PBOOK && AGENTS+=(pbook-worker)
$WITH_OCR && AGENTS+=(ocr-worker)
$WITH_BACKUP && AGENTS+=(db-backup)
$UNINSTALL && AGENTS=(forge-stack forge-worker-1 forge-worker-2 pbook-worker ocr-worker db-backup)

program_args() {  # program_args <name> -> <string> elements, \n-escaped for awk -v
  # Each worker's second argument is its BASE Temporal identity, which
  # run-worker.sh exports as FORGE_WORKER_IDENTITY; the worker appends the
  # launch-time git version (prod-forge-worker-1@<sha>). The base names the lane,
  # not the machine: "desktop-" dated from the D99 EC2 retirement and stopped
  # distinguishing anything once the desktop was the only host, whereas "prod-"
  # agrees with FORGE_ENV=prod (set below) and the `forge-prod` namespace these
  # agents poll — derived from FORGE_ENV, see sax-temporal/docs/namespaces.md.
  # The dev lane's tmux workers use dev-<app>-worker.
  local w="$SCRIPT_DIR/run-worker.sh"
  case "$1" in
    forge-stack)    printf '        <string>%s</string>\\n' "$SCRIPT_DIR/start-stack.sh" ;;
    forge-worker-*) printf '        <string>%s</string>\\n' "$w" forge "prod-${1}" ;;
    pbook-worker)   printf '        <string>%s</string>\\n' "$w" pbook "prod-${1}" ;;
    ocr-worker)     printf '        <string>%s</string>\\n' "$w" ocr "prod-${1}" ;;
  esac
}

keep_alive() {  # the stack agent is one-shot; workers are supervised
  case "$1" in
    forge-stack) echo '<false/>' ;;
    *)           echo '<true/>' ;;
  esac
}

env_vars() {  # env_vars <name> -> EnvironmentVariables dict, \n-escaped for awk -v
  # The workers are the production processes (T0.9): they declare FORGE_ENV=prod
  # and the FORGE_PROD_ACK=yes acknowledgement in launchd, so production access
  # is explicit and never the result of an unset variable. The forge-stack agent
  # only runs `make stack-up`, which touches no application database, so it gets
  # no app-DB environment.
  case "$1" in
    forge-stack) ;;
    *)
      printf '    <key>EnvironmentVariables</key>\\n'
      printf '    <dict>\\n'
      printf '        <key>FORGE_ENV</key>\\n'
      printf '        <string>prod</string>\\n'
      printf '        <key>FORGE_PROD_ACK</key>\\n'
      printf '        <string>yes</string>\\n'
      printf '    </dict>\\n'
      ;;
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

  if [[ "$name" == "db-backup" ]]; then
    awk -v label="$label" -v name="$name" -v repo="$REPO_ROOT" \
        -v logs="$LOGS_DIR" -v script="$BACKUP_SCRIPT" '
      { gsub(/@LABEL@/, label); gsub(/@NAME@/, name); gsub(/@REPO@/, repo);
        gsub(/@LOGS@/, logs);   gsub(/@SCRIPT@/, script); print }
    ' "$BACKUP_TEMPLATE" > "$plist"
  else
    args="$(program_args "$name")"
    envvars="$(env_vars "$name")"
    awk -v label="$label" -v name="$name" -v repo="$REPO_ROOT" \
        -v logs="$LOGS_DIR" -v keep="$(keep_alive "$name")" \
        -v args="$args" -v envvars="$envvars" '
      { gsub(/@LABEL@/, label); gsub(/@NAME@/, name); gsub(/@REPO@/, repo);
        gsub(/@LOGS@/, logs);   gsub(/@KEEP_ALIVE@/, keep);
        if ($0 ~ /@PROGRAM_ARGS@/) { printf "%s", args }
        else if ($0 ~ /@ENV_VARS@/) { printf "%s", envvars }
        else { print } }
    ' "$TEMPLATE" > "$plist"
  fi

  launchctl bootstrap "$DOMAIN" "$plist"
  echo "installed + started $label"
done

$UNINSTALL || echo "logs: $LOGS_DIR — env profiles: ${XDG_CONFIG_HOME:-$HOME/.config}/forge/envs/<env>.env"
