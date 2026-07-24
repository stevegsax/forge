# launchd agents (worker + stack supervision)

The macOS analog of the retired EC2 systemd units (D99): launchd keeps
the workers running on the always-on desktop and brings the podman stack
up at login.

| Agent | Behavior |
| --- | --- |
| `com.saxcapital.forge-stack` | RunAtLoad one-shot: `podman machine start` if needed, then `make stack-up` |
| `com.saxcapital.forge-worker-1` / `-2` | KeepAlive: `uv run forge worker` (identities `desktop-forge-worker-1/2`) |
| `com.saxcapital.pbook-worker` | KeepAlive: `uv run pbook worker` (opt-in, `--with-pbook`) |
| `com.saxcapital.ocr-worker` | KeepAlive: `uv run --package ocr ocr worker` (opt-in, `--with-ocr`) |
| `com.saxcapital.db-backup` | Daily 03:30: `pg_dump` forge + pbook → S3 (opt-in, `--with-backup`) |

## Install

```bash
cp deploy/launchd/envs/prod.env.example ~/.config/forge/envs/prod.env
chmod 600 ~/.config/forge/envs/prod.env   # then fill in the CHANGEMEs
deploy/launchd/install.sh --with-backup   # + --with-pbook / --with-ocr as needed
```

`install.sh` renders `com.saxcapital.forge.plist.in` (workers) and
`com.saxcapital.db-backup.plist.in` (backup) per agent into
`~/Library/LaunchAgents/` and `launchctl bootstrap`s them; re-running is
idempotent, `--uninstall` removes everything (all agents, whether or not
they were installed). Logs land under `$XDG_STATE_HOME/forge/logs/` (one
file per agent). The opt-in flags are `--with-pbook`, `--with-ocr`, and
`--with-backup` (all default off — enable `--with-backup` on the
production host so the databases have offsite durability).

## Environment guard (T0.9)

Every worker (and the backup job) must declare which environment it
targets — there is no default, and production is an explicit act:

- The **launchd plist** sets `FORGE_ENV=prod` in `EnvironmentVariables`,
  and for the workers/backup also `FORGE_PROD_ACK=yes` (the production
  acknowledgement — deliberately never in a profile file).
- `run-worker.sh` (and `backup-app-dbs.sh`) source the shared
  `load-env.sh`, which loads the matching profile
  `~/.config/forge/envs/$FORGE_ENV.env`, parsing `KEY=VALUE` lines
  **without shell-evaluating them** (the T0.7/G35 property — a `&` in a
  DB URL is inert) and refusing to start unless the file is chmod 600.
- Each profile declares `FORGE_ENV_TAG` (`prod` / `dev`); `load-env.sh`
  aborts if the tag disagrees with `FORGE_ENV`, so you cannot source a
  dev profile into a prod agent (or vice-versa). `FORGE_PROD_ACK` is
  never set by the profile or the scripts — only by the plist (or an
  interactive shell), so sourcing a profile can never by itself grant
  production access.
- Each profile also declares `FORGE_TEMPORAL_NAMESPACE` (prod `default`,
  dev `forge-dev`) — the [staging-lane isolation](../../docs/operations/WORKERS.md#staging-lane-dev-namespace).
  A coherence check refuses to connect unless it matches `FORGE_ENV`
  (`prod`→`default`; `dev`/`test`→anything but `default`), so a dev
  worker can never poll production's queues in the shared Temporal server.

To run a worker or `make backup-app-dbs` **interactively**, declare the
environment yourself, and use `set -a` so the profile's values (including
`FORGE_ENV_TAG`) export — a plain `source` does not export, and the guard
rejects an unexported tag by design:

```bash
set -a; source ~/.config/forge/envs/prod.env; set +a
export FORGE_ENV=prod FORGE_PROD_ACK=yes
```

Example profiles: `deploy/launchd/envs/prod.env.example` (local
`forge` + `pbook` databases, real S3, `FORGE_BACKUP_S3_BUCKET`) and
`envs/dev.env.example` (`forge_dev`, local MinIO, no prod ack). After a
fresh boot the workers crash-loop politely (ThrottleInterval 10s) until
the stack agent has Temporal listening.

## Operate

```bash
make workers-status                                             # both lanes: prod (launchd) + dev (tmux)
make workers-restart                                            # restart PROD workers only (see below)
make dev-worker-restart WORKER=ocr                              # restart one DEV worker (tmux lane)
launchctl print gui/$UID/com.saxcapital.forge-worker-1 | head   # one agent's launchd status
launchctl kickstart -k gui/$UID/com.saxcapital.forge-worker-1   # force-restart one agent (fallback; SIGKILL — skips the drain)
tail -f ~/.local/state/forge/logs/forge-worker-1.log            # logs
```

`make workers-restart` is the standard way to restart production; the
`launchctl kickstart` line above is a per-agent troubleshooting fallback
(it kills without the graceful drain).

Both the forge and pbook workers run their own migrations at startup
(verified from the pbook worker's startup log, 2026-07-24: "Database
migrations applied (head)" — the earlier "never auto-migrates" claim
predated the T3.4 worker scaffold). `uv run pbook migrate` remains
available for migrating without starting a worker.

## Restarting workers

`make workers-restart` restarts the **production lane only**: it resolves
each launchd worker label to its pid and sends `SIGTERM` to that pid —
never a command-line pattern. (The dev tmux workers run byte-identical
command lines — the env split lives in environment variables — so the
earlier pkill-by-pattern restart took the staging lane down with
production; observed 2026-07-24. The dev lane restarts independently via
`make dev-worker-restart WORKER=<name>`.) Each worker handles `SIGTERM`
(and `SIGINT`, for foreground runs) by draining gracefully — it stops
polling for new work, waits up to `graceful_shutdown_timeout` (5 minutes)
for in-flight activities to finish, then exits 0. launchd's `KeepAlive` is
unconditional, so a clean exit relaunches the agent immediately from
whatever code is on disk in the repo checkout (`ThrottleInterval` is
10s, so the relaunch is near-immediate). This is the standard way to
pick up newly merged code without a manual `launchctl kickstart` per
agent — no plist changes, no install.

It prints one line per worker label. A `not installed — skipped` line is
expected for any agent not installed on this machine (ocr and pbook are
opt-in) — it is not an error.

`make workers-status` shows both lanes — the launchd production agents and
the dev tmux sessions, with crashed dev panes called out (read-only, safe
to run anytime).

## Nightly backups (`--with-backup`)

The `com.saxcapital.db-backup` agent runs
`deploy/local-stack/backup-app-dbs.sh` daily at 03:30: `pg_dump -Fc` of
the `forge` and `pbook` databases out of the `forge-postgres` container,
uploaded to `s3://$FORGE_BACKUP_S3_BUCKET/db-backups/` with a UTC
timestamp (offsite durability now that the app store is local — T0.9).
It loads the prod profile the same way the workers do (its plist sets
`FORGE_ENV=prod` + `FORGE_PROD_ACK=yes`; `FORGE_BACKUP_S3_BUCKET` and the
AWS creds come from the profile). Run it by hand with `make
backup-app-dbs` (needs `FORGE_ENV` set in the shell). Logs:
`~/.local/state/forge/logs/db-backup.log`.

## Always-on

The desktop is the availability story: D88's batch polling stalls while
the machine sleeps. Keep it awake when on AC power:

```bash
sudo pmset -c sleep 0 displaysleep 10 disksleep 0
```
