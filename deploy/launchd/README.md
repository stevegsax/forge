# launchd agents (worker supervision)

The macOS analog of the retired EC2 systemd units (D99): launchd keeps
the workers running on the always-on desktop. Every agent here is a
worker — since D104 forge starts no infrastructure, because Postgres
(`~/repos-sax/sax-datastores`) and Temporal (`~/repos-sax/sax-temporal`)
are shared stacks that boot themselves.

| Agent | Behavior |
| --- | --- |
| `com.saxcapital.forge-worker-1` / `-2` | KeepAlive: `uv run forge worker` (identities `prod-forge-worker-1/2`) |
| `com.saxcapital.pbook-worker` | KeepAlive: `uv run pbook worker` (identity `prod-pbook-worker`; opt-in, `--with-pbook`) |
| `com.saxcapital.ocr-worker` | KeepAlive: `uv run --package ocr ocr worker` (identity `prod-ocr-worker`; opt-in, `--with-ocr`) |

Two agents were retired by D104 and no longer exist: `com.saxcapital.forge-stack`
(RunAtLoad: brought forge's own podman stack up at login) and
`com.saxcapital.db-backup` (nightly `pg_dump` of the forge + pbook databases
to S3 — now sax-datastores' nightly, which dumps every database on the
instance). `install.sh --uninstall` does not know those labels any more, so
a host that still has their plists needs them booted out by hand.

Each worker identity is a *base*: the worker appends the git version of the
tree it was launched from, so a poller reports `prod-forge-worker-1@bb64d88`
(`-dirty` when the tree had uncommitted changes at launch). The `prod-` prefix
names the lane, matching `FORGE_ENV=prod` and the `forge-prod` namespace these
agents poll; the tmux staging workers use `dev-<app>-worker`. See
[WORKERS.md](../../docs/operations/WORKERS.md#which-code-is-a-worker-running).

## Install

Run the installer **from the production checkout** (`~/repos-sax/forge-prod`,
D103): `install.sh` renders every plist path from its own location, so the
copy you run decides which checkout all of these agents execute.

```bash
cp deploy/launchd/envs/prod.env.example ~/.config/forge/envs/prod.env
chmod 600 ~/.config/forge/envs/prod.env   # then fill in the CHANGEMEs
make prod-deploy REF=main                 # creates/pins ~/repos-sax/forge-prod
~/repos-sax/forge-prod/deploy/launchd/install.sh --with-pbook --with-ocr
```

`install.sh` renders `com.saxcapital.forge.plist.in` per agent into
`~/Library/LaunchAgents/` and `launchctl bootstrap`s them; re-running is
idempotent, `--uninstall` removes everything (all three worker kinds,
whether or not they were installed). Logs land under
`$XDG_STATE_HOME/forge/logs/` (one file per agent). The opt-in flags are
`--with-pbook` and `--with-ocr` (both default off).

> A `launchctl bootstrap` that fails with a bare
> `Bootstrap failed: 5: Input/output error` on a valid plist usually means the
> label was **disabled** by an earlier teardown — a per-user override stored
> outside the plists that no error message names. `launchctl print-disabled
> gui/$UID` reveals it; `launchctl enable gui/$UID/<label>` clears it (observed
> 2026-07-31).

After this one-time install, code deploys are `make prod-deploy REF=<ref>`
alone: it re-pins the checkout and restarts the workers. Re-run the
installer only to change a job definition (identities, environment,
`KeepAlive`) — see [Changing a plist](#changing-a-plist-worker-identities-environment-keepalive).

## Environment guard (T0.9)

Every worker must declare which environment it targets — there is no
default, and production is an explicit act:

- The **launchd plist** sets `FORGE_ENV=prod` and `FORGE_PROD_ACK=yes` in
  `EnvironmentVariables` (the production acknowledgement — deliberately
  never in a profile file).
- `run-worker.sh` sources the shared
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
- Neither profile declares a Temporal namespace or address — both are derived
  from `FORGE_ENV` before every connect (`forge-prod` on `:7243`, `forge-dev` on
  `:7236`), the [staging-lane isolation](../../docs/operations/WORKERS.md#staging-lane-dev-namespace).
  `FORGE_TEMPORAL_NAMESPACE` is retired; `FORGE_TEMPORAL_ADDRESS` survives as an
  override that dev and prod refuse unless it restates their own server, so a dev
  worker can never poll production's queues — it is not even on production's
  server (`sax-temporal/docs/namespaces.md`).

To run a worker **interactively**, declare the
environment yourself, and use `set -a` so the profile's values (including
`FORGE_ENV_TAG`) export — a plain `source` does not export, and the guard
rejects an unexported tag by design:

```bash
set -a; source ~/.config/forge/envs/prod.env; set +a
export FORGE_ENV=prod FORGE_PROD_ACK=yes
```

Example profiles: `deploy/launchd/envs/prod.env.example`
(`forge_prod`/`pbook_prod` on the shared prod Postgres `:5442`, real S3) and
`envs/dev.env.example` (`forge_dev`/`pbook_dev` on `:5432`, the shared dev
MinIO, no prod ack). Neither carries a backup bucket — offsite durability is
sax-datastores' nightly now (D104). After a fresh boot the workers crash-loop
politely (ThrottleInterval 10s) until the shared Temporal stack is listening.

## Operate

```bash
make prod-deploy REF=main                                       # deploy code to PROD (pin + restart)
make workers-status                                             # both lanes: prod (launchd) + dev (tmux)
make workers-restart                                            # restart PROD workers in place (see below)
make dev-worker-restart WORKER=ocr                              # restart one DEV worker (tmux lane)
launchctl print gui/$UID/com.saxcapital.forge-worker-1 | head   # one agent's launchd status
launchctl kickstart -k gui/$UID/com.saxcapital.forge-worker-1   # force-restart one agent (fallback; SIGKILL — skips the drain)
tail -f ~/.local/state/forge/logs/forge-worker-1.log            # logs
```

`make prod-deploy REF=<ref>` is how production gets new code (D103): it
re-pins `~/repos-sax/forge-prod` to that commit, syncs it, and then
restarts. `make workers-restart` restarts the workers *on the code already
pinned there* — the right tool after an environment-profile change, not a
way to ship a commit; the `launchctl kickstart` line is a per-agent
troubleshooting fallback (it kills without the graceful drain).

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
whatever code is on disk in the **pinned production checkout**
(`ThrottleInterval` is 10s, so the relaunch is near-immediate).

Since D103 that checkout is `~/repos-sax/forge-prod`, pinned to a commit,
so a bare restart re-runs the same code it was already running. To ship a
commit, use `make prod-deploy REF=<ref>`, which re-pins the checkout and
then runs exactly this restart. A worker that finds its prod checkout
dirty or unverifiable refuses to start (exit 78) — production may only run
a commit.

It prints one line per worker label. A `not installed — skipped` line is
expected for any agent not installed on this machine (ocr and pbook are
opt-in) — it is not an error.

`make workers-status` shows both lanes — the launchd production agents and
the dev tmux sessions, with crashed dev panes called out (read-only, safe
to run anytime).

### Changing a plist (worker identities, environment, KeepAlive)

`make workers-restart` cannot adopt a changed **job definition**, by
construction. It signals the running pid; `KeepAlive` relaunches the agent
from the definition launchd already has loaded, which it read at the last
`launchctl bootstrap`. Editing `install.sh` — or the rendered
`~/Library/LaunchAgents/com.saxcapital.*.plist` — changes what the *next*
bootstrap will load and nothing else. That asymmetry is the point: a restart
picks up new **code** (the relaunch re-execs `uv run` against the live
checkout) and is inert for **configuration** (`ProgramArguments`,
`EnvironmentVariables`, `KeepAlive`).

`install.sh` is the only thing that reloads a definition. Per agent it
`launchctl bootout`s the label, waits for the teardown to finish, re-renders
the plist from the template, then `launchctl bootstrap`s it again. So the
sequence that adopts the lane-based identities (each agent's second
`ProgramArguments` entry, e.g. `prod-forge-worker-1`) is:

```bash
deploy/launchd/install.sh --with-pbook --with-ocr   # repeat the SAME opt-in flags used originally
make workers-status                       # agents loaded
temporal task-queue describe --task-queue forge-task-queue   # new identity + version stamp
```

Two things to know before running it:

- **Repeat the original opt-in flags.** Agents not selected by the flags never
  enter the installer's list, so they are neither booted out nor updated: a run
  without `--with-ocr` leaves a running ocr agent on its old definition (old
  identity included), silently.
- **`bootout` terminates; it does not drain.** It will not wait out the
  worker's 5-minute `graceful_shutdown_timeout`, so in-flight activities are
  cut short and Temporal retries them on the relaunched worker. Nothing is
  lost, but work is repeated — prefer a quiet queue.

The dev lane has no such indirection: `make dev-worker-restart WORKER=<name>`
kills the tmux session and rebuilds the command line — including
`FORGE_WORKER_IDENTITY=dev-<worker>-worker` — from the current Makefile, so a
changed base takes effect at that restart.

## Nightly backups (not forge's job — D104)

Forge no longer runs a backup agent. sax-datastores dumps every database on
each instance nightly and verifies the dumps restore, so `forge_prod` and
`pbook_prod` are covered without forge configuring anything — including the
Temporal databases, which forge's own leg never touched. `FORGE_BACKUP_S3_BUCKET`
is retired; setting it does nothing. See `~/repos-sax/sax-datastores` for the
schedule and the restore check.

## Always-on

The desktop is the availability story: D88's batch polling stalls while
the machine sleeps. Keep it awake when on AC power:

```bash
sudo pmset -c sleep 0 displaysleep 10 disksleep 0
```
