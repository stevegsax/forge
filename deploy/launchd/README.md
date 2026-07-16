# launchd agents (worker + stack supervision)

The macOS analog of the retired EC2 systemd units (D99): launchd keeps
the workers running on the always-on desktop and brings the podman stack
up at login.

| Agent | Behavior |
| --- | --- |
| `com.saxcapital.forge-stack` | RunAtLoad one-shot: `podman machine start` if needed, then `make stack-up` |
| `com.saxcapital.forge-worker-1` / `-2` | KeepAlive: `uv run forge worker` (identities `desktop-forge-worker-1/2`) |
| `com.saxcapital.pbook-worker` | KeepAlive: `uv run pbook worker` (opt-in) |
| `com.saxcapital.ocr-worker` | KeepAlive: `uv run --package ocr ocr worker` (opt-in) |

## Install

```bash
cp deploy/launchd/forge.env.example ~/.config/forge/forge.env
chmod 600 ~/.config/forge/forge.env   # then fill in the CHANGEMEs
deploy/launchd/install.sh             # --with-pbook for ingestion, --with-ocr for OCR
```

`install.sh` renders `com.saxcapital.forge.plist.in` per agent into
`~/Library/LaunchAgents/` and `launchctl bootstrap`s them; re-running is
idempotent, `--uninstall` removes everything. Logs land under
`$XDG_STATE_HOME/forge/logs/` (one file per agent).

Workers read the env file through `run-worker.sh`, which parses
`KEY=VALUE` lines without shell-evaluating them (the T0.7/G35 fix — a
`&` in a DB URL is inert) and refuses to start unless the file is
chmod 600. After a fresh boot the workers crash-loop politely
(ThrottleInterval 10s) until the stack agent has Temporal listening.

## Operate

```bash
launchctl print gui/$UID/com.saxcapital.forge-worker-1 | head   # status
launchctl kickstart -k gui/$UID/com.saxcapital.forge-worker-1   # restart
tail -f ~/.local/state/forge/logs/forge-worker-1.log            # logs
```

`pbook migrate` is NOT automatic (the pbook worker never auto-migrates):
run `uv run pbook migrate` once before first enabling the pbook agent,
and after upgrades that ship pbook migrations. The forge worker runs its
own migrations at startup.

## Always-on

The desktop is the availability story: D88's batch polling stalls while
the machine sleeps. Keep it awake when on AC power:

```bash
sudo pmset -c sleep 0 displaysleep 10 disksleep 0
```
