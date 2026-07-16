# Forge deployment: local-first on an always-on desktop

Forge deploys to the operator's always-on macOS desktop (D99). Temporal
self-hosts in the local podman stack with persistence in that stack's
Postgres; the workers run as launchd-supervised host processes; Forge's
and pbook's state of record stays in Supabase Postgres, blobs in S3.
There is no remote access — Temporal binds to loopback only.

The full walkthrough (process, configuration, gotchas, backup) is
[../docs/operations/DEPLOYMENT.md](../docs/operations/DEPLOYMENT.md).

## Contents

```text
deploy/
├── local-stack/    the podman stack: Postgres + Temporal + UI + MinIO
│                   (make stack-up / stack-down wrap it)
├── launchd/        worker + stack supervision: install.sh, run-worker.sh,
│                   plist template, forge.env.example
├── s3/             bucket lifecycle policy (apply once with aws s3api)
└── README.md
```

## Quick start

```bash
podman machine start
make stack-up                                     # Postgres + Temporal + UI + MinIO
cp deploy/launchd/forge.env.example ~/.config/forge/forge.env
chmod 600 ~/.config/forge/forge.env               # fill in the CHANGEMEs
deploy/launchd/install.sh                         # --with-pbook for ingestion
open http://localhost:8233                        # Temporal UI
```

## History

The predecessor design — a single EC2 instance with Terraform, SSM
secret bootstrap, and an nginx mutual-TLS gateway for remote CLIs — was
removed by D99 (2026-07-16) and survives in git history.
