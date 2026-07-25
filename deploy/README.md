# Forge deployment: local-first on an always-on desktop

Forge deploys to the operator's always-on macOS desktop (D99). Temporal
self-hosts in the local podman stack with persistence in that stack's
Postgres; the workers run as launchd-supervised host processes out of a
**commit-pinned checkout** (`~/repos-sax/forge-prod`, deployed only by
`prod-deploy.sh` — D103); Forge's and pbook's state of record lives in
that same local Postgres (D102, rehomed off Supabase), blobs in S3.
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
├── prod-deploy.sh  pin ~/repos-sax/forge-prod to a commit + restart (D103)
├── s3/             bucket lifecycle policy (apply once with aws s3api)
└── README.md
```

## Quick start (first time)

```bash
podman machine start
make stack-up                                     # Postgres + Temporal + UI + MinIO
cp deploy/launchd/envs/prod.env.example ~/.config/forge/envs/prod.env
chmod 600 ~/.config/forge/envs/prod.env           # fill in the CHANGEMEs (per-env profiles, T0.9)
make prod-deploy REF=main                         # create + pin ~/repos-sax/forge-prod
~/repos-sax/forge-prod/deploy/launchd/install.sh  # --with-pbook / --with-ocr / --with-backup
open http://localhost:8233                        # Temporal UI
```

The installer must be run from the production checkout: it renders every
plist path from its own location, so that copy decides which checkout the
agents execute.

## Deploying afterwards

```bash
make prod-deploy REF=<ref>      # re-pin forge-prod to a commit, sync, restart
```

That is the whole deploy. Workers refuse to start on a dirty or
unverifiable checkout (exit 78), so production always runs a commit.

## History

The predecessor design — a single EC2 instance with Terraform, SSM
secret bootstrap, and an nginx mutual-TLS gateway for remote CLIs — was
removed by D99 (2026-07-16) and survives in git history.
