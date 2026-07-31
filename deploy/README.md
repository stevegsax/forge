# Forge deployment: local-first on an always-on desktop

Forge deploys to the operator's always-on macOS desktop (D99). It runs
**no infrastructure of its own** (D104): Temporal comes from the
`~/repos-sax/sax-temporal` stacks (dev `:7236`, prod `:7243`) and
Postgres from the `~/repos-sax/sax-datastores` stacks (dev `:5432`, prod
`:5442`); both boot themselves, and sax-datastores' nightly backup covers
every database on its instances — forge's and Temporal's alike, since the
Temporal servers persist there too. What lives here is the
worker supervision: the forge/pbook/ocr workers run as launchd-supervised
host processes out of a **commit-pinned checkout** (`~/repos-sax/forge-prod`,
deployed only by `prod-deploy.sh` — D103). Blobs are in S3. There is no
remote access — every service binds to loopback only.

The full walkthrough (process, configuration, gotchas) is
[../docs/operations/DEPLOYMENT.md](../docs/operations/DEPLOYMENT.md).

## Contents

```text
deploy/
├── launchd/        worker supervision: install.sh, run-worker.sh,
│                   load-env.sh, plist template, envs/*.env.example
├── prod-deploy.sh  pin ~/repos-sax/forge-prod to a commit + restart (D103)
├── prod-ocr        run one ocr CLI command against prod, from that checkout
├── s3/             blob-bucket lifecycle policy (apply once with aws s3api)
└── README.md
```

## Quick start (first time)

Prerequisite: the shared stacks are up (`sax-datastores` and `sax-temporal`
own that, and forge's `forge_prod` role/database must already be
provisioned there).

```bash
cp deploy/launchd/envs/prod.env.example ~/.config/forge/envs/prod.env
chmod 600 ~/.config/forge/envs/prod.env           # fill in the CHANGEMEs (per-env profiles, T0.9)
make prod-deploy REF=main                         # create + pin ~/repos-sax/forge-prod
~/repos-sax/forge-prod/deploy/launchd/install.sh  # --with-pbook / --with-ocr
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

Forge's own podman stack (`deploy/local-stack/`: Postgres + MinIO +
Temporal + UI), its `forge-stack` boot agent, and its nightly `db-backup`
agent were removed by D104 (2026-07-31), when the data and workflow layers
moved to the shared stacks. They too survive in git history.
