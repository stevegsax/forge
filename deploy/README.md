# Forge deployment: internet-facing, mTLS-secured EC2

This directory provisions a single EC2 instance that hosts **Temporal** and the
**Forge/pbook workers**, with Temporal's and Forge's state in **Supabase
Postgres** and OCR blobs in **S3**. Remote users run the `forge`, `ocr`, and
`pbook` CLIs from their laptops; the connection to Temporal crosses the internet
and is gated by **mutual TLS** — only a client holding a certificate signed by
your internal CA can connect, and all traffic is encrypted.

See [../docs/operations/SECURE-REMOTE-ACCESS.md](../docs/operations/SECURE-REMOTE-ACCESS.md)
for the threat model and design rationale, and
[../docs/operations/DEPLOYMENT.md](../docs/operations/DEPLOYMENT.md) for the broader
Supabase/S3 deployment context.

## Architecture

```text
  Remote laptops                          EC2 (Amazon Linux 2023)
  forge / ocr / pbook                ┌───────────────────────────────────────────┐
  + client cert (mTLS)               │ docker compose (deploy/temporal)            │
        │                            │  ┌─ temporal-gateway (nginx) :443 ◄─ mTLS ──┼── internet (SG:443)
        │  TLS 443 (gRPC, mTLS)      │  │     verifies client cert vs client-ca    │
        └────────────────────────────► │     grpc_pass → temporal:7233            │
                                     │  ├─ temporal  (frontend, private net only)  │
                                     │  └─ temporal-ui :8080 (loopback; SSM tunnel)│
                                     │ systemd: forge-worker@1/@2, pbook-worker    │
                                     │   → 127.0.0.1:7233 (plaintext, loopback)    │
                                     └───────────┬───────────────────┬────────────┘
                                       TLS 5432  │                   │ HTTPS / AWS
                                                 ▼                   ▼
                                         Supabase Postgres     S3 · Anthropic · Mistral · GitHub
```

The **only** internet-exposed port is **443** on the nginx gateway, and it
rejects any client without a CA-signed certificate. Temporal's own gRPC frontend
(7233) and UI (8080) are loopback-only. The workers are on-box clients that talk
to Temporal over loopback in plaintext — they never traverse the gateway.

## Directory map

| Path | What |
| ------ | ------ |
| `certs/` | Internal CA + server/client certificate issuance (`gen-*.sh`) |
| `temporal/compose.yaml` | Temporal frontend + UI + the nginx mTLS gateway |
| `temporal/nginx/temporal-grpc.conf` | The mTLS gateway (the security boundary) |
| `systemd/` | `forge-worker@.service`, `pbook-worker.service` |
| `scripts/` | `bootstrap-instance.sh`, `fetch-secrets.sh`, `smoke-test.sh` |
| `terraform/` | EC2 + EIP + SG (443 only) + scoped IAM + private S3 |
| `client/` | End-user onboarding + env template |
| `local-database/` | **Local dev only** (not part of the EC2 deploy): podman-managed Postgres + MinIO (S3) for running Forge against real store/blob surfaces locally — see `local-database/README.md` and the repo-root `Makefile` `db-*` targets |

## Order of operations

1. **Certs (operator workstation):**
   `cd certs && ./gen-ca.sh && ./gen-server-cert.sh <dns-or-ip>`
2. **Secrets → SSM:** put the API keys, Supabase creds, and the TLS material
   (`TLS_SERVER_CERT`, `TLS_SERVER_KEY`, `TLS_CLIENT_CA`) under `/forge/*`
   (see `certs/README.md`).
3. **Provision:** `cd terraform && cp terraform.tfvars.example terraform.tfvars`
   (edit), then `terraform init && terraform apply`. Note the
   `gateway_endpoint` output. If you issued the server cert before you knew the
   EIP, re-issue it for the EIP/DNS now and update SSM.
4. **Issue user certs:** `cd certs && ./gen-client-cert.sh <name>` per user;
   distribute with `server-ca.crt`.
5. **Users:** follow `client/ONBOARDING.md`.
6. **Verify:** on the instance (via SSM), run `scripts/smoke-test.sh`.

## Notes

- `terraform apply` provisions infra and boots the app via user-data; it does
  **not** create the CA or secrets — those are deliberately operator-controlled
  and never in Terraform state.
- Management is via **SSM Session Manager** (no SSH, no key pair). Reach the
  Temporal UI with an SSM port-forward to `127.0.0.1:8080`.
- This is a single-instance, small-team design. See SECURE-REMOTE-ACCESS.md for
  scaling and the alternative of terminating mTLS on the Temporal frontend.
