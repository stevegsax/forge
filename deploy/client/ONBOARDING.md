# Connecting to remote Forge (client onboarding)

You run the `forge`, `ocr`, and `pbook` commands on **your own computer**; they
submit Temporal workflows that execute on the remote EC2 instance and stream
results back. The connection is secured with **mutual TLS**: you authenticate
with a client certificate, and your CLI verifies the server.

> "ocr" is not a separate program — it's Forge's OCR surface, e.g.
> `forge start OcrSyncWorkflow '{"file_path": "…"}'  --wait` and
> `forge ocr-jobs`. Securing `forge` secures it too.

## 1. Install the CLIs

You need the `forge` and `pbook` Python packages installed locally (same repos
as the server). With `uv`:

```bash
git clone https://github.com/stevegsax/forge && cd forge
uv sync
uv run forge --help
```

## 2. Place the three files your operator sent you

You will receive, over a secure channel:

- `server-ca.crt` — lets your CLI verify the gateway
- `<you>.crt` — your client certificate
- `<you>.key` — your private key (keep it secret)

```bash
mkdir -p ~/.forge/certs && chmod 700 ~/.forge/certs
mv server-ca.crt <you>.crt <you>.key ~/.forge/certs/
chmod 600 ~/.forge/certs/<you>.key
```

## 3. Set the environment

Copy [`client-env.example`](client-env.example), edit the endpoint and paths,
and source it from your shell profile:

```bash
export FORGE_TEMPORAL_ADDRESS="203.0.113.10:443"     # from the operator
export FORGE_TEMPORAL_TLS=1
export FORGE_TEMPORAL_TLS_SERVER_CA="$HOME/.forge/certs/server-ca.crt"
export FORGE_TEMPORAL_TLS_CLIENT_CERT="$HOME/.forge/certs/<you>.crt"
export FORGE_TEMPORAL_TLS_CLIENT_KEY="$HOME/.forge/certs/<you>.key"
export FORGE_TEMPORAL_TLS_SERVER_NAME="temporal.forge.internal"   # when dialing by IP
# pbook uses the same values under PBOOK_* (see client-env.example).
```

Set `FORGE_TEMPORAL_TLS_SERVER_NAME` when you connect by **IP address**: the
gateway's certificate carries the stable name `temporal.forge.internal`, and
pinning it lets TLS host verification pass. If your operator gave you a **DNS
name** that's in the certificate, you can dial that and omit this variable.

## 4. Run

```bash
forge status --limit 5
forge run --task-id demo --description "Add a docstring" --target-file path/to/file.py
forge start OcrSyncWorkflow '{"file_path": "/data/repos/proj/sample.pdf"}' --wait   # "ocr"
pbook list --limit 5
```

## Troubleshooting

- **`connection refused` / handshake failure, certless** — your cert/key env
  vars aren't set or point at the wrong files. All three of CA, cert, key are
  required.
- **`certificate signed by unknown authority`** — `FORGE_TEMPORAL_TLS_SERVER_CA`
  isn't the `server-ca.crt` you were given.
- **`x509: ... doesn't match`/host verification** — set
  `FORGE_TEMPORAL_TLS_SERVER_NAME=temporal.forge.internal` (you're dialing by IP).
- **`PermissionDenied`/handshake rejected** — your client cert may be expired or
  revoked; ask your operator to re-issue (`gen-client-cert.sh`).
- File paths in OCR/`forge run` target the **server's** filesystem (the repos
  checked out on the instance), not your laptop.
