# Forge Temporal mTLS — certificate authority & issuance

This directory issues the certificates that secure remote access to Temporal.
The security model is **mutual TLS (mTLS)**: every CLI must present a client
certificate signed by *our* client CA, and it must verify the gateway's server
certificate against *our* server CA. No valid certificate ⇒ no connection.

## Trust model (two CAs)

```text
            server-ca.key  ──signs──►  server.crt   (gateway presents this)
   (operator, offline)                     ▲
                                           │ verified by clients via server-ca.crt
   remote CLI ───────presents──►  alice.crt│
        ▲                                   │
        │ verified by gateway via client-ca.crt
   client-ca.key  ──signs──►  alice.crt, bob.crt, …   (one per user)
   (operator, offline)
```

Two CAs keep the two trust directions independent:

| CA | Signs | Lives | Who trusts it |
| ---- | ------- | ------- | ---------------- |
| **server CA** | the gateway's `server.crt` | offline (operator) | **clients** (`FORGE_TEMPORAL_TLS_SERVER_CA`) |
| **client CA** | each user's `*.crt` | offline (operator) | the **gateway** (`ssl_client_certificate`) |

Both CA *private keys* (`*-ca.key`) stay on a trusted operator workstation or a
vault. They are never copied to the EC2 instance and never committed. The
instance only ever holds: `server.crt`, `server.key`, and `client-ca.crt`.

## One-time setup

```bash
cd deploy/certs
./gen-ca.sh                              # -> ca/{server,client}-ca.{crt,key}
./gen-server-cert.sh temporal.example.com   # -> server/server.{crt,key}
# (use your public DNS name, or the EIP, e.g. ./gen-server-cert.sh 203.0.113.10)
```

Upload the instance-side material to SSM Parameter Store (SecureString):

```bash
aws ssm put-parameter --type SecureString --name /forge/TLS_SERVER_CERT \
  --value "$(cat server/server.crt)"
aws ssm put-parameter --type SecureString --name /forge/TLS_SERVER_KEY \
  --value "$(cat server/server.key)"
aws ssm put-parameter --type SecureString --name /forge/TLS_CLIENT_CA \
  --value "$(cat ca/client-ca.crt)"
```

The instance's `fetch-secrets.sh` writes these to `/etc/forge/certs/` at boot,
where the gateway container mounts them.

## Per-user issuance

```bash
./gen-client-cert.sh alice               # -> clients/alice.{crt,key}
```

Deliver `clients/alice.crt`, `clients/alice.key`, and `ca/server-ca.crt` to
Alice over a secure channel (e.g. encrypted file transfer, a password manager
item). She follows [../client/ONBOARDING.md](../client/ONBOARDING.md).

## Rotation

- **Client certs** default to 1 year (`CLIENT_DAYS`). Re-issue before expiry.
- **Server cert** defaults to 825 days (`SERVER_DAYS`). Re-issue, re-upload to
  SSM, and restart the gateway (`docker compose restart temporal-gateway`).
- **CAs** default to ~10 years. Rotating a CA means re-issuing everything it
  signed.

## Revocation

For a small team the simplest revocation is **rotate the client CA and re-issue
every user** — the lost/abused cert is then signed by a CA the gateway no longer
trusts. To revoke individuals without re-issuing everyone, run a CRL:

```bash
# Maintain a CRL with the client CA, publish it, and add to the gateway:
#   ssl_crl /etc/nginx/certs/client-ca.crl;   (in temporal-grpc.conf)
```

Until a revocation mechanism is wired in, treat client-cert distribution as the
control point and keep validity short.

## Files (git-ignored)

`ca/`, `server/`, and `clients/` hold private keys — they are listed in
`deploy/.gitignore` and must never be committed.
