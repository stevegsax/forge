# Secure remote access to Temporal over the internet (mTLS)

This document describes how remote users run the `forge`, `ocr`, and `pbook`
CLIs from their own machines against Temporal on the EC2 instance, securely,
over the public internet. It is the security companion to
[DEPLOYMENT.md](DEPLOYMENT.md) and the concrete artifacts in
[`deploy/`](../../deploy/).

It supersedes DEPLOYMENT.md's original "no inbound ports; reach Temporal only via
an SSM tunnel" stance for the case where users must submit work directly over the
internet.

## Goal & threat model

**Goal.** Authorized users, on laptops anywhere, submit Temporal workflows to the
instance and retrieve results. Nobody else can reach the orchestration plane, and
traffic is confidential and integrity-protected in transit.

**What we defend.** The Temporal frontend is the crown jewel: anyone who can call
it can start/terminate/query workflows and read inputs and results.
**Temporal OSS ships with no authentication of its own** — exposing port 7233
directly to the internet would let anyone who can reach it drive the cluster.

**In scope:** authenticating remote CLIs; encrypting client↔server traffic;
restricting the network surface. **Out of scope (here):** per-user authorization
(everyone authenticated is fully authorized — see [Authorization](#authorization)),
DDoS protection, and securing the AWS account itself.

## Design: an mTLS gateway in front of Temporal

Remote CLIs connect to an **nginx gRPC reverse proxy** that terminates **mutual
TLS** on port 443 and forwards to the Temporal frontend on the private network.

```
client (cert) ──TLS/mTLS:443──► nginx gateway ──plaintext:7233──► temporal (private)
                                  │ ssl_verify_client on
                                  │ ssl_client_certificate = client-ca.crt
                                  └ ssl_certificate = server.crt (signed by server-ca)
```

Two properties give us the guarantee:

1. **Encryption** — TLS 1.2/1.3 between client and gateway.
2. **Authentication** — `ssl_verify_client on` makes the gateway **require** a
   client certificate signed by our **client CA**. No cert (or one from any other
   issuer) ⇒ the TLS handshake fails before a single gRPC call reaches Temporal.
   Possession of a CA-signed client cert *is* "authorized user."

Temporal's frontend (7233) and UI (8080) stay **loopback-only**; the security
group opens **only 443**. The on-box workers connect to `127.0.0.1:7233` in
plaintext (loopback, never exposed), so they need no certificates.

### Why a gateway rather than mTLS on the Temporal frontend

Temporal *can* terminate mTLS itself (see [Alternative](#alternative-terminate-mtls-on-the-temporal-frontend)).
We chose a dedicated gateway because:

- **One auditable boundary.** The entire access-control policy is a handful of
  `ssl_*` directives in one file. With native termination the same guarantee is
  spread across the frontend server config, the internode config, the *internal*
  frontend client (the system worker and health checks must also present certs or
  the cluster won't boot), and the Web UI — five places, each a chance to
  fail-open or fail-to-start.
- **Temporal stays simple and proven.** The frontend runs exactly as in the
  existing single-host design (localhost, no TLS). Workers and the UI are
  unaffected.
- **Cert rotation is decoupled** from the Temporal cluster — swap the gateway's
  certs and restart one container.

The trade-off is one extra component (nginx) and gRPC proxy tuning (long-poll
timeouts; see `deploy/temporal/nginx/temporal-grpc.conf`). **The client side is
identical either way** — the exact Temporal SDK `TLSConfig` mTLS (server CA +
client cert/key), so nothing about the CLI experience or the code below depends on
this choice, and you can move to native termination later without touching
clients.

## Client side (the code change)

Both CLIs now build their Temporal connection through a single helper
(`forge/temporal_client.py`, `pbook/temporal_client.py`) that reads TLS settings
from the environment and passes a `temporalio.service.TLSConfig` to
`Client.connect`. Every connect site (CLI commands and the worker) routes through
it, so TLS is configured in exactly one place per repo.

| Variable (`FORGE_…` / `PBOOK_…`) | Meaning |
|---|---|
| `…_TEMPORAL_TLS` | `1` to enable TLS. Unset ⇒ plaintext (the worker's loopback default). |
| `…_TEMPORAL_TLS_SERVER_CA` | PEM of the **server CA** — verifies the gateway. |
| `…_TEMPORAL_TLS_CLIENT_CERT` | PEM of the user's client certificate (mTLS). |
| `…_TEMPORAL_TLS_CLIENT_KEY` | PEM of the user's private key (mTLS). |
| `…_TEMPORAL_TLS_SERVER_NAME` | Override the expected server name (set when dialing by IP). |

Plaintext remains the default, so co-located workers (`127.0.0.1:7233`) are
unaffected; only remote clients set `…_TEMPORAL_TLS=1`. See
[`deploy/client/ONBOARDING.md`](../../deploy/client/ONBOARDING.md).

## Certificates

A **two-CA** model (issuance scripts and full detail in
[`deploy/certs/`](../../deploy/certs/README.md)):

- **server CA** signs the gateway's `server.crt`; **clients** trust it.
- **client CA** signs each user's cert; the **gateway** trusts it.

Both CA private keys stay offline with the operator and never touch the instance.
The instance holds only `server.crt`, `server.key`, and `client-ca.crt`, fetched
from SSM Parameter Store (SecureString) at boot via the instance role — no static
secrets on disk in the AMI or repo.

## Authorization

Per the deployment decision, **authentication equals full authorization**: any
user with a valid client certificate may submit and read any workflow. This fits
a small, trusted team. The certificate CN is forwarded to Temporal
(`X-Forwarded-Client-CN`) for **audit**.

To evolve later without re-architecting:

- **Namespace isolation** — run separate Temporal namespaces (e.g. one per team
  or per app) and issue certs/route per namespace.
- **Role-based authorization** — add a Temporal
  [Authorizer + ClaimMapper](https://docs.temporal.io/self-hosted-guide/security#authorization)
  that maps the client-cert subject to permissions (e.g. read-only vs submit).
  The gateway already passes the authenticated identity downstream.

## Operations

- **Add a user:** `deploy/certs/gen-client-cert.sh <name>`; deliver cert+key plus
  `server-ca.crt`.
- **Rotate server cert:** re-issue, update the three SSM params, restart the
  gateway container.
- **Revoke:** for a small team, rotate the client CA and re-issue everyone; or run
  a CRL (`ssl_crl` in the gateway config). Keep client certs short-lived
  (default 1 year) so leaks age out.
- **Manage the box:** SSM Session Manager only (no SSH). Reach the Temporal UI via
  an SSM port-forward to `127.0.0.1:8080`.

## Residual risks & hardening

- **Pre-auth network surface.** Even with mTLS, the TLS handshake is reachable
  from `allowed_client_cidrs`. Narrow that list to known office/VPN egress where
  practical; mTLS is the gate, IP-allowlisting is defense in depth.
- **No per-cert revocation by default.** Until a CRL/OCSP is wired in, distribution
  control + short lifetimes are the mitigation.
- **Stolen client key.** Treat `*.key` as a credential; short lifetimes bound the
  blast radius; rotate on suspicion.
- **gRPC through a proxy.** Long-poll timeouts are tuned; if you see dropped polls,
  revisit `grpc_read_timeout`/keepalive, or switch to an NLB (TCP passthrough) +
  native termination.

## Alternative: terminate mTLS on the Temporal frontend

If you prefer no gateway, configure the auto-setup/server image directly (verified
against Temporal's `config_template.yaml`). Use **one CA for the cluster** and add
your **user client CA** as a second accepted client CA:

```yaml
# docker env on the temporal service
TEMPORAL_TLS_REQUIRE_CLIENT_AUTH: "true"
TEMPORAL_TLS_SERVER_CERT: /certs/cluster.crt   # frontend+internode server cert
TEMPORAL_TLS_SERVER_KEY:  /certs/cluster.key
TEMPORAL_TLS_SERVER_CA_CERT: /certs/cluster-ca.crt   # internode trust
TEMPORAL_TLS_FRONTEND_CERT: /certs/cluster.crt
TEMPORAL_TLS_FRONTEND_KEY:  /certs/cluster.key
TEMPORAL_TLS_CLIENT1_CA_CERT: /certs/cluster-ca.crt  # internal callers' CA
TEMPORAL_TLS_CLIENT2_CA_CERT: /certs/user-client-ca.crt  # YOUR remote users' CA
TEMPORAL_TLS_INTERNODE_SERVER_NAME: tls-sample
TEMPORAL_TLS_FRONTEND_SERVER_NAME:  tls-sample
# Web UI + temporal CLI/health checks must ALSO present the cluster cert:
#   TEMPORAL_CLI_TLS_CERT/KEY/CA, TEMPORAL_TLS_* on the UI container.
```

The key insight that keeps `requireClientAuth=true` from breaking the cluster:
the internal callers (system worker, health checks, UI) present the **cluster**
cert, which is accepted because `CLIENT1_CA` is the cluster CA; your remote users
present certs from `CLIENT2_CA`. Clients then dial `7233` directly (open it in the
SG) with the same SDK `TLSConfig` variables above — no other client change. This
is more native but spreads the trust config across five places; weigh that against
the single-file gateway.

## References

- Temporal security (mTLS, authorizer): https://docs.temporal.io/self-hosted-guide/security
- Temporal service TLS config: https://docs.temporal.io/temporal-service/configuration
- Python SDK `TLSConfig`: `temporalio.service.TLSConfig`
