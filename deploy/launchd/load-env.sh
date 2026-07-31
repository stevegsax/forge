# load-env.sh — SOURCED (not executed) by run-worker.sh (bash). No shebang, no
# exec bit. It avoids bashisms zsh would reject, which was load-bearing while
# the nightly backup job (zsh) sourced it too; that job retired with forge's
# local stack (T10.1/D104), so the portability is now free insurance, not a
# requirement.
#
# Contract (T0.9 explicit-environment guard):
#   1. FORGE_ENV must already be set in the process environment. Workers receive
#      it from their launchd plist; an interactive shell must export it. There
#      is no default — an unset FORGE_ENV aborts loudly.
#   2. The profile $XDG_CONFIG_HOME/forge/envs/$FORGE_ENV.env is loaded
#      line-by-line as KEY=VALUE and NEVER shell-evaluated (T0.7/G35): a value
#      containing `&`, `;`, `$(...)` is inert.
#   3. After loading, the profile's FORGE_ENV_TAG must equal FORGE_ENV.
# It DELIBERATELY never sets FORGE_PROD_ACK — that acknowledgement comes only
# from the plist (or interactive shell), so a profile can never by itself grant
# production access.

if [ -z "${FORGE_ENV:-}" ]; then
  echo "load-env.sh: FORGE_ENV is unset — there is no default environment." >&2
  echo "  Workers receive FORGE_ENV from their launchd plist's" >&2
  echo "  EnvironmentVariables; an interactive shell must export it (prod/dev/test)." >&2
  exit 78  # EX_CONFIG
fi

case "$FORGE_ENV" in
  prod|dev|test) ;;
  *)
    echo "load-env.sh: FORGE_ENV=$FORGE_ENV is not valid (expected prod|dev|test)." >&2
    exit 78
    ;;
esac

_env_file="${XDG_CONFIG_HOME:-$HOME/.config}/forge/envs/$FORGE_ENV.env"
if [ ! -f "$_env_file" ]; then
  echo "load-env.sh: $_env_file not found — copy" >&2
  echo "  deploy/launchd/envs/$FORGE_ENV.env.example there and chmod 600." >&2
  exit 78
fi

_perms="$(stat -f '%Lp' "$_env_file")"
if [ "$_perms" != "600" ]; then
  echo "load-env.sh: $_env_file must be chmod 600 (is $_perms) — it holds API keys." >&2
  exit 78
fi

# Parse KEY=VALUE without shell-evaluating values (the T0.7/G35 property).
while IFS= read -r line || [ -n "$line" ]; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue ;; esac
  key="${line%%=*}"
  value="${line#*=}"
  case "$key" in
    [A-Za-z_]*)
      # further-char check below
      ;;
    *)
      echo "load-env.sh: malformed line in $_env_file (not KEY=VALUE): $key" >&2
      exit 78
      ;;
  esac
  if [ -n "$(printf '%s' "$key" | tr -d 'A-Za-z0-9_')" ]; then
    echo "load-env.sh: malformed key in $_env_file (not KEY=VALUE): $key" >&2
    exit 78
  fi
  export "$key=$value"
done < "$_env_file"

# The profile must agree with the declared environment (T0.9 tag check). An
# empty/absent tag also fails here — prod/dev profiles both declare one.
if [ "${FORGE_ENV_TAG:-}" != "$FORGE_ENV" ]; then
  echo "load-env.sh: FORGE_ENV_TAG=${FORGE_ENV_TAG:-<unset>} in $_env_file" >&2
  echo "  does not match FORGE_ENV=$FORGE_ENV. Source the $FORGE_ENV profile," >&2
  echo "  or correct FORGE_ENV." >&2
  exit 78
fi

unset _env_file _perms
