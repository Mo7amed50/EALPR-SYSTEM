#!/usr/bin/env bash
set -euo pipefail

# Use the saved local token for developer machines, or inject API_KEY_21ST in CI.
# This avoids interactive `21st login` in scripts and automation.
# Examples:
#   ./scripts/21st.sh whoami
#   API_KEY_21ST=... ./scripts/21st.sh search "card design"
#   21st --api-key "$API_KEY_21ST" search "card design"
if [ -n "${API_KEY_21ST:-}" ]; then
  exec 21st "$@" --api-key "$API_KEY_21ST"
fi

if [ -n "${TWENTYFIRST_TOKEN:-}" ]; then
  exec 21st "$@" --api-key "$TWENTYFIRST_TOKEN"
fi

if [ -t 0 ]; then
  echo "No API_KEY_21ST or TWENTYFIRST_TOKEN set; falling back to local saved login." >&2
fi

exec 21st "$@"
