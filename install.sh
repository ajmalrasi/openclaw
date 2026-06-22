#!/usr/bin/env bash
# Provision the shared OpenClaw LLM on this host. Idempotent: safe to re-run
# after editing the Modelfile to swap the underlying model.
#
#   ./install.sh           # create/refresh the "openclaw" model + warmup script
#   ./install.sh --service # also install & start the systemd user service
#
# With --service the service is installed and started FIRST, so `ollama create`
# has a daemon to talk to even when migrating from a different (now-stopped)
# service. Assumes Ollama is installed at ~/.local/ollama/bin/ollama; it does
# NOT install Ollama itself.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLLAMA="${OLLAMA_BIN:-$HOME/.local/ollama/bin/ollama}"
BASE_MODEL="$(grep -m1 '^FROM ' "$HERE/Modelfile" | awk '{print $2}')"
WANT_SERVICE="${1:-}"

echo ">> Installing warmup script ..."
mkdir -p "$HOME/.local/bin"
install -m 0755 "$HERE/bin/openclaw-warmup.sh" "$HOME/.local/bin/openclaw-warmup.sh"

if [[ "$WANT_SERVICE" == "--service" ]]; then
  echo ">> Installing & starting systemd user service (before model create) ..."
  mkdir -p "$HOME/.config/systemd/user"
  install -m 0644 "$HERE/systemd/openclaw.service" "$HOME/.config/systemd/user/openclaw.service"
  systemctl --user daemon-reload
  systemctl --user enable --now openclaw.service
fi

echo ">> Waiting for Ollama to answer on :11434 ..."
for _ in $(seq 1 60); do
  curl -sf http://localhost:11434/api/version >/dev/null 2>&1 && break
  sleep 1
done
curl -sf http://localhost:11434/api/version >/dev/null 2>&1 \
  || { echo "!! Ollama not reachable. Start it (or pass --service) and retry."; exit 1; }

echo ">> Base model: $BASE_MODEL"
if ! "$OLLAMA" list | awk '{print $1}' | grep -qx "$BASE_MODEL"; then
  echo ">> Pulling $BASE_MODEL ..."
  "$OLLAMA" pull "$BASE_MODEL"
fi

echo ">> Creating/refreshing the 'openclaw' model alias ..."
"$OLLAMA" create openclaw -f "$HERE/Modelfile"

if [[ "$WANT_SERVICE" == "--service" ]]; then
  echo ">> Pinning the model resident ..."
  bash "$HOME/.local/bin/openclaw-warmup.sh" || true
fi

echo ">> Done. Apps should use: base_url=http://<host>:11434  model=openclaw"
