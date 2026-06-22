#!/usr/bin/env bash
# Provision the shared OpenClaw LLM on this host. Idempotent: safe to re-run
# after editing the Modelfile to swap the underlying model.
#
#   ./install.sh           # create/refresh the "openclaw" model + warmup script
#   ./install.sh --service # also install & start the systemd user service
#
# Assumes Ollama is already installed at ~/.local/ollama/bin/ollama (as on the
# Jetson). It does NOT install Ollama itself.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLLAMA="${OLLAMA_BIN:-$HOME/.local/ollama/bin/ollama}"
BASE_MODEL="$(grep -m1 '^FROM ' "$HERE/Modelfile" | awk '{print $2}')"

echo ">> Base model: $BASE_MODEL"
if ! "$OLLAMA" list | awk '{print $1}' | grep -qx "$BASE_MODEL"; then
  echo ">> Pulling $BASE_MODEL ..."
  "$OLLAMA" pull "$BASE_MODEL"
fi

echo ">> Creating/refreshing the 'openclaw' model alias ..."
"$OLLAMA" create openclaw -f "$HERE/Modelfile"

echo ">> Installing warmup script ..."
mkdir -p "$HOME/.local/bin"
install -m 0755 "$HERE/bin/openclaw-warmup.sh" "$HOME/.local/bin/openclaw-warmup.sh"

if [[ "${1:-}" == "--service" ]]; then
  echo ">> Installing systemd user service ..."
  mkdir -p "$HOME/.config/systemd/user"
  install -m 0644 "$HERE/systemd/openclaw.service" "$HOME/.config/systemd/user/openclaw.service"
  systemctl --user daemon-reload
  systemctl --user enable --now openclaw.service
  echo ">> openclaw.service started."
fi

echo ">> Done. Apps should use: base_url=http://<host>:11434  model=openclaw"
