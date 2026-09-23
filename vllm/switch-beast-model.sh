#!/usr/bin/env bash
# Switch Beast's shared OpenClaw endpoint between the two validated Qwen3.5-4B models.
set -euo pipefail

UNIT=openclaw-vllm.service
DROPIN_DIR="$HOME/.config/systemd/user/${UNIT}.d"
DROPIN="$DROPIN_DIR/model-switch.conf"
API=http://127.0.0.1:11434
QUANTRIO_REVISION=32c292e3a73afe1138518180b1b6d2868c980ee2
QUANTRIO_SNAPSHOT="$HOME/.cache/huggingface/hub/models--QuantTrio--Qwen3.5-4B-AWQ/snapshots/$QUANTRIO_REVISION"

usage() {
  echo "Usage: $0 status | use quanttrio | use hauhaucs" >&2
  exit 2
}

[[ "$(hostname -s)" == beast ]] || {
  echo "Run this command on Beast (current host: $(hostname -s))." >&2
  exit 1
}

case "${1:-}" in
  status)
    if [[ -f "$DROPIN" ]] && grep -q -- '--revision '"$QUANTRIO_REVISION" "$DROPIN"; then
      echo "Selected model: QuantTrio/Qwen3.5-4B-AWQ"
    else
      echo "Selected model: HauhauCS Qwen3.5-4B-VL W4A16"
    fi
    curl -fsS --max-time 5 "$API/v1/models"
    echo
    systemctl --user is-active "$UNIT"
    ;;
  use)
    [[ $# == 2 ]] || usage
    case "$2" in
      quanttrio)
        [[ -f "$QUANTRIO_SNAPSHOT/config.json" ]] || {
          echo "QuantTrio checkpoint revision $QUANTRIO_REVISION is not cached." >&2
          exit 1
        }
        mkdir -p "$DROPIN_DIR"
        temp="$DROPIN.tmp.$$"
        cat >"$temp" <<EOF
[Service]
ExecStart=
ExecStart=/usr/bin/docker run --rm --name openclaw-vllm --gpus all --ipc=host -p \${OPENCLAW_PORT}:8000 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -v %h/.cache/huggingface:/root/.cache/huggingface \${VLLM_IMAGE} QuantTrio/Qwen3.5-4B-AWQ --revision $QUANTRIO_REVISION --served-model-name openclaw --language-model-only --default-chat-template-kwargs '{"enable_thinking":false}' --max-model-len 6144 --max-num-seqs 4 --max-num-batched-tokens 1024 --gpu-memory-utilization 0.95 --kv-cache-dtype fp8 --kv-cache-memory 1200000000 --enforce-eager --enable-auto-tool-choice --tool-call-parser qwen3_coder
EOF
        chmod 0644 "$temp"
        mv -f "$temp" "$DROPIN"
        ;;
      hauhaucs)
        rm -f "$DROPIN"
        rmdir "$DROPIN_DIR" 2>/dev/null || true
        ;;
      *) usage ;;
    esac
    systemctl --user daemon-reload
    echo "Restarting $UNIT with $2..."
    systemctl --user restart "$UNIT"
    for _ in $(seq 1 180); do
      if curl -fsS --max-time 2 "$API/v1/models" >/dev/null 2>&1; then
        echo "Ready: $2 model is serving under API alias openclaw."
        curl -fsS --max-time 5 "$API/v1/models"
        echo
        exit 0
      fi
      sleep 2
    done
    echo "The endpoint did not become ready. Check: journalctl --user -u $UNIT -n 100 --no-pager" >&2
    exit 1
    ;;
  *) usage ;;
esac
