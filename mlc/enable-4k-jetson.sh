#!/usr/bin/env bash
# enable-4k-jetson.sh — Step 1 of enabling a 4096-token context on the Jetson.
#
# Root cause of the 2048 cap: the 4B model needs ~3.73 GB resident for a 4096 KV
# cache, but the Tegra CUDA allocator only uses *genuinely free* RAM — it will not
# reclaim the ~2.2 GB sitting in page cache. Freeing that cache right before the
# model loads gives it the headroom to fit 4096.
#
# This script installs a tiny root-owned helper that drops the page cache, plus a
# narrow sudoers rule so the (non-root) MLC service can call ONLY that helper
# without a password. It changes nothing about the running model yet — after this
# succeeds, the service wrapper gets wired to call it on startup.
#
# Run ON THE JETSON:  bash enable-4k-jetson.sh   (it will prompt for sudo once)
set -euo pipefail

HELPER=/usr/local/bin/openclaw-drop-caches
SUDOERS=/etc/sudoers.d/openclaw-drop-caches

echo "[1/3] Installing cache-drop helper at $HELPER ..."
sudo tee "$HELPER" >/dev/null <<'EOF'
#!/bin/sh
# Flush dirty pages, then drop clean page/dentry/inode cache so the CUDA
# allocator sees the freed RAM as genuinely free before the model loads.
sync
echo 3 > /proc/sys/vm/drop_caches
EOF
sudo chmod 755 "$HELPER"

echo "[2/3] Granting $USER passwordless sudo for ONLY that helper ..."
echo "$USER ALL=(root) NOPASSWD: $HELPER" | sudo tee "$SUDOERS" >/dev/null
sudo chmod 440 "$SUDOERS"
# Validate the sudoers file so a typo can never lock you out of sudo.
sudo visudo -cf "$SUDOERS"

echo "[3/3] Verifying passwordless invocation + measuring what it frees ..."
before=$(awk '/^MemFree:/{print $2}' /proc/meminfo)
sudo -n "$HELPER"
after=$(awk '/^MemFree:/{print $2}' /proc/meminfo)
freed_mb=$(( (after - before) / 1024 ))

echo
echo "DROP_OK — helper runs without a password."
printf 'Free RAM: %d MB -> %d MB  (reclaimed ~%d MB)\n' \
  $(( before/1024 )) $(( after/1024 )) "$freed_mb"
echo
echo "Next: tell Claude 'DROP_OK, freed ~${freed_mb}MB' and it will run the 4096"
echo "load test over SSH before touching your live service."
