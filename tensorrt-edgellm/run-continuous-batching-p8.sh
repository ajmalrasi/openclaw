#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

# Controlled rollout: candidate activation, live checks, restart, rollback, and
# final reactivation. Run remotely under one external five-minute deadline.
set -euo pipefail

task_dir=${P8_TASK_DIR:-/home/ajmalrasi/continuous-batching-p8-20260916}
candidate_dir=/home/ajmalrasi/continuous-batching-p7-20260915
reference_dir=/home/ajmalrasi/TensorRT-Edge-LLM
engine_dir=/home/ajmalrasi/Qwen3.5-4B/engines/llm-b2-input6144-kv8192-vanilla
checkpoint_dir=/home/ajmalrasi/Qwen3.5-4B/quantized
model_unit=openclaw-tensorrt-edgellm.service
timer_unit=openclaw-tensorrt-watchdog.timer
watchdog_unit=openclaw-tensorrt-watchdog.service
override_dir=/home/ajmalrasi/.config/systemd/user/openclaw-tensorrt-edgellm.service.d
override_file="$override_dir/continuous-batching-p8.conf"
validator=/tmp/validate-continuous-batching-p8.py

test ! -e "$task_dir"
test ! -e "$override_file"
test -r "$candidate_dir/build/_edgellm_runtime.cpython-312-aarch64-linux-gnu.so"
test -r "$candidate_dir/source/experimental/server/runtime/engine.py"
test -r "$validator"
systemctl --user is-active --quiet "$model_unit"

timer_active=false
if systemctl --user is-active --quiet "$timer_unit"; then
    timer_active=true
fi

mkdir -p "$task_dir/evidence"
cp "$validator" "$task_dir/validate.py"
exec > >(tee "$task_dir/evidence/rollout.log") 2>&1
date -Is
curl --fail --silent --max-time 10 http://127.0.0.1:11434/v1/models > "$task_dir/evidence/before-models.json"
curl --fail --silent --max-time 10 http://127.0.0.1:11434/health > "$task_dir/evidence/before-health.json"
jq -e '.status == "healthy" and .active_requests == 0 and .queued_requests == 0 and .capabilities.max_num_seqs == 1' \
    "$task_dir/evidence/before-health.json"
systemctl --user cat "$model_unit" > "$task_dir/evidence/before-unit.txt"
systemctl --user show "$model_unit" > "$task_dir/evidence/before-show.txt"

rollback() {
    status=$?
    trap - ERR INT TERM EXIT
    set +e
    echo "ROLLBACK begin status=$status"
    rm -f "$override_file"
    rmdir "$override_dir" 2>/dev/null || true
    systemctl --user daemon-reload
    systemctl --user stop "$model_unit"
    systemctl --user reset-failed "$model_unit"
    systemctl --user start "$model_unit"
    for ((attempt = 0; attempt < 90; ++attempt)); do
        if curl --fail --silent --max-time 2 http://127.0.0.1:11434/health > "$task_dir/evidence/failure-restored-health.json" && \
            jq -e '.status == "healthy" and .capabilities.max_num_seqs == 1' "$task_dir/evidence/failure-restored-health.json"; then
            break
        fi
        sleep 2
    done
    if "$timer_active"; then
        systemctl --user reset-failed "$timer_unit"
        systemctl --user start "$timer_unit"
    fi
    systemctl --user is-active --quiet "$model_unit" || true
    if "$timer_active"; then
        systemctl --user is-active --quiet "$timer_unit" || true
    fi
    echo "ROLLBACK complete status=$status"
    exit "$status"
}
trap rollback ERR INT TERM

cp -a "$candidate_dir/source" "$task_dir/release"
mkdir -p "$task_dir/build"
cp -a "$candidate_dir/build/_edgellm_runtime.cpython-312-aarch64-linux-gnu.so" "$task_dir/build/"
cp -a "$candidate_dir/build/continuous_batching_probe" "$task_dir/build/"
sha256sum "$task_dir/build/_edgellm_runtime.cpython-312-aarch64-linux-gnu.so" > "$task_dir/evidence/runtime.sha256"
grep -q 'EDGELLM_CONTINUOUS_GRAPHS' "$task_dir/release/experimental/server/runtime/engine.py"

write_candidate_override() {
    mkdir -p "$override_dir"
    cat > "$override_file" <<EOF
[Service]
WorkingDirectory=$task_dir/release
Environment=EDGELLM_PYBIND_DIR=$task_dir/build
Environment=EDGELLM_CONTINUOUS_BATCHING=1
Environment=EDGELLM_CONTINUOUS_GRAPHS=1
Environment=EDGELLM_PLUGIN_PATH=$reference_dir/build/libNvInfer_edgellm_plugin.so
ExecStart=
ExecStart=$reference_dir/.venv/bin/python -m experimental.server $checkpoint_dir --engine-dir $engine_dir --host 0.0.0.0 --port 11434 --served-model-name openclaw --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice --max-queued-requests 8 --queue-timeout 30
EOF
}

wait_healthy() {
    expected_seqs=$1
    result_file=$2
    for ((attempt = 0; attempt < 90; ++attempt)); do
        if curl --fail --silent --max-time 2 http://127.0.0.1:11434/health > "$result_file" && \
            jq -e --argjson expected "$expected_seqs" '.status == "healthy" and .capabilities.max_num_seqs == $expected' "$result_file"; then
            return 0
        fi
        sleep 2
    done
    return 1
}

start_candidate() {
    write_candidate_override
    systemctl --user daemon-reload
    systemctl --user stop "$model_unit"
    systemctl --user reset-failed "$model_unit"
    systemctl --user start "$model_unit"
    wait_healthy 2 "$1"
}

systemctl --user stop "$timer_unit" "$watchdog_unit"
start_candidate "$task_dir/evidence/candidate-health-initial.json"
"$reference_dir/.venv/bin/python" "$task_dir/validate.py" | tee "$task_dir/evidence/live-initial.log"

systemctl --user restart "$model_unit"
wait_healthy 2 "$task_dir/evidence/candidate-health-restart.json"
"$reference_dir/.venv/bin/python" "$task_dir/validate.py" | tee "$task_dir/evidence/live-restart.log"

rm -f "$override_file"
rmdir "$override_dir" 2>/dev/null || true
systemctl --user daemon-reload
systemctl --user stop "$model_unit"
systemctl --user reset-failed "$model_unit"
systemctl --user start "$model_unit"
wait_healthy 1 "$task_dir/evidence/rollback-health.json"
curl --fail --silent --max-time 10 http://127.0.0.1:11434/v1/models > "$task_dir/evidence/rollback-models.json"
echo "P8_ROLLBACK_GATE passed"

start_candidate "$task_dir/evidence/candidate-health-final.json"
"$reference_dir/.venv/bin/python" "$task_dir/validate.py" | tee "$task_dir/evidence/live-final.log"
systemctl --user start "$watchdog_unit"
test "$(systemctl --user show -p Result --value "$watchdog_unit")" = success
if "$timer_active"; then
    systemctl --user reset-failed "$timer_unit"
    systemctl --user start "$timer_unit"
    systemctl --user is-active --quiet "$timer_unit"
fi
systemctl --user is-active --quiet "$model_unit"
curl --fail --silent --max-time 10 http://127.0.0.1:11434/health > "$task_dir/evidence/final-health.json"
jq -e '.status == "healthy" and .active_requests == 0 and .queued_requests == 0 and .capabilities.max_num_seqs == 2' \
    "$task_dir/evidence/final-health.json"
date -Is
echo "P8_DEPLOYMENT_GATE passed"
