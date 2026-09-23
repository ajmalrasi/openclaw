#!/usr/bin/env bash
set -euo pipefail

# Run on the Jetson as ajmalrasi. Switches only the TensorRT Edge-LLM runtime
# binding; the engine, alias, port, and base service remain unchanged.
readonly home_dir=/home/ajmalrasi
readonly service=openclaw-tensorrt-edgellm.service
readonly timer=openclaw-tensorrt-watchdog.timer
readonly watchdog=openclaw-tensorrt-watchdog.service
readonly dropin_dir="$home_dir/.config/systemd/user/$service.d"
readonly active_dropin="$dropin_dir/zz-json-schema-production.conf"
readonly state_dir="$home_dir/.local/state/openclaw/tensorrt-runtime-switch"
readonly schema_backup="$state_dir/zz-json-schema-production.conf"
readonly legacy_binding="$home_dir/continuous-batching-p8-20260916-attempt3/build/_edgellm_runtime.cpython-312-aarch64-linux-gnu.so"
readonly legacy_source="$home_dir/continuous-batching-p8-20260916-attempt3/release"
readonly schema_source="$home_dir/tensorrt-json-schema-release-b921d36/source"
readonly schema_binding="$home_dir/tensorrt-json-schema-release-b921d36/pybind/_edgellm_runtime.cpython-312-aarch64-linux-gnu.so"
readonly engine_dir="$home_dir/Qwen3.5-4B/engines/llm-b2-input6144-kv8192-vanilla"
readonly api=http://127.0.0.1:11434

usage() {
    echo "Usage: $0 legacy|json-schema" >&2
    echo "  legacy      Restore the preserved pre-JSON-Schema P8 binding" >&2
    echo "  json-schema Select the versioned JSON-Schema binding" >&2
    exit 2
}

[[ $# -eq 1 ]] || usage
target=$1
[[ "$target" == legacy || "$target" == json-schema ]] || usage

mkdir -p "$state_dir" "$dropin_dir"
exec 9>"$state_dir/switch.lock"
flock -n 9 || { echo "Another runtime switch is in progress" >&2; exit 1; }

test -s "$legacy_binding"
test -d "$legacy_source"
test -s "$schema_binding"
test -d "$schema_source"
test -d "$engine_dir"
systemctl --user is-active --quiet "$service" || {
    echo "$service must be active before switching" >&2
    exit 1
}
curl --fail --silent --max-time 5 "$api/health" \
    | jq -e '.status == "healthy" and .active_requests == 0 and .queued_requests == 0' >/dev/null

if [[ -e "$active_dropin" ]]; then
    grep -q 'tensorrt-json-schema-release-b921d36' "$active_dropin" || {
        echo "Unrecognized active JSON-Schema drop-in; refusing to overwrite it" >&2
        exit 1
    }
    if [[ -e "$schema_backup" ]]; then
        cmp -s "$active_dropin" "$schema_backup" || {
            echo "Saved JSON-Schema config differs from active config" >&2
            exit 1
        }
    else
        install -m 0644 "$active_dropin" "$schema_backup"
    fi
fi
test -s "$schema_backup"
grep -q 'tensorrt-json-schema-release-b921d36' "$schema_backup"

previous_schema=false
if [[ -e "$active_dropin" ]]; then previous_schema=true; fi
timer_was_active=false
if systemctl --user is-active --quiet "$timer"; then timer_was_active=true; fi
changed=false

restore_previous() {
    result=$?
    trap - EXIT INT TERM
    if [[ "$changed" == true && $result -ne 0 ]]; then
        echo "Switch failed; restoring previous runtime configuration" >&2
        if [[ "$previous_schema" == true ]]; then
            install -m 0644 "$schema_backup" "$active_dropin"
        elif [[ -e "$active_dropin" ]]; then
            mv "$active_dropin" "$state_dir/failed-schema-dropin-$(date +%Y%m%d-%H%M%S).conf"
        fi
        systemctl --user daemon-reload || true
        systemctl --user restart "$service" || true
    fi
    if [[ "$timer_was_active" == true ]]; then
        systemctl --user start "$timer" || result=1
    else
        systemctl --user stop "$timer" || true
    fi
    echo "RUNTIME_SWITCH_EXIT=$result"
    exit "$result"
}
trap restore_previous EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

systemctl --user stop "$timer"
if systemctl --user is-active --quiet "$watchdog"; then
    for _ in {1..20}; do
        systemctl --user is-active --quiet "$watchdog" || break
        sleep 1
    done
    systemctl --user is-active --quiet "$watchdog" && {
        echo "Watchdog probe did not finish; refusing to restart model" >&2
        exit 1
    }
fi

changed=true
case "$target" in
    legacy)
        if [[ -e "$active_dropin" ]]; then
            mv "$active_dropin" "$state_dir/disabled-schema-dropin-$(date +%Y%m%d-%H%M%S).conf"
        fi
        ;;
    json-schema)
        if [[ ! -e "$active_dropin" ]]; then
            install -m 0644 "$schema_backup" "$active_dropin"
        fi
        ;;
esac
systemctl --user daemon-reload
systemctl --user restart "$service"

healthy=false
for _ in {1..90}; do
    if curl --fail --silent --max-time 2 "$api/health" \
        | jq -e '.status == "healthy" and .model == "openclaw"' >/dev/null; then
        healthy=true
        break
    fi
    systemctl --user is-active --quiet "$service" || break
    sleep 1
done
[[ "$healthy" == true ]]
curl --fail --silent --max-time 5 "$api/v1/models" \
    | jq -e 'any(.data[]; .id == "openclaw")' >/dev/null

stamp=$(date +%Y%m%d-%H%M%S)
smoke_file="$state_dir/smoke-$target-$stamp.json"
curl --fail --silent --show-error --max-time 45 \
    -H 'Content-Type: application/json' \
    -d '{"model":"openclaw","messages":[{"role":"user","content":"Reply with exactly READY."}],"temperature":0,"max_tokens":8}' \
    "$api/v1/chat/completions" > "$smoke_file"
jq -e '.choices[0].message.content | type == "string" and length > 0' "$smoke_file" >/dev/null

if [[ "$target" == legacy ]]; then
    test ! -e "$active_dropin"
    expected_workdir="$legacy_source"
    expected_binding="$home_dir/continuous-batching-p8-20260916-attempt3/build"
else
    test -e "$active_dropin"
    expected_workdir="$schema_source"
    expected_binding="$home_dir/tensorrt-json-schema-release-b921d36/pybind"
fi
test "$(systemctl --user show "$service" -p WorkingDirectory --value)" = "$expected_workdir"
test "$(systemctl --user show "$service" -p Environment --value | tr ' ' '\n' | grep -Fx "EDGELLM_PYBIND_DIR=$expected_binding")"
systemctl --user show "$service" -p ActiveState -p MainPID -p WorkingDirectory
systemctl --user show "$timer" -p ActiveState -p UnitFileState
jq -c '{status,model,active_requests,queued_requests}' < <(curl --fail --silent --max-time 3 "$api/health")
jq -c '{model:.model,finish_reason:.choices[0].finish_reason,content:.choices[0].message.content}' "$smoke_file"
echo "RUNTIME_SWITCH_TARGET=$target"
changed=false
