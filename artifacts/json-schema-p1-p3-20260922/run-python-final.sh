#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
task_dir=/home/ajmalrasi/json-schema-p1-p3-20260922
source_dir="$task_dir/source"
reference_dir=/home/ajmalrasi/TensorRT-Edge-LLM
exec > >(tee -a "$task_dir/python-final.log") 2>&1
date -Is
export EDGELLM_PYBIND_DIR="$task_dir/build"
cd "$source_dir"
LLM_SDK_DIR="$source_dir" "$reference_dir/.venv/bin/python" -m pytest -q \
  -o addopts= \
  tests/python-unittests/test_structured_output.py \
  tests/python-unittests/test_continuous_http.py \
  tests/python-unittests/test_server_requests.py \
  tests/python-unittests/test_server_runtime.py \
  --junitxml="$task_dir/python-tests-final.xml"
echo PYTHON_FINAL_GATE=passed
date -Is
