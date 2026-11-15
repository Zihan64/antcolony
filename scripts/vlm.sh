#!/bin/bash
# Start the VLM server on this Mac WITHOUT internet (local model files, no -hf download check).
# Usage:  bash scripts/vlm.sh        (leave this window open; Ctrl-C stops the server)
SNAP="$HOME/.cache/huggingface/hub/models--ggml-org--Qwen2.5-VL-3B-Instruct-GGUF/snapshots/5037fcf163dd95d1e41d1974465f0898ed108ca2"
if curl -s -m 2 http://127.0.0.1:8090/health | grep -q ok; then
  echo "VLM server is already running on port 8090 - nothing to do."
  exit 0
fi
export HF_HUB_OFFLINE=1
exec llama-server -m "$SNAP/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf" --mmproj "$SNAP/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf" \
     --host 0.0.0.0 --port 8090 -ngl 99 -c 8192 --image-max-tokens 1024
