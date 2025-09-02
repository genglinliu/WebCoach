mkdir -p /fsx-neo/dedicated-fsx-data-repo-neo-us-east-1/genglin/.cache/huggingface

docker run --gpus '"device=4"' \
    --shm-size 32g \
    -p 30002:30002 \
    -v /fsx-neo/dedicated-fsx-data-repo-neo-us-east-1/genglin/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=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" \
    --ipc=host \
    lmsysorg/sglang:latest \
    bash -c "python3 -m sglang.launch_server --model-path Qwen/Qwen3-8B --host 0.0.0.0 --port 30002 --tp-size 1 --mem-fraction-static 0.7"