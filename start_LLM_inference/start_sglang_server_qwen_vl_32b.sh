mkdir -p /fsx-neo/dedicated-fsx-data-repo-neo-us-east-1/genglin/.cache/huggingface

docker run --gpus '"device=6, 7"' \
    --shm-size 32g \
    -p 30001:30001 \
    -v /fsx-neo/dedicated-fsx-data-repo-neo-us-east-1/genglin/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=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" \
    --ipc=host \
    lmsysorg/sglang:latest \
    bash -c "python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-32B-Instruct --host 0.0.0.0 --port 30001 --tp-size 2 --mem-fraction-static 0.7"