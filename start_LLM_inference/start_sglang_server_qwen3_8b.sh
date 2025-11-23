mkdir -p /.cache/huggingface

docker run --gpus '"device=4"' \
    --shm-size 32g \
    -p 30002:30002 \
    -v /.cache/huggingface \
    --env "HF_TOKEN" \
    --ipc=host \
    lmsysorg/sglang:latest \
    bash -c "python3 -m sglang.launch_server --model-path Qwen/Qwen3-8B --host 0.0.0.0 --port 30002 --tp-size 1 --mem-fraction-static 0.7"