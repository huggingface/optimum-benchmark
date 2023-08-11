if [ $# -eq 1 ]; then
    if [ $1 = "cpu" ]; then
        optimum-benchmark --config-dir ./ --config-name whisper_baseline -m benchmark.new_tokens=10,100 device=cpu
        optimum-benchmark --config-dir ./ --config-name whisper_auto_qnt -m -m benchmark.new_tokens=10,100 device=cpu
        optimum-benchmark --config-dir ./ --config-name whisper_auto_opt+qnt -m -m benchmark.new_tokens=10,100 device=cpu
        elif [ $1 = "cuda" ]; then
        # find a cuda device that is not being used by reading nvidia-smi output and parsing it
        CUDA_DEVICE=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | sort -n -k 2 | tail -n 1 | cut -d ',' -f 1)
        optimum-benchmark --config-dir ./ --config-name whisper_baseline -m benchmark.input_shapes.batch_size=1,8 benchmark.new_tokens=10,100 device=cuda:$CUDA_DEVICE
        optimum-benchmark --config-dir ./ --config-name whisper_auto_opt -m benchmark.input_shapes.batch_size=1,8 benchmark.new_tokens=10,100 device=cuda:$CUDA_DEVICE
    else
        echo "Invalid argument"
    fi
fi
