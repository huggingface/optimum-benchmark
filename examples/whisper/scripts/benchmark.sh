if [ $1 = "cpu" ]; then
    optimum-benchmark --config-dir configs --config-name whisper_baseline -m device=cpu
    optimum-benchmark --config-dir configs --config-name whisper_auto_qnt -m device=cpu
    optimum-benchmark --config-dir configs --config-name whisper_auto_opt+qnt -m device=cpu
    elif [ $1 = "cuda" ]; then
    optimum-benchmark --config-dir configs --config-name whisper_baseline -m device=cuda
    optimum-benchmark --config-dir configs --config-name whisper_auto_opt -m device=cuda
else
    echo "Invalid argument"
fi
