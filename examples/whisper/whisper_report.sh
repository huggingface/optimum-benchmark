# one argument: cpu or cuda
if [ $# -eq 1 ]; then
    # if argument is cpu
    if [ $1 = "cpu" ]; then
        python reporter.py \
            -b experiments/cpu_1_10/pytorch/whisper_baseline/ \
            -e experiments/cpu_1_10/onnxruntime

        python reporter.py \
            -b experiments/cpu_1_100/pytorch/whisper_baseline/ \
            -e experiments/cpu_1_100/onnxruntime

    # if argument is cuda
    elif [ $1 = "cuda" ]; then
        python reporter.py \
            -b experiments/cuda_1_10/pytorch/whisper_baseline/ \
            -e experiments/cuda_1_10/onnxruntime \
            experiments/cuda_1_10/pytorch/whisper_baseline_with_fp16/
        python reporter.py \
            -b experiments/cuda_1_100/pytorch/whisper_baseline/ \
            -e experiments/cuda_1_100/onnxruntime \
            experiments/cuda_1_100/pytorch/whisper_baseline_with_fp16/
        python reporter.py \
            -b experiments/cuda_8_100/pytorch/whisper_baseline/ \
            -e experiments/cuda_8_100/onnxruntime \
            experiments/cuda_8_100/pytorch/whisper_baseline_with_fp16/
        python reporter.py \
            -b experiments/cuda_8_10/pytorch/whisper_baseline/ \
            -e experiments/cuda_8_10/onnxruntime \
            experiments/cuda_8_10/pytorch/whisper_baseline_with_fp16/

    # if argument is not cpu or cuda
    else
        echo "Invalid argument"
    fi
fi
