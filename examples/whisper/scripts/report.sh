if [ $1 = "cpu" ]; then
    optimum-report -e experiments/cpu_onnxruntime_1_10 -b experiments/cpu_pytorch_8_100 -n cpu_1_10
    optimum-report -e experiments/cpu_onnxruntime_1_100 -b experiments/cpu_pytorch_8_100 -n cpu_1_100
    elif [ $1 = "cuda" ]; then
    optimum-report -e experiments/cuda_onnxruntime_64_10 -b experiments/cuda_pytorch_64_10 -n cuda_64_10
    optimum-report -e experiments/cuda_onnxruntime_64_100 -b experiments/cuda_pytorch_64_100 -n cuda_64_100
    optimum-report -e experiments/cuda_onnxruntime_128_10 -b experiments/cuda_pytorch_128_10 -n cuda_128_10
    optimum-report -e experiments/cuda_onnxruntime_128_100 -b experiments/cuda_pytorch_128_100 -n cuda_128_100
else
    echo "Invalid argument"
fi