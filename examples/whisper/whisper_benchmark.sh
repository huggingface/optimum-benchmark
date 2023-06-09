if [ $# -eq 1 ]; then
  # if argument is cpu
  if [ $1 = "cpu" ]; then
    python main.py --config-name whisper_baseline \
      -m device=cpu benchmark.new_tokens=10,100
    python main.py --config-name whisper_auto_qnt -m \
      -m device=cpu benchmark.new_tokens=10,100
    python main.py --config-name whisper_auto_opt+qnt -m \
      -m device=cpu benchmark.new_tokens=10,100
  # if argument is cuda
  elif [ $1 = "cuda" ]; then
    python main.py --config-name whisper_baseline \
      -m device=cuda benchmark.batch_size=1,8 benchmark.new_tokens=10,100 \
      experiment_name=whisper_baseline_with_fp16 backend.fp16=true
    python main.py --config-name whisper_auto_opt -m \
      -m device=cuda benchmark.batch_size=1,8 benchmark.new_tokens=10,100
  # if argument is not cpu or cuda
  else
    echo "Invalid argument"
  fi
fi
