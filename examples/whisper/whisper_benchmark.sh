# For GPU
python main.py --config-name whisper_baseline \
  -m device=cuda benchmark.batch_size=1,8 benchmark.new_tokens=10,100

python main.py --config-name whisper_baseline -m \
  -m device=cuda benchmark.batch_size=1,8 benchmark.new_tokens=10,100 \
  experiment_name=whisper_baseline_with_fp16 backend.fp16=true

python main.py --config-name whisper_auto_opt -m \
  -m device=cuda benchmark.batch_size=1,8 benchmark.new_tokens=10,100

# For CPU
python main.py --config-name whisper_baseline \
  -m device=cpu benchmark.new_tokens=10,100

python main.py --config-name whisper_auto_qnt -m \
  -m device=cpu benchmark.new_tokens=10,100

python main.py --config-name whisper_auto_opt+qnt -m \
  -m device=cpu benchmark.new_tokens=10,100
