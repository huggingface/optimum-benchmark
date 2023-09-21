for f in configs/*.yaml; do
    if [ "$f" = "configs/bge_base_config.yaml" ]; then
        # skip
        continue
    fi
    optimum-benchmark --config-dir configs --config-name $(basename $f .json) -m
done
