DEFAULT_GENERATE_KWARGS = {
    "max_new_tokens": 100,
    "min_new_tokens": 100,
    "do_sample": False,
    "use_cache": True,
    "pad_token_id": 0,
    "num_beams": 1,
}

DEFAULT_DIFUSION_KWARGS = {
    "num_images_per_prompt": 1,
}

DEFAULT_INPUT_SHAPES = {
    # used with all tasks
    "batch_size": 2,
    # used with text input tasks
    "sequence_length": 16,
    # used with multiple choice tasks where input
    # is of shape (batch_size, num_choices, sequence_length)
    "num_choices": 1,
    # used with audio input tasks
    "feature_size": 80,
    "nb_max_frames": 3000,
    "audio_sequence_length": 16000,
}


def format_float(x: float) -> float:
    return float(f"{x:.3g}")


def three_sig_figs(func):
    def wrapper(*args, **kwargs):
        return format_float(func(*args, **kwargs))

    return wrapper
