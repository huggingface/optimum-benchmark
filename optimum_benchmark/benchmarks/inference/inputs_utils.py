def extract_text_generation_inputs(inputs):
    if "pixel_values" in inputs:
        # image input
        text_generation_inputs = {"inputs": inputs["pixel_values"]}
    elif "input_values" in inputs:
        # speech input
        text_generation_inputs = {"inputs": inputs["input_values"]}
    elif "input_features" in inputs:
        # waveform input
        text_generation_inputs = {"inputs": inputs["input_features"]}
    elif "input_ids" in inputs:
        # text input
        text_generation_inputs = {"inputs": inputs["input_ids"]}
    else:
        raise ValueError("Could not find any valid text generation inputs.")

    return text_generation_inputs
