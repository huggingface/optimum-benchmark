import torch


def run_with_torch_no_grad(module, inputs):
    """
    Run a model with torch.no_grad()

    Args:
        module (torch.nn.Module): model to run
        inputs (dict): model inputs

    Returns:
        None
    """
    with torch.no_grad():
        module(**inputs)


def run_with_torch_backward(module, inputs, output_name):
    """
    Run a model with backward pass

    Args:
        module (torch.nn.Module): model to run
        inputs (dict): model inputs
        output_name (str): name of the output to run backward on

    Returns:
        None
    """
    module(**inputs)[output_name].mean().backward()


def sequence_dummy_inputs(batch_size, seq_len, sparsity, vocab_size=1000, device='cpu', right_pad=True):
    """
    Generate dummy sequence inputs (example for BERT)

    Args:
        batch_size (int): number of sequences in the batch
        seq_length (int): length of each sequence
        sparsity (float): sparsity of the attention mask (0.0 - 1.0)
        vocab_size (int): vocabulary size of the model
        device (str): device to put the inputs on
        right_pad (bool): whether to right pad the inputs

    Returns:
        inputs (dict): dummy inputs
    """

    # Generate random sparse input tensor
    input_ids = torch.randint(
        low=1, high=vocab_size, size=(batch_size, seq_len), device=device, dtype=torch.long)

    # Generate random sparse attention mask
    attention_mask = torch.ones(
        (batch_size, seq_len), device=device, dtype=torch.long)

    # apply mask
    mask = torch.rand((batch_size, seq_len))
    attention_mask[mask < sparsity] = 0
    if right_pad:
        attention_mask, _ = attention_mask.sort(dim=-1, descending=True)

    input_ids[attention_mask == 0] = 0

    token_type_ids = torch.zeros(
        (batch_size, seq_len), device=device, dtype=torch.long)

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}


def test_outputs(original_outputs, transformed_outputs, test_keys, atol=1e-4, rtol=1e-5):
    """
    Test a transformed module's outputs compared to its original implementation

    Args:
        original_outputs (dict): original module's outputs
        transformed_outputs (dict): transformed module's outputs
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    Returns:
        None

    Raises:
        AssertionError: if the transformed module's outputs don't match the original module's outputs
    """

    for key in test_keys:
        torch.testing.assert_close(
            original_outputs[key], transformed_outputs[key], atol=atol, rtol=rtol, equal_nan=True)
