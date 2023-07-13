from transformers import AutoTokenizer


class GluePreprocessor:
    def __init__(self, model_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def __call__(self, examples):
        return self.tokenizer(
            examples["sentence"],
            padding="max_length",
            max_length=128,
            truncation=True,
        )
