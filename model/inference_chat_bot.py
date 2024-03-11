class GenerationBot:
    def __init__(
        self,
        trained_model_dir,
        model_name="PY007/TinyLlama-1.1B-step-50K-105b"
    ):
        self.config = PeftConfig.from_pretrained(trained_model_dir)


if __name__ == "__main__":
    pass
