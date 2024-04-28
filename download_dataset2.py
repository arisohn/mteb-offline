from datasets import load_dataset_builder
builder = load_dataset_builder("mteb/banking77")
builder.download_and_prepare("datasets/mteb-banking77")
