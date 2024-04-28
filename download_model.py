import os
import sentence_transformers
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/Users/arisohn/Work/mteb-offline/models"
sentence_transformers_cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME")
model_repo="sentence-transformers/allenai-specter"
revision="29f9f45ff2a85fe9dfe8ce2cef3d8ec4e65c5f37"
model_path = os.path.join(sentence_transformers_cache_dir, model_repo.replace("/", "_"))
model_path_tmp = sentence_transformers.util.snapshot_download(
    repo_id=model_repo,
    revision=revision,
    cache_dir=sentence_transformers_cache_dir,
    library_name="sentence-transformers",
    library_version=sentence_transformers.__version__,
    ignore_files=["flax_model.msgpack", "rust_model.ot", "tf_model.h5",],
)
os.rename(model_path_tmp, model_path)
