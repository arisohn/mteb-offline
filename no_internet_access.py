import os
os.environ["HF_DATASETS_OFFLINE"]="1" # 1 for offline
os.environ["TRANSFORMERS_OFFLINE"]="1" # 1 for offline

"""
os.environ["TRANSFORMERS_CACHE"]="/gpfswork/rech/six/commun/models"
os.environ["HF_DATASETS_CACHE"]="/gpfswork/rech/six/commun/datasets"
os.environ["HF_MODULES_CACHE"]="/gpfswork/rech/six/commun/modules"
os.environ["HF_METRICS_CACHE"]="/gpfswork/rech/six/commun/metrics"
"""

from mteb import MTEB
from sentence_transformers import SentenceTransformer

model_path = "/Users/arisohn/Work/mteb-offline/models/sentence-transformers_allenai-specter"
model_name = "xxxxx"
model = SentenceTransformer(model_path)
evaluation = MTEB(tasks=["Banking77Classification"])
evaluation.run(model, output_folder=f"results/{model_name}")
