import os
os.environ["HF_DATASETS_OFFLINE"]="1" # 1 for offline
os.environ["TRANSFORMERS_OFFLINE"]="1" # 1 for offline

"""
os.environ["TRANSFORMERS_CACHE"]="/gpfswork/rech/six/commun/models"
os.environ["HF_DATASETS_CACHE"]="/gpfswork/rech/six/commun/datasets"
os.environ["HF_MODULES_CACHE"]="/gpfswork/rech/six/commun/modules"
os.environ["HF_METRICS_CACHE"]="/gpfswork/rech/six/commun/metrics"
"""

import datasets
datasets.load_dataset("/Users/arisohn/Work/mteb-offline/datasets/mteb-banking77") 
