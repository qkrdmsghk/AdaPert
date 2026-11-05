from pathlib import Path

# File locations
DATA_DIR = Path(__file__).parent.parent / "research" / "data"
CACHE_DIR = Path(__file__).parent.parent / "research" / "cache"
CHECKPOINT_DIR = Path(__file__).parent.parent / "research" / "checkpoints"

# Columns of adata.obs
CONTROL = "control"                 # 1 for the non-targeting control cells
CELL_TYPE = "cell_line"
CONDITION = "condition"             # e.g. UBL5+ctrl
CONDITION_NAME = "condition_name"   # e.g. K562_UBL5+ctrl_1+1
BATCH = "batch"
CONTROL_LABEL = "ctrl"

TEST = "test"
VAL = "val"

# Keys of the output of src.evaluation.evaluate
RES_PRED = "pred"
RES_TRUTH = "truth"
RES_PERT_CAT = "pert_cat"
RES_CELL_TYPE = "cell_type"
RES_EXP_BATCH = "experimental_batch"
