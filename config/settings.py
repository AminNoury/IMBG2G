# ================== Experiment Settings ==================

TEST_MODE = False

# ----- Imbalance -----
RUN_IMBALANCE = False
USE_FOCAL = False
FOCAL_GAMMA = 1.0

# ==================== AUGMENTATION ====================
RUN_AUGMENTATION = True          # Enable/disable graph augmentations
AUG_NODE_DROP    = True          # Enable node dropping
NODE_DROP_RATIO  = 0.2           # Ratio of nodes to drop
AUG_EDGE_PERTURB = True          # Enable edge perturbation (remove/add edges)
EDGE_PERTURB_RATIO = 0.2         # Ratio of edges to perturb
AUG_FEATURE_MASK = True          # Enable feature masking
FEATURE_MASK_RATIO = 0.1         # Ratio of features to mask


# ----- Model Level -----
RUN_G2G = False
