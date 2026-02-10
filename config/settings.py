# ================== Experiment Settings ==================

TEST_MODE = True              # Run quick small test

# ----- Imbalance Handling -----
RUN_IMBALANCE = True          # Use imbalance-aware loss
USE_FOCAL = True              # If True → Focal Loss, else → Weighted CE
FOCAL_GAMMA = 2.0             # Strength of focal effect

# ----- Data Level Methods -----
RUN_AUGMENTATION = False      # Apply graph augmentation

# ----- Model Level Methods -----
RUN_G2G = False               # Use Graph-to-Graph module
