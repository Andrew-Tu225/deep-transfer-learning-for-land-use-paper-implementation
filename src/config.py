# Paper hyperparameters
# shared by every experiment notebook so the 4-run comparison stays controlled.

NUM_CLASSES = 10

# Data (Section 3.1)
BATCH_SIZE = 64
VALID_SIZE = 0.25
SEED = 42

# Optimization (Section 3.2)
LEARNING_RATE = 1e-4
MAX_EPOCHS = 25

# Gradient clipping (Section 3.3.2)
GRAD_CLIP_NORM = 0.1

# Early stopping (Section 3.3.3)
EARLY_STOPPING_PATIENCE = 5

# ReduceLROnPlateau (Section 3.3.4)
LR_SCHEDULER_FACTOR = 0.1
LR_SCHEDULER_PATIENCE = 2
