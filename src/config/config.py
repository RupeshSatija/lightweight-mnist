from pathlib import Path


class Config:
    # Base paths
    ROOT_DIR = Path(__file__).parent.parent.parent
    OUTPUT_DIR = ROOT_DIR / "outputs"

    # Output subdirectories
    METRICS_DIR = OUTPUT_DIR / "metrics"
    PLOTS_DIR = OUTPUT_DIR / "plots"
    CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
    LOGS_DIR = OUTPUT_DIR / "logs"

    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 1
    DEVICE = "cuda"  # or "cpu"

    # Model parameters
    NUM_CLASSES = 10

    @classmethod
    def setup_directories(cls):
        """Create all necessary directories if they don't exist."""
        dirs = [
            cls.OUTPUT_DIR,
            cls.METRICS_DIR,
            cls.PLOTS_DIR,
            cls.CHECKPOINTS_DIR,
            cls.LOGS_DIR,
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_instance(cls):
        """Returns an instance of the Config class."""
        return cls()
