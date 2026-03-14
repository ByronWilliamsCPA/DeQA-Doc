CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "./demo_logs"

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<|image|>"

# Quality Level Constants (DeQA convention: excellent=5 → bad=1)
LEVEL_NAMES = ["excellent", "good", "fair", "poor", "bad"]
LEVEL_SCORES = [5.0, 4.0, 3.0, 2.0, 1.0]
LEVEL_PREFIX = "The quality of the image is"
