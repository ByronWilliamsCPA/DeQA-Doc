import logging

logger = logging.getLogger(__name__)

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


def get_level_token_ids(tokenizer, level_names=None):
    """Get token IDs for quality level names.

    Handles tokenizer families that prepend BOS or space tokens.
    Uses the last token to ensure we get the actual word token.

    Args:
        tokenizer: HuggingFace tokenizer.
        level_names: Override level names (default: LEVEL_NAMES).

    Returns:
        List of token IDs, one per level.
    """
    if level_names is None:
        level_names = LEVEL_NAMES
    ids = []
    for name in level_names:
        token_ids = tokenizer(name, add_special_tokens=False)["input_ids"]
        if len(token_ids) < 1:
            raise ValueError(f"Tokenizer produced empty output for '{name}'")
        if len(token_ids) > 1:
            logger.warning(
                "Level name '%s' tokenized to %d tokens; using last token ID %d",
                name, len(token_ids), token_ids[-1],
            )
        ids.append(token_ids[-1])
    return ids
