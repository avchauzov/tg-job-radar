import json

from _production.airflow.plugins.production.token_optimizer import (
    optimize_prefiltering_tokens,
)
from _production.config import config

# Load current tokens from config
with open("_production/config/config.json", "r") as f:
    config = json.load(f)
current_tokens = config["prefiltering_tokens"]

# Run the optimization
result = optimize_prefiltering_tokens(current_tokens)
print(result)
