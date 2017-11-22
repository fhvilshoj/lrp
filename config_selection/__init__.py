import yaml
import logging.config

LOGGER = 'configLogger'

# Configure logger
with open('/home/fhv/Projects/AU/thesis/graph/config_selection/logging.yaml') as f:
    c = yaml.load(f)
logging.config.dictConfig(c)
logger = logging.getLogger(LOGGER)
