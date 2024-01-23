# Importing required libraries
import numpy as np
import pandas as pd
import logging
import os
import sys
import json
from sklearn import datasets
from sklearn.model_selection import train_test_split
# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
from utils import singleton, get_project_dir, configure_logging

DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data'))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = '../settings.json'

# Load configuration settings from JSON
logger.info("Loading configuration settings from JSON...")
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
logger.info("Defining paths...")
DATA_DIR = get_project_dir(conf['general']['data_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])
INFERENCE_PATH = os.path.join(DATA_DIR, conf['inference']['inp_table_name'])


# Singleton class for generating XOR data set

# Main execution
if __name__ == "__main__":
    configure_logging()
    logger.info("Starting script...")
    df = datasets.load_iris(as_frame=True)
    x = df.data
    y = df.target
    test_size = conf['general'].get('test_size')
    x_train, x_inference, y_train, y_inference = (
        train_test_split(x, y, test_size=test_size, random_state=42))
    logger.info("Saving train and inference data...")
    train = pd.concat([x_train, y_train], axis=1)
    inference = pd.concat([x_inference, y_inference], axis=1)
    train.to_csv(TRAIN_PATH, index=False)
    inference.to_csv(INFERENCE_PATH, index=False)
    logger.info("Data saved in {} and {}".format(TRAIN_PATH, INFERENCE_PATH))