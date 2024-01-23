"""
Script loads the latest trained model, data for inference and predicts results.
Imports necessary packages and modules.
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime
from utils import get_project_dir, configure_logging
from Model import Model
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import torch
import numpy as np

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = "../settings.json"


# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf["general"]["data_dir"])
MODEL_DIR = get_project_dir(conf["general"]["models_dir"])
RESULTS_DIR = get_project_dir(conf["general"]["results_dir"])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--infer_file",
    help="Specify inference data file",
    default=conf["inference"]["inp_table_name"],
)
parser.add_argument("--out_path", help="Specify the path to the output table")


def get_latest_model_path() -> str:
    """Gets the path of the latest saved model"""
    latest = None
    for dirpath, dirnames, filenames in os.walk(MODEL_DIR):
        for filename in filenames:
            if not latest or datetime.strptime(
                latest, conf["general"]["datetime_format"] + ".pickle"
            ) < datetime.strptime(
                filename, conf["general"]["datetime_format"] + ".pickle"
            ):
                latest = filename
    return os.path.join(MODEL_DIR, latest)


def get_model_by_path(path: str) -> DecisionTreeClassifier:
    """Loads and returns the specified model"""
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
            logging.info(f"Path of the model: {path}")
            return model
    except Exception as e:
        logging.error(f"An error occurred while loading the model: {e}")
        sys.exit(1)


def get_inference_data(path: str) -> pd.DataFrame:
    """loads and returns data for inference from the specified csv file"""
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logging.error(f"An error occurred while loading inference data: {e}")
        sys.exit(1)


def predict_results(model: any, infer_data: pd.DataFrame) -> pd.DataFrame:
    """Predict the results and join it with the infer_data"""
    model.eval()
    results = np.zeros((infer_data.shape[0], 1))

    temp_results = (
        model(torch.tensor(infer_data.iloc[:, :-1].values.astype(np.float32)))
        .detach()
        .numpy()
    )

    for i in range(temp_results.shape[0]):
        results[i] = np.argmax(temp_results[i])
    infer_data = results
    return infer_data


def store_results(results: pd.DataFrame, path: str = None) -> None:
    """Store the prediction results in 'results' directory with current datetime as a filename"""
    if not path:
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        path = datetime.now().strftime(conf["general"]["datetime_format"]) + ".csv"
        path = os.path.join(RESULTS_DIR, path)
    pd.DataFrame(results).to_csv(path, index=False)
    logging.info(f"Results saved to {path}")


def calcuate_accuracy(infer_data: pd.DataFrame, results: pd.DataFrame) -> float:
    """Calculate the accuracy of the model"""
    accuracy = 0
    for i in range(results.shape[0]):
        if infer_data.iloc[i, -1] == results[i]:
            accuracy += 1

    return accuracy / results.shape[0]


def main():
    """Main function"""
    configure_logging()
    args = parser.parse_args()
    model = Model(4)
    model.load_state_dict(torch.load(get_latest_model_path()))
    model.eval()
    infer_file = args.infer_file
    infer_data = get_inference_data(os.path.join(DATA_DIR, infer_file))
    t1 = time.time()
    results = predict_results(model, infer_data)
    store_results(results, args.out_path)
    acc = calcuate_accuracy(infer_data, results)
    t2 = time.time()
    logging.info(f"Prediction results: {results}")
    logging.info(f"Accuracy of inference: {acc}")
    logging.info(f"Time taken to predict: {t2-t1} seconds")


if __name__ == "__main__":
    main()
