"""
This script prepares the data, runs the training, and saves the model.
"""

import argparse
import os
import sys
import pickle
import json
import logging
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import datasets
import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
# Comment this lines if you have problems with MLFlow installation
import mlflow
from utils import get_project_dir, configure_logging

mlflow.autolog()

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

CONF_FILE = "../settings.json"
# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)
    print(conf['train'].get('data_sample'))

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
# TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
# parser.add_argument("--train_file",
#                    help="Specify inference data file",
#                    default=conf['train']['table_name'])
parser.add_argument("--model_path",
                    help="Specify the path for the output model")


class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x

class DataProcessor:

    def give_train_dim(self) -> int:
        return datasets.load_iris(as_frame=True).data.shape[1]

    def __init__(self) -> None:
        pass

    def prepare_data(self, max_rows: int = None) -> (
            pd.DataFrame, pd.DataFrame):
        logging.info("Preparing data for training...")
        df = datasets.load_iris(as_frame=True)
        x = df.data
        y = df.target
        x, y = self.data_rand_sampling(x, y, max_rows)
        return x, y

    def data_rand_sampling(self, x: pd.DataFrame, y: pd.DataFrame,
                           max_rows: int) -> (pd.DataFrame, pd.DataFrame):
        if not max_rows or max_rows < 0:
            logging.info('Max_rows not defined. Skipping sampling.')
        elif len(x) < max_rows:
            logging.info(
                'Size of dataframe is less than max_rows. Skipping sampling.')
        else:
            x = x.sample(n=max_rows, replace=False,
                         random_state=conf['general']['random_state'])
            y = y[x.index]
            logging.info(f'Random sampling performed. Sample size: {max_rows}')
        return x, y


class Training:

    def __init__(self, x_dim: int) -> None:
        self.model = Model(x_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.CrossEntropyLoss()
        print(self.model)

    def run_training(self, x: pd.DataFrame, y: pd.DataFrame, out_path: str = None,
                     test_size: float = 0.33) -> None:
        logging.info("Running training...")
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=test_size)
        start_time = time.time()
        self.train(x_train, y_train, x_test, y_test)
        end_time = time.time()
        logging.info(f"Training completed in {end_time - start_time} seconds.")
        self.save(out_path)

    def train(self, x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
        logging.info("Training the model...")

        EPOCHS = 100
        x_train = Variable(torch.from_numpy(x_train.to_numpy())).float()
        y_train = Variable(torch.from_numpy(y_train.to_numpy())).long()
        x_test = Variable(torch.from_numpy(x_test.to_numpy())).float()
        y_test = Variable(torch.from_numpy(y_test.to_numpy())).long()

        loss_list = np.zeros((EPOCHS,))
        accuracy_list = np.zeros((EPOCHS,))

        for epoch in tqdm.trange(EPOCHS):
            y_pred = self.model(x_train)
            loss = self.loss_fn(y_pred, y_train)
            loss_list[epoch] = loss.item()

            # Zero gradients
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                y_pred = self.model(x_test)
                correct = (torch.argmax(y_pred, dim=1) == y_test).type(
                    torch.FloatTensor)
                accuracy_list[epoch] = correct.mean()

        logging.info(f"Accuracy: {accuracy_list[-1]}")

    def save(self, path: str) -> None:
        logging.info("Saving the model...")
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        if not path:
            path = os.path.join(MODEL_DIR, datetime.now().strftime(
                conf['general']['datetime_format']) + '.pickle')
        else:
            path = os.path.join(MODEL_DIR, path)

        with open(path, 'wb') as f:
            pickle.dump(self.model, f)


def main():
    configure_logging()

    data_proc = DataProcessor()
    tr = Training(data_proc.give_train_dim())
    x, y = data_proc.prepare_data(max_rows=conf['train'].get('data_sample'))
    tr.run_training(x, y, test_size=conf['train']['test_size'])


if __name__ == "__main__":
    main()
