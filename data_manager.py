import os
import gzip
import json
import pandas as pd
from multiprocessing import cpu_count
import requests
import tqdm


CPU = cpu_count() / 2

DATA_DIR = "data/"
TEXT_DIR = DATA_DIR + "texts/"

DATASET_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz"
DATASET_RAW = TEXT_DIR + "electronics.json.gz"
DATASET = DATA_DIR + "electronics.feather"

def download_dataset():
    response = requests.get(DATASET_URL, stream=True)

    with open(DATASET_RAW, "wb") as handle:
        for data in tqdm.tqdm(response.iter_content(chunk_size=1024)):
            if data:
                handle.write(data)

def to_dataframe(filename):
    data = []
    i = 0
    with gzip.open(filename, "r") as f:
        for line in f:
            data.append(json.loads(line))
            # if i == 1000:
            #     break
            # i += 1
    data = pd.DataFrame(data)
    data["positive"], data["all"] = zip(*data["helpful"])
    data["helpful"] = data.apply(lambda row: row["positive"] / row["all"] if row["all"] > 0 else 0, axis=1)
    return data[["reviewerID", "asin", "reviewTime", "summary", "reviewText", "overall", "positive", "all", "helpful"]]


def read_textfile(filename):
    text = ""
    with open(TEXT_DIR + filename, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            text += line
    return text


def save_pickle(filename, data):
    from pickle import dump

    with open(filename, "wb") as file:
        dump(data, file)


def get_pickle(filename):
    from pickle import load

    with open(filename, "rb") as file:
        return load(file)


def get_dataset():
    if os.path.isfile(DATASET):
        return pd.read_feather(DATASET, nthreads=CPU)
    else:
        if not os.path.isfile(DATASET_RAW):
            download_dataset()
        data = to_dataframe(DATASET_RAW)
        data.to_feather(DATASET)
        return data

if __name__ == '__main__':
    get_dataset()
