import data_manager as data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def remove_unrated(dataset):
    # removes all reviews that aren't rated more than 2 times
    return dataset.loc[dataset["all"] > 2]


def remap_helpfulness(dataset):
    # remaps helpfulness score:  > 0.667 = 1 (helpful)  else 0 (unhelpful)
    dataset.helpful = np.where(dataset.helpful >= 0.8, 1, 0)
    return dataset


def plot_helpfulness(dataset):
    hist = dataset.helpful.hist(bins=2, rwidth=0.4)
    hist.set_xlabel("helpfulness")
    hist.set_ylabel("n hamples")
    hist.set_xticks([0.25, 0.75])
    hist.set_xticklabels(("unhelpful", "helpful"))
    plt.show()


def select_even(dataset):
    #plot_helpfulness(dataset)
    # count occurrences of each helpful class
    counts = dataset.helpful.value_counts()
    print(counts)
    # get the min occurrence count
    smallest_count = counts.min()
    print("Selecting smaller subsample of: ", smallest_count)
    splits = []
    # select a random subset of each helpful class to even out the dataset
    for key in counts.keys().tolist():
        splits.append(dataset[dataset.helpful == key].sample(n=smallest_count, random_state=42))

    return pd.concat(splits, ignore_index=True)


def get_selection():
    dataset = data.get_dataset()
    original_count = dataset.shape[0]
    print("Number of original examples: ", original_count)
    dataset = remove_unrated(dataset)
    removed_unrated_count = original_count - dataset.shape[0]
    print("Number of samples without rating: ", removed_unrated_count)
    dataset = remap_helpfulness(dataset)
    final_selection_count = dataset.shape[0]
    print("Number of final examples: ", final_selection_count)
    return select_even(dataset)#.sample(n=10000, random_state=42)


def get_tags():
    return data.get_pickle("data/text_tags.pkl")

if __name__ == '__main__':
    dataset = get_selection()
    plot_helpfulness(dataset)
    print(dataset.describe())

