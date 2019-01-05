import re
from string import punctuation
import os
import pandas as pd
import data_manager as data
import numpy as np
from tqdm import tqdm

remove_characters = re.sub("[,.!?\'\n]", '', punctuation)

def clean(text, characters=remove_characters):
    """
    Lowercase words, removes provided characters
    :param text: input text
    :param characters: set of characters to remove
    :return: text
    """
    import re

    regex = re.compile('[%s]' % re.escape(characters))
    return regex.sub('', text).lower()


def get_sentences(text):
    """
    Splits text into sentences
    :param text: input text
    :return: sentence array
    """
    from nltk.tokenize import sent_tokenize

    return sent_tokenize(text)


def get_words(sentences):
    """
    Splits sentences into words
    :param sentences: sentence array
    :return: nested word array
    """
    from nltk.tokenize import word_tokenize

    return [word_tokenize(sentence) for sentence in sentences]


def get_ngrams(words, n):
    """
    Combines nested word array into n-grams
    :param words: nested word array
    :param n: number of windowed combinations
    :return: nested n-gram array
    """
    from nltk.util import ngrams

    return [list(ngrams(word_array, n)) for word_array in words]


def remove_stopwords(words):
    """
    Removes stopwords from text
    :param words: nested word array
    :return: text
    """
    from nltk.corpus import stopwords

    return [[word for word in word_array if word not in stopwords.words()] for word_array in words]


def get_pos_tags(words):
    """
    Tags words with part-of-speech
    Only makes sense to use when tokenized to sentences
    :param words: nested word array
    :return: tagged words in sentences
    """
    from nltk import pos_tag

    return [pos_tag(word_array) for word_array in words]


def lemmatize(tagged_words):
    """
    Lemmatizes words
    :param tagged_words: nested tagged word array
    :return: nested lemmatized word array
    """
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet

    lemmatizer = WordNetLemmatizer()

    def remap_pos_tag(tag):
        if tag.startswith("J"):
            return wordnet.ADJ
        elif tag.startswith("V"):
            return wordnet.VERB
        elif tag.startswith("N"):
            return wordnet.NOUN
        elif tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    return [[lemmatizer.lemmatize(word, pos=remap_pos_tag(tag))  for word, tag in word_array] for word_array in tagged_words]


def flatten_sentences(words):
    """
    Joins words inside nested arrays to sentence array
    :param words: nested word array
    :return: sentence array
    """

    return [" ".join(sentence) for sentence in words]

def flatten_paragraph(sentences):
    return " ".join(sentences)


def flatten_array(array):
    """
    Flattens nested array one level
    :param array: n-D array
    :return: (n-1)-D array
    """
    from itertools import chain

    return list(chain(*array))


PREPROCESSED_DIR = "data/preprocessed.pkl"
def preprocess(texts):
    if os.path.isfile(PREPROCESSED_DIR):
        return data.get_pickle(PREPROCESSED_DIR)
    else:
        result = []
        for text in texts:
            text =           clean(text)
            text =         get_sentences(text)
            text =             get_words(text)
            # text =          get_pos_tags(text)
            # text =        lemmatize(text)
            text =    flatten_sentences(text)
            text = flatten_paragraph(text)
            print(text)

            result.append(text)
        data.save_pickle(PREPROCESSED_DIR, result)
    return result


TAGS_DIR = "data/text_tags.pkl"
def extract_tags(texts):
    if os.path.isfile(TAGS_DIR):
        return data.get_pickle(TAGS_DIR)
    else:
        texts = preprocess(texts)
        texts = get_pos_tags(map(lambda text: text.split(), texts))
        tags = list(map(lambda texts: [tagged[1] for tagged in texts], texts))
        data.save_pickle(TAGS_DIR, tags)
        return tags




TAGS_BOW = "data/tags_bow.pkl"

def get_tags_bow(sentences):
    if os.path.isfile(TAGS_BOW):
        return data.get_pickle(TAGS_BOW)
    else:
        from collections import Counter
        from nltk.data import load
        corpus = list(load('help/tagsets/upenn_tagset.pickle').keys())
        f = lambda x: Counter([y for y in x if y in corpus])
        df = pd.DataFrame({"tags": sentences})
        df["bow"] = (pd.DataFrame(df["tags"].apply(f).values.tolist())
                     .reindex(columns=corpus)
                     .fillna(0)
                     .astype(int)
                     .values
                     .tolist())
        result = df["bow"].tolist()
        data.save_pickle(TAGS_BOW, result)
        return result


def get_sentence_lengths():
    if os.path.isfile("data/sentence_lengths.pkl"):
        return data.get_pickle("data/sentence_lengths.pkl")
    else:
        texts = data.get_pickle("data/preprocessed.pkl")

        texts = list(map(lambda t: t.split(" "), texts))
        result = []
        for review in texts:
            review_len = len(review)
            sent_lengths = []
            c = 1
            for i, word in enumerate(review):
                if word in ".!?" or i == review_len-1:
                    sent_lengths.append(c)
                    c = 1
                else:
                    c += 1
            result.append(sent_lengths)
        data.save_pickle("data/sentence_lengths.pkl", result)
        return result



def get_word_counts(texts):
    from statistics import mean
    if os.path.isfile("data/word_counts.pkl"):
        return data.get_pickle("data/word_counts.pkl")
    else:
        result = []
        i = 0
        for text in texts:
            text = clean(text)
            sent = get_sentences(text)
            word = get_words(sent)
            sentence_lengths = [len(sentence) for sentence in word]
            try:
                if sentence_lengths:
                    result.append(sum(sentence_lengths))
                else:
                    result.append(0)
                i += 1
            except:
                print(text, i)
                return
        data.save_pickle("data/word_counts.pkl", result)
        return result


def normalize(x):
    mini = min(x)
    maxi = max(x)
    return [(x_i - mini) / (maxi - mini) for x_i in x]


def normalize_tag_bow():
    if os.path.isfile("data/tags_bow_norm.pkl"):
        return
    tag_counts = data.get_pickle(TAGS_BOW)
    result = []
    for tag in tag_counts:
        res = tag
        if any(tag):
            res = normalize(tag)
        result.append(res)
    data.save_pickle("data/tags_bow_norm.pkl", result)
    print(result[0])


def normalize_sentence_lengths():
    sentence_lengths = data.get_pickle("data/sentence_lengths.pkl")
    normed = normalize(sentence_lengths)
    data.save_pickle("data/sentence_lengths_norm.pkl", normed)
    print(normed)


def normalize_word_counts():
    word_counts = data.get_pickle("data/word_counts.pkl")
    normed = normalize(word_counts)
    data.save_pickle("data/word_counts_norm.pkl", normed)
    print(normed)


def get_spelling_ratios():
    if os.path.isfile("data/spelling_ratios.pkl"):
        return data.get_pickle("data/spelling_ratios.pkl")
    import tables as tb
    arr_file = tb.open_file("data/docvecs.hdf", "r", filters=tb.Filters(complib='zlib', complevel=0))
    docvecs = arr_file.create_earray(arr_file.root, "docvecs")

    sent_lengths = get_sentence_lengths()
    sent_lengths = list(map(lambda sent: sum(sent), sent_lengths))
    print(sent_lengths[:2])

    ratios = []
    for i, embeddings in enumerate(tqdm(docvecs)):
        count = 0
        for embedding in embeddings:
            if not np.any(embedding):
                break
            count += 1

        ratio = count / min(sent_lengths[i], 300)
        ratios.append(ratio)

    data.save_pickle("data/spelling_ratios.pkl", ratios)

    return ratios


def get_sent_word_distribution():
    sent_lengths = get_sentence_lengths()

    sent_nums = list(map(lambda sent: len(sent), sent_lengths))
    sent_nums_normed = normalize(sent_nums)

    word_counts = list(map(lambda sent: sum(sent) / len(sent), sent_lengths))
    word_counts_normed = normalize(word_counts)

    data.save_pickle("data/sent_counts.pkl", sent_nums)
    data.save_pickle("data/sent_counts_norm.pkl", sent_nums_normed)
    data.save_pickle("data/word_counts.pkl", word_counts)
    data.save_pickle("data/word_counts_norm.pkl", word_counts_normed)



def discreticize_ratings():
    if os.path.isfile("data/ratings_disc.pkl"):
        return
    import data_selection as select
    ratings = select.get_selection().overall.astype(int).tolist()
    ratings = list(map(lambda r: r-1, ratings))
    n_values = np.max(ratings) + 1
    discrete = np.eye(n_values)[ratings]
    data.save_pickle("data/ratings_disc.pkl", discrete)
    print(discrete[:2])

def discreticize_labels():
    if os.path.isfile("data/labels_disc.pkl"):
        return
    import data_selection as select
    ratings = select.get_selection().helpful.astype(int).tolist()
    n_values = np.max(ratings) + 1
    discrete = np.eye(n_values)[ratings]
    data.save_pickle("data/labels_disc.pkl", discrete)
    print(discrete[:2])

def get_review_counts():
    import data_selection as select
    sel = select.get_selection()
    reviewer_counts = sel.reviewerID.value_counts()
    reviewer_result = []
    for index, row in sel.reviewerID.iteritems():
        reviewer_result.append(reviewer_counts[row])

    product_counts = sel.asin.value_counts()
    product_result = []
    for index, row in sel.asin.iteritems():
        product_result.append(product_counts[row])

    data.save_pickle("data/reviewer_review_counts.pkl", reviewer_result)
    data.save_pickle("data/product_review_counts.pkl", product_result)



if __name__ == '__main__':
    import data_selection as select
    sel = select.get_selection()
    texts = sel.reviewText.tolist()
    preprocessed = preprocess(texts)
    sent_lengths = get_sentence_lengths()
    print(sent_lengths[:2])
    get_word_counts(texts)
    pos_tags = extract_tags(texts)
    print(pos_tags[:2])
    pos_tags_bow = get_tags_bow(pos_tags)
    print(pos_tags_bow[:2])
    discreticize_labels()
    discreticize_ratings()

    normalize_tag_bow()

    ratios = get_spelling_ratios()
    print(ratios[:10])

    get_sent_word_distribution()

    normalize_sentence_lengths()
    normalize_word_counts()

    get_review_counts()
