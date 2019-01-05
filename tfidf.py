from sklearn.feature_extraction.text import TfidfVectorizer
import data_manager as data

def dummy(x):
    return x

def get_tfidf():
    documents = data.get_pickle("data/preprocessed.pkl")
    documents = list(map(lambda d: d.split(" "), documents))
    print(documents[:2])
    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=None,
        stop_words="english",
        tokenizer=dummy,
        preprocessor=dummy,
        ngram_range=(1,3),
        max_features=12000,
        max_df=0.99
    )

    matrix = vectorizer.fit_transform(documents)

    data.save_pickle("data/tfidf.pkl", matrix)

    print(vectorizer.get_feature_names())



def test():
    import numpy as np
    matrix = data.get_pickle("data/tfidf.pkl").toarray()
    indices = [1,2,3]
    print([matrix[i] for i in indices])
    #print(matrix[1])


if __name__ == '__main__':
    test()
    #get_tfidf()