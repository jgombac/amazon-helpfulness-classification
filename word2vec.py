
def get_word2vec():
    import data_manager as data
    import numpy as np
    import os
    from gensim.models import KeyedVectors
    from tqdm import tqdm
    import tables as tb

    print("loading word2vec...")
    model = data.get_pickle("data/word2vec.model.pkl")
    # model = KeyedVectors.load_word2vec_format(os.path.join(os.path.dirname(__file__), "data/word2vec.300d.txt"))
    # data.save_pickle("data/word2vec.model.pkl", model)
    print("word2vec loaded")

    texts = data.get_pickle("data/preprocessed.pkl")
    print("splitting sentences into words")
    texts = list(map(lambda t: t.split(" "), texts))
    print("done!")

    arr_file = tb.open_file("data/docvecs.hdf", "w", filters=tb.Filters(complib='zlib', complevel=0))
    document_vecs = None # arr_file.create_earray(arr_file.root, "docvecs")
    for sentence in tqdm(texts):
        word_vecs = []
        for word in sentence:
            try:
                word_vecs.append(model.get_vector(word))
            except:
                pass

        # if len(word_vecs) == 0:
        #     word_vecs.append([0 for i in range(300)])
        # document_vecs.append(np.mean(word_vecs, axis=0))

        # Tried keeping order and padding but too heavy
        if len(word_vecs) > 300:
            word_vecs = word_vecs[:300]
        len_vecs = len(word_vecs)
        for i in range(300-len_vecs):
            word_vecs.append([0 for i in range(300)])

        word_vecs = np.array([word_vecs])
        if document_vecs is None:
            document_vecs = arr_file.create_earray(arr_file.root, "docvecs", obj=word_vecs)
        else:
            document_vecs.append(word_vecs)

    #data.save_pickle("data/wordvecs_np.pkl", document_vecs)

    return document_vecs

if __name__ == '__main__':
    import os
    import tables as tb
    os.remove("data/docvecs.hdf")
    docvecs = get_word2vec()



