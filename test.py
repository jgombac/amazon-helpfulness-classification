import data_manager as data
import numpy as np
import tables as tb
import model as models
from keras.models import load_model
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

np.random.seed(123)

class Batcher():
    def __init__(self,product_counts,reviewer_counts,ratings,sent_counts,word_counts,spell_ratios,pos_dist,tfidf,embeddings,helpfulness,n=1):
        self.product_counts = product_counts
        self.reviewer_counts = reviewer_counts
        self.ratings = ratings
        self.sent_counts = sent_counts
        self.word_counts = word_counts
        self.spell_ratios = spell_ratios
        self.pos_dist = pos_dist
        self.tfidf = tfidf
        self.embeddings = embeddings
        self.helpfulness = helpfulness
        self.n = n

        self.size = len(helpfulness)

        self.validation_split = np.sort(np.random.choice(self.size, int(self.size * 0.15)))
        self.train_split = [i for i in range(self.size) if i not in self.validation_split]

        self.train_steps = len(self.train_split) // n
        self.validation_steps = len(self.validation_split) // n

        self.current_train_index = 0
        self.current_valid_index = 0


    def get_items(self, indices):
        features = {
            "product_counts": np.array([self.product_counts[i] for i in indices]),
            "reviewer_counts": np.array([self.reviewer_counts[i] for i in indices]),
            "ratings": np.array([self.ratings[i] for i in indices]),
            "sent_counts": np.array([self.sent_counts[i] for i in indices]),
            "word_counts": np.array([self.word_counts[i] for i in indices]),
            "spell_ratios": np.array([self.spell_ratios[i] for i in indices]),
            "pos_dist": np.array([self.pos_dist[i] for i in indices]),
            "tfidf": np.array([self.tfidf[i] for i in indices]),
            "embeddings": np.array([self.embeddings[i] for i in indices])
        }
        targets = {
            "helpfulness": np.array([self.helpfulness[i] for i in indices])
        }
        return features, targets

    def train_gen(self):
        train_size = len(self.train_split)
        np.random.shuffle(self.train_split)
        while True:
            if self.current_train_index >= train_size - self.n:
                self.current_train_index = 0
                np.random.shuffle(self.train_split)
            indices = self.train_split[self.current_train_index:self.current_train_index + self.n]
            self.current_train_index += self.n
            yield self.get_items(indices)

    def validation_gen(self):
        validation_size = len(self.validation_split)
        while True:
            if self.current_valid_index >= validation_size - self.n:
                self.current_valid_index = 0
            indices = self.validation_split[self.current_valid_index:self.current_valid_index + self.n]
            self.current_valid_index += self.n
            yield self.get_items(indices)

    def reset(self):
        self.current_train_index = 0
        self.current_valid_index = 0


arr_file = tb.open_file("data/docvecs.hdf", "r", filters=tb.Filters(complib='zlib', complevel=0))
embeddings = arr_file.create_earray(arr_file.root, "docvecs")
tfidf = data.get_pickle("data/tfidf.pkl").toarray()
product_counts = np.array(data.get_pickle("data/product_review_counts.pkl"))
reviewer_counts = np.array(data.get_pickle("data/reviewer_review_counts.pkl"))
ratings = np.array(data.get_pickle("data/ratings_disc.pkl"))
sent_counts = np.array(data.get_pickle("data/sent_counts_norm.pkl"))
word_counts = np.array(data.get_pickle("data/word_counts_norm.pkl"))
spell_ratios = np.array(data.get_pickle("data/spelling_ratios.pkl"))
pos_dist = np.array(data.get_pickle("data/tags_bow_norm.pkl"))
helpfulness = np.array(data.get_pickle("data/labels_disc.pkl"))

batch_size = 128
batcher = Batcher(product_counts, reviewer_counts, ratings, sent_counts, word_counts, spell_ratios, pos_dist, tfidf,
                  embeddings, helpfulness, batch_size)


def train_cnn():
    #model = load_model("data/models/cnn_double_chkpt.h5")
    model = models.model_cnn()
    model.summary()

    tensorboard  = TensorBoard(log_dir='./cnn1_graph', histogram_freq=0, write_graph=True, write_images=True)
    checkpoint = ModelCheckpoint("data/models/cnn_chkpt.h5", monitor='val_loss', save_best_only=True, verbose=1, mode="min")
    stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1)

    model.fit_generator(batcher.train_gen(),
                    steps_per_epoch=batcher.train_steps,
                    validation_data=batcher.validation_gen(),
                    validation_steps=batcher.validation_steps,
                    epochs=8,
                    callbacks=[tensorboard, checkpoint, stopping])

    model.save("data/models/cnn1.h5")
    batcher.reset()
    model = load_model("data/models/cnn_chkpt.h5")
    eval = model.evaluate_generator(batcher.validation_gen(), steps=batcher.validation_steps)
    print("CNN", eval)

    batcher.reset()


def train_lstm():
    #model = load_model("data/models/original.h5")
    model = models.model_original()
    model.summary()

    tensorboard  = TensorBoard(log_dir='./lstm1_graph', histogram_freq=0, write_graph=True, write_images=True)
    checkpoint = ModelCheckpoint("data/models/lstm_chkpt.h5", monitor='val_loss', save_best_only=True, verbose=1, mode="min")
    stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=1)

    model.fit_generator(batcher.train_gen(),
                    steps_per_epoch=batcher.train_steps,
                    validation_data=batcher.validation_gen(),
                    validation_steps=batcher.validation_steps,
                    epochs=8,
                    callbacks=[tensorboard, checkpoint, stopping])

    model.save("data/models/lstm1.h5")
    batcher.reset()
    model = load_model("data/models/lstm_chkpt.h5")
    eval = model.evaluate_generator(batcher.validation_gen(), steps=batcher.validation_steps)
    print("LSTM", eval)

    batcher.reset()


def train_baseline():
    #model = load_model("data/models/baseline.h5")

    model = models.baseline()
    model.summary()

    tensorboard  = TensorBoard(log_dir='./baseline1_graph', histogram_freq=0, write_graph=True, write_images=True)
    checkpoint = ModelCheckpoint("data/models/baseline_chkpt.h5", monitor='val_loss', save_best_only=True, verbose=1, mode="min")
    stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=1)

    model.fit_generator(batcher.train_gen(),
                    steps_per_epoch=batcher.train_steps,
                    validation_data=batcher.validation_gen(),
                    validation_steps=batcher.validation_steps,
                    epochs=8,
                    callbacks=[tensorboard, checkpoint, stopping])

    model.save("data/models/baseline1.h5")
    batcher.reset()
    model = load_model("data/models/baseline_chkpt.h5")
    eval = model.evaluate_generator(batcher.validation_gen(), steps=batcher.validation_steps)
    print("BASELINE", eval)

    batcher.reset()


if __name__ == '__main__':
    train_baseline()
    train_lstm()
    train_cnn()