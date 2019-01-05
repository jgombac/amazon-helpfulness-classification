from keras_layers import *
from keras.layers import Input, Dense, Dropout, Concatenate, LSTM, Conv1D, MaxPooling1D, Flatten
from keras.models import Model
from keras.optimizers import Adam, SGD



def baseline():
    tfidf_input = Input(shape=(12000,), name="tfidf")
    tfidf1 = Dense(2, activation="softmax", name="helpfulness")(tfidf_input)

    model = Model(inputs=[tfidf_input], outputs=tfidf1)

    model.compile(optimizer="adam",
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def get_conv_pool(x_input, n_grams=[3,4,5], feature_maps=32):
    branches = []
    for n in n_grams:
        branch = Conv1D(filters=feature_maps, kernel_size=n, activation="relu", name='Conv_'+str(n))(x_input)
        branch = Dropout(0.5)(branch)
        branch = MaxPooling1D(pool_size=2)(branch)
        branch = Conv1D(filters=feature_maps//2, kernel_size=n, activation="relu", name='Conv2_'+str(n))(branch)
        branch = Dropout(0.5)(branch)
        branch = MaxPooling1D(pool_size=2)(branch)
        branch = Flatten(name='Flatten2_'+str(n))(branch)
        branches.append(branch)
    return branches

def model_cnn():
    product_counts_input = Input(shape=(1,), name="product_counts")
    reviewer_counts_input = Input(shape=(1,), name="reviewer_counts")
    ratings_input = Input(shape=(5,), name="ratings")

    sent_counts_input = Input(shape=(1,), name="sent_counts")
    word_counts_input = Input(shape=(1,), name="word_counts")
    spell_ratios_input = Input(shape=(1,), name="spell_ratios")
    pos_dist_input = Input(shape=(45,), name="pos_dist")

    tfidf_input = Input(shape=(12000,), name="tfidf")
    tfidf1 = Dense(2, activation="softmax")(tfidf_input)

    embeddings_input = Input(shape=(300,300), name="embeddings")

    conv = get_conv_pool(embeddings_input)
    conv = Concatenate()(conv)
    # cnn1 = Conv1D(128, 5, activation=)(embeddings_input)
    # cnn1_pool = MaxPooling1D(pool_length=3)(cnn1)
    # cnn1_drop = Dropout(0.1)(cnn1_pool)
    # cnn2 = Conv1D(64, 5)(cnn1_drop)
    # cnn2_pool = MaxPooling1D(pool_length=3)(cnn2)
    # cnn2_drop = Dropout(0.1)(cnn2_pool)
    # cnn3 = Conv1D(32, 5)(cnn2_drop)
    # cnn3_pool = MaxPooling1D(pool_length=3)(cnn3)
    # cnn3_drop = Dropout(0.1)(cnn3_pool)
    # flat = Flatten()(cnn3_drop)

    merged = Concatenate()([product_counts_input, reviewer_counts_input, ratings_input, sent_counts_input, word_counts_input, spell_ratios_input, pos_dist_input, tfidf1, conv])

    dense1 = Dense(80, activation='relu')(merged)
    dense1_drop = Dropout(0.1)(dense1)
    dense2 = Dense(60, activation='relu')(dense1_drop)
    dense2_drop = Dropout(0.1)(dense2)
    dense3 = Dense(30, activation='relu')(dense2_drop)
    dense3_drop = Dropout(0.1)(dense3)

    output = Dense(2, activation='softmax', name="helpfulness")(dense3_drop)

    model = Model(inputs=[product_counts_input, reviewer_counts_input, ratings_input, sent_counts_input, word_counts_input, spell_ratios_input, pos_dist_input, tfidf_input, embeddings_input],
                  outputs=output)


    model.compile(optimizer="adam",
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def model_original():

    product_counts_input = Input(shape=(1,), name="product_counts")
    reviewer_counts_input = Input(shape=(1,), name="reviewer_counts")
    ratings_input = Input(shape=(5,), name="ratings")

    sent_counts_input = Input(shape=(1,), name="sent_counts")
    word_counts_input = Input(shape=(1,), name="word_counts")
    spell_ratios_input = Input(shape=(1,), name="spell_ratios")
    pos_dist_input = Input(shape=(45,), name="pos_dist")

    tfidf_input = Input(shape=(12000,), name="tfidf")
    tfidf1 = Dense(2, activation="softmax")(tfidf_input)

    embeddings_input = Input(shape=(300,300), name="embeddings")
    lstm1 = LSTM(50, return_sequences=True)(embeddings_input)
    lstm1_drop = Dropout(0.1)(lstm1)
    lstm2 = LSTM(50)(lstm1_drop)
    lstm2_drop = Dropout(0.1)(lstm2)

    merged = Concatenate()([product_counts_input, reviewer_counts_input, ratings_input, sent_counts_input, word_counts_input, spell_ratios_input, pos_dist_input, tfidf1, lstm2_drop])

    dense1 = Dense(80, activation='relu')(merged)
    dense1_drop = Dropout(0.1)(dense1)
    dense2 = Dense(60, activation='relu')(dense1_drop)
    dense2_drop = Dropout(0.1)(dense2)
    dense3 = Dense(30, activation='relu')(dense2_drop)
    dense3_drop = Dropout(0.1)(dense3)

    output = Dense(2, activation='softmax', name="helpfulness")(dense3_drop)

    model = Model(inputs=[product_counts_input, reviewer_counts_input, ratings_input, sent_counts_input, word_counts_input, spell_ratios_input, pos_dist_input, tfidf_input, embeddings_input],
                  outputs=output)

    model.compile(optimizer="adam",
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def get_model(model_name=None):

    if model_name:
        if os.path.isfile("data/models/" + model_name + ".m5"):
            from keras.models import load_model
            return load_model("data/models/" + model_name + ".m5")
    else:
        model_name = "model_1"
    encodings_input = Input(shape=(300,), name="encodings")
    encodings_drop1 = Dropout(0.1)(encodings_input)
    encodings_x1 = Dense(200, activation="relu", name="encodings_x1")(encodings_drop1)
    encodings_drop2 = Dropout(0.1)(encodings_x1)
    encodings_x2 = Dense(100, activation="relu", name="encodings_x2")(encodings_drop2)

    tags_bow_input = Input(shape=(45,), name="tags_bow")
    tags_bow_drop1 = Dropout(0.1)(tags_bow_input)
    tags_bow_x1 = Dense(2, activation="relu", name="tags_bow_x1")(tags_bow_drop1)

    sentence_len_input = Input(shape=(1,), name="sentence_lengths")
    #sentence_len_x1 = Dense(1, activation="relu", name="sentence_lengths_x1")(sentence_len_input)

    word_counts_input = Input(shape=(1,), name="word_counts")
    #word_counts_x1 = Dense(1, activation="relu", name="word_counts_x1")(word_counts_input)

    ratings_input = Input(shape=(5,), name="ratings")
    ratings_drop1 = Dropout(0.1)(ratings_input)
    ratings_x1 = Dense(1, activation="relu", name="ratings_x1")(ratings_drop1)

    merged = Concatenate()([encodings_x2, tags_bow_x1, sentence_len_input, word_counts_input, ratings_x1])
    merged_drop1 = Dropout(0.2)(merged)


    hidden_1 = Dense(100, activation="relu")(merged_drop1)
    hidden_drop1 = Dropout(0.1)(hidden_1)
    hidden_2 = Dense(26, activation="relu")(hidden_drop1)
    hidden_drop2 = Dropout(0.1)(hidden_2)
    output = Dense(2, activation="softmax", name="output")(hidden_drop2)

    model = Model(
        inputs=[encodings_input, tags_bow_input, sentence_len_input, word_counts_input, ratings_input],
        outputs=output
    )

    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])

    return model_name, model