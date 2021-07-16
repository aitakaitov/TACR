import os
import numpy as np

import tensorflow
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

import neural_net.validation


def create_model(train_dir: str, test_dir: str, preprocessor, model_path=None, tb_logdir=None,
                 model_json=None, epochs=8, learning_rate=None, batch_size=32):
    """
    Creates, trains and validates a model
    :param batch_size: batch size, defaults to 32
    :param learning_rate: learning rate, defaults to 0.001 (tensorboard default)
    :param epochs: number of training epochs, defaults to 8
    :param train_dir: directory with files containing training pages
    :param test_dir: directory with files containing testing pages
    :param preprocessor: PreprocessorConfig - contains embedding matrix, Tokenizer and maximum document length
    :param model_path: Path to save the trained model. If None, model is not saved
    :param tb_logdir: Path to save the Tensorboard logs. If None, logs are not saved and displayed
    :param model_json: JSON file with the model architecture. The model_json should be used only with saved model
    configuration, as the embedding matrix dimension have to match
    :return: None
    """

    # check if we are loading model architecture from JSON or we are creating a new one
    if model_json is not None:
        with open(model_json, "r", encoding='utf-8') as f:
            model = model_from_json(f.read())

            # set the embedding weights
            try:
                embedding_layer = model.get_layer(name="embedding")
            except ValueError:
                embedding_layer = model.get_layer(name="embedding_1")

            embedding_layer.set_weights([preprocessor.emb_matrix])
    else:
        model = get_model(preprocessor)

    # if the learning rate is specified, use it, otherwise use default (0.001)
    if learning_rate is not None:
        model.compile(loss='categorical_crossentropy', optimizer=tensorflow.keras.optimizers.Adam(learning_rate=learning_rate),
                      metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    # prepare training and testing data
    X_train, y_train = get_dataset(train_dir, preprocessor.tokenizer, preprocessor.doc_max_len)
    X_test, y_test = get_dataset(test_dir, preprocessor.tokenizer, preprocessor.doc_max_len)

    # save the model architecture
    if model_path is not None:
        with open(model_path + ".json", "w+", encoding='utf-8') as json_file:
            json_file.write(model.to_json())

    # train with logging callback
    if tb_logdir is not None:
        tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=tb_logdir, histogram_freq=1)
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=[tensorboard_callback])

    # save the model
    if model_path is not None:
        model.save(model_path, overwrite=False, include_optimizer=True)

    # validate the model against the reference labels
    print("Validating the model against true labels of FP dataset")
    neural_net.validation.validate_model(model, preprocessor, "validation_datasets/validation_dataset_dev_fp",
                   "validation_datasets/validation_dataset_dev_fp_annotated")

    print("Validating the model against true labels of FP+TN dataset")
    neural_net.validation.validate_model(model, preprocessor, "validation_datasets/validation_dataset_dev_extended",
                   "validation_datasets/validation_dataset_dev_extended_annotated")

    print("Validating the model against true labels of CONTROL dataset")
    neural_net.validation.validate_model(model, preprocessor, "validation_datasets/validation_dataset_control",
                   "validation_datasets/validation_dataset_control_annotated")

    return


def get_model(preprocessor):
    """
    Contains a definition of a model, which it returns
    :return: model
    """
    embedding_dim = preprocessor.emb_matrix.shape[1]  # how large will the embedded vectors be
    input_length = preprocessor.doc_max_len  # how many words will be supplied in a document
    input_dim = len(preprocessor.tokenizer.word_index) + 1  # how long is OHE of a word
    embedding_weights = preprocessor.emb_matrix  # embedding matrix of size input_dim x embedding_dim

    # switch between sequential and parallel models
    sequential = True

    if not sequential:
        input = layers.Input(shape=(None,), dtype='int64')

        embedding_0 = layers.Embedding(
            input_dim=input_dim,
            output_dim=embedding_dim,
            input_length=input_length,
            weights=[embedding_weights],
            trainable=False
        )(input)

        reshape_0 = layers.Reshape((
            input_length,
            embedding_dim
        ))(embedding_0)

        conv_0 = layers.Conv1D(
            filters=32,
            kernel_size=12,
            kernel_initializer=tensorflow.keras.initializers.he_normal()
        )(reshape_0)

        conv_1 = layers.Conv1D(
            filters=32,
            kernel_size=5,
            kernel_initializer=tensorflow.keras.initializers.he_normal()
        )(reshape_0)

        conv_2 = layers.Conv1D(
            filters=32,
            kernel_size=3,
            kernel_initializer=tensorflow.keras.initializers.he_normal()
        )(reshape_0)

        #norm_0 = layers.BatchNormalization()(conv_0)
        #norm_1 = layers.BatchNormalization()(conv_1)
        #norm_2 = layers.BatchNormalization()(conv_2)

        act_0 = layers.Activation(tensorflow.keras.activations.relu)(conv_0)
        act_1 = layers.Activation(tensorflow.keras.activations.relu)(conv_1)
        act_2 = layers.Activation(tensorflow.keras.activations.relu)(conv_2)

        glomax_pool_0 = layers.GlobalMaxPool1D()(act_0)
        glomax_pool_1 = layers.GlobalMaxPool1D()(act_1)
        glomax_pool_2 = layers.GlobalMaxPool1D()(act_2)

        concat_0 = layers.Concatenate(axis=1)([glomax_pool_0, glomax_pool_1, glomax_pool_2])
        #dropout_0 = layers.Dropout(0.3)(concat_0)
        #flatten = layers.Flatten()(dropout_0)
        dense_0 = layers.Dense(units=24, activation='relu')(concat_0)
        output = layers.Dense(units=3, activation='softmax')(dense_0)
        model = tensorflow.keras.models.Model(input, output)
    else:

        model = Sequential()
        model.add(layers.Embedding(
            input_dim=input_dim,
            output_dim=embedding_dim,
            input_length=input_length,
            weights=[embedding_weights],
            trainable=False
        ))
        model.add(layers.Dropout(0.5, noise_shape=(None, input_length, 1)))
        model.add(layers.Conv1D(filters=16, kernel_size=4, kernel_initializer=tensorflow.keras.initializers.he_normal()))
        model.add(layers.MaxPooling1D(pool_size=6))
        model.add(layers.Activation(tensorflow.keras.activations.relu))
        model.add(layers.Conv1D(filters=16, kernel_size=8, kernel_initializer=tensorflow.keras.initializers.he_normal()))              # aktivace po poolingu relu
        #model.add(layers.MaxPooling1D(pool_size=8))
        model.add(layers.GlobalMaxPool1D())
        model.add(layers.Activation(tensorflow.keras.activations.relu))
        model.add(layers.Flatten())
        #model.add(layers.Dropout(0.2))
        model.add(layers.Dense(32, activation='relu', kernel_initializer=tensorflow.keras.initializers.he_normal()))
        #model.add(layers.Dropout(0.2))
        model.add(layers.Dense(3, activation='softmax'))

    return model


def get_dataset(dir: str, tokenizer, max_len) -> tuple:
    """
    Loads pages from a directory and returns their vectors and labels
    :param dir: directory
    :param tokenizer: Tokenizer
    :param max_len: maximum document length
    :return: Tuple of (X, y)
    """
    files = os.listdir(dir)
    X = np.zeros((len(files), max_len))
    y = np.zeros((len(files), 3))
    i = 0
    for file in files:
        with open(dir + "/" + file, "r", encoding='utf-8') as f:
            text = f.read()
            y[i][int(text[0])] = 1
            seq = tokenizer.texts_to_sequences([text[2:]])[0]
            #random.shuffle(seq)
            X[i] = pad_sequences(
                [seq]
                , maxlen=max_len, truncating="post", padding="post")
            i += 1

    return np.array(X), np.array(y)


