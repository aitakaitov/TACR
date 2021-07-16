import os
import random

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow
import neural_net.preprocessing


def validate_model(model, config, validation_dataset_path: str, reference_file_path: str):
    """
    Validates a model against reference labels
    Creates a file with differences, so that it is shown where the model fails
    Also prints a confusion matrix with true labels as columns and predictions as rows
    Calculates and prints precision, recall and f-score
    :param model: Can be a path to a saved model or an already created instance
    :param validation_dataset_path: path to the directory with validation pages
    :param config: Can be a path to a pickled preprocessing configuration, or an already created instance of
    PreprocessingConfig
    :param reference_file_path: path to the reference file to compare the model against
    """

    # check if the config is a path or an already loaded object
    if isinstance(config, str):
        config = neural_net.preprocessing.Preprocessing().load(config)

    tokenizer = config.tokenizer
    max_len = config.doc_max_len

    # check if the model has to be loaded
    if isinstance(model, str):
        model = load_model(model, compile=False)

    # load the validation pages
    validation_files = os.listdir(validation_dataset_path)
    validation_files.sort()

    # prepare the files to be classified
    urls = []
    X = np.zeros((len(validation_files), max_len))
    i = 0
    for file in validation_files:
        with open(os.path.join(validation_dataset_path, file), "r", encoding='utf-8') as f:
            lines = f.readlines()
            text = ""
            urls.append(lines[0][5:])
            for line in lines[2:]:
                text += line

            seq = tokenizer.texts_to_sequences([text])[0]
            X[i] = pad_sequences([seq], maxlen=max_len, truncating="post", padding="post")
            i += 1

    # remove previous validation results file if it exists
    if os.path.exists("validation_results"):
        os.remove("validation_results")
    if os.path.exists("validation_differences"):
        os.remove("validation_differences")

    # create the results file and make the predictions
    res_file = open("validation_results", "w+", encoding='utf-8')
    predictions = model.predict(X, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1,
                                use_multiprocessing=False)

    # write the classification results into a file
    for i in range(len(validation_files)):
        maximum = max(predictions[i][0], predictions[i][1], predictions[i][2])

        if predictions[i][0] == maximum:
            line = "IRRELEVANT" + "\t" + urls[i]
            res_file.writelines([line])
        elif predictions[i][1] == maximum:
            line = "COOKIES" + "\t" + urls[i]
            res_file.writelines([line])
        elif predictions[i][2] == maximum:
            line = "TERMS" + "\t" + urls[i]
            res_file.writelines([line])

    res_file.close()

    # load the results and references
    input_file = open("validation_results", "r", encoding='utf-8')
    reference_file = open(reference_file_path, "r", encoding='utf-8')

    input_lines = input_file.readlines()
    reference_lines = reference_file.readlines()

    input_file.close()
    reference_file.close()

    # the differences will be written into this file
    diff_file = open("validation_differences", "w+", encoding='utf-8')

    # prepare the confusion matrix
    ref_index = dict()
    ref_index["TERMS"] = 0
    ref_index["TERMS-OTHER"] = 0
    ref_index["COOKIES"] = 1
    ref_index["IRRELEVANT"] = 2
    conf_matrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, "X"]]

    for i in range(len(reference_lines)):
        input_line = input_lines[i].split()
        reference_line = reference_lines[i].split()
        conf_matrix[ref_index[input_line[0]]][ref_index[reference_line[0]]] += 1

        # if the classification and reference arent the same, write the difference
        if input_line[0] != reference_line[0]:
            diff_file.write(
                "Predicted: {}\tReference: {}\t URL: {}\n".format(input_line[0], reference_line[0], input_line[1]))

    diff_file.close()

    # calculate the precision and recall for each row and col
    for i in range(3):
        try:
            conf_matrix[i][3] = conf_matrix[i][i] / (conf_matrix[i][0] + conf_matrix[i][1] + conf_matrix[i][2])
        except ZeroDivisionError:
            conf_matrix[i][3] = "NaN"

    for i in range(3):
        try:
            conf_matrix[3][i] = conf_matrix[i][i] / (conf_matrix[0][i] + conf_matrix[1][i] + conf_matrix[2][i])
        except ZeroDivisionError:
            conf_matrix[3][i] = "NaN"

    # print the confusion matrix
    col_width = 20
    for row in conf_matrix:
        print("".join(str(word).ljust(col_width) for word in row))

    # calculate the average precision and recall + the f-score
    precisions = []
    recalls = []

    for i in range(3):
        try:
            row_prec = conf_matrix[i][i] / (conf_matrix[i][0] + conf_matrix[i][1] + conf_matrix[i][2])
        except ZeroDivisionError:
            row_prec = "NaN"
        precisions.append(row_prec)

    for i in range(3):
        try:
            col_rec = conf_matrix[i][i] / (conf_matrix[0][i] + conf_matrix[1][i] + conf_matrix[2][i])
        except ZeroDivisionError:
            col_rec = "NaN"
        recalls.append(col_rec)

    try:
        precision = (precisions[0] + precisions[1] + precisions[2]) / 3
        recall = (recalls[0] + recalls[1] + recalls[2]) / 3
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F-score: {}".format(2 * (precision * recall) / (precision + recall)))
    except TypeError:
        print("Division by zero, metrics cannot be calculated")


