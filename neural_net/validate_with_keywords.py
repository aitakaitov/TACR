import os
import numpy as np
import neural_net.preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


def validate_model_control(model, config):
    # check if the config is a path or an already loaded object
    if isinstance(config, str):
        config = neural_net.preprocessing.Preprocessing().load(config)

    tokenizer = config.tokenizer
    max_len = config.doc_max_len

    # check if the model has to be loaded
    if isinstance(model, str):
        model = load_model(model)

    validation_files = os.listdir("validation_datasets/validation_dataset_control")
    validation_files.sort()

    # remove previous validation results file if it exists
    if os.path.exists("validation_results"):
        os.remove("validation_results")
    if os.path.exists("validation_differences"):
        os.remove("validation_differences")

    results_file = open("validation_results", "w+", encoding='utf-8')
    reference_file = open("validation_datasets/validation_dataset_control_annotated", "r", encoding='utf-8')
    crawler_reference_file = open("validation_datasets/validation_dataset_control_crawler", "r", encoding='utf-8')

    # prepare the files to be classified
    urls = []
    X = np.zeros((len(validation_files), max_len))
    i = 0
    for file in validation_files:
        with open(os.path.join("validation_datasets/validation_dataset_control", file), "r", encoding='utf-8') as f:
            lines = f.readlines()
            text = ""
            urls.append(lines[0][5:])
            for line in lines[2:]:
                text += line

            seq = tokenizer.texts_to_sequences([text])[0]
            X[i] = pad_sequences([seq], maxlen=max_len, truncating="post", padding="post")
            i += 1

    predictions = model.predict(X, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1,
                                use_multiprocessing=False)

    res_lines = []
    # write the classification results into a file
    for i in range(len(validation_files)):
        maximum = max(predictions[i][0], predictions[i][1], predictions[i][2])

        if predictions[i][0] == maximum:
            line = "IRRELEVANT" + "\t" + urls[i]
        elif predictions[i][1] == maximum:
            line = "COOKIES" + "\t" + urls[i]
        elif predictions[i][2] == maximum:
            line = "TERMS" + "\t" + urls[i]

        res_lines.append(line)

    crawler_res_lines = crawler_reference_file.readlines()
    for i in range(len(res_lines)):
        crawler_res_split = crawler_res_lines[i].split()

        if crawler_res_split[0] == "IRRELEVANT":
            res_lines[i] = crawler_res_lines[i]

    results_file.writelines(res_lines)
    results_file.close()

    input_lines = res_lines
    reference_lines = reference_file.readlines()

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
