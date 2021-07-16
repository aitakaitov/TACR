import neural_net.network
import neural_net.preprocessing
from sys import argv

"""
Script to load a saved model architecture, preprocessing configuration, train the model and validate it
"""

if len(argv) != 2:
    print("Specify the model (folder model_configurations)")
    exit()

configuration_path = "model_configurations/" + argv[1]
configuration_file = open(configuration_path, "r", encoding='utf-8')
lines = configuration_file.readlines()

model_json = "model_configurations/model_jsons/" + lines[0][:-1]
preprocessing_configuration = "model_configurations/tokenizer_configurations/" + lines[1][:-1]
train_dir = "split_datasets/" + lines[2][:-1]
test_dir = "split_datasets/" + lines[3][:-1]
epochs = int(lines[4][:-1])
learning_rate = float(lines[5][:-1])
batch_size = int(lines[6])

neural_net.network.create_model(train_dir, test_dir, neural_net.preprocessing.Preprocessing().load(preprocessing_configuration),
                                "saved-model", "tensorboard", model_json, epochs, learning_rate, batch_size)


