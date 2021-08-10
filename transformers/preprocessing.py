import os
import random
from transformers import AutoTokenizer, TFAutoModel, PreTrainedTokenizerBase, BatchEncoding
import tensorflow as tf


def load_model(model_name):
    """
    Loads tokenizer and model
    :param model_name: repo/model
    :return: tokenizer, model
    """
    print("Loading model")
    return AutoTokenizer.from_pretrained(model_name), TFAutoModel.from_pretrained(model_name)


def create_split(pos_examples: list, neg_examples: list):
    """
    Do test train split
    :param pos_examples:
    :param neg_examples:
    :return: train, test
    """
    print("Creating split")
    random.shuffle(pos_examples)
    random.shuffle(neg_examples)
    ratio = 0.7
    pos_train = pos_examples[:int(len(pos_examples) * 0.7)]
    pos_test = pos_examples[int(len(pos_examples) * 0.7):]
    neg_train = neg_examples[:int(len(neg_examples) * 0.7)]
    neg_test = neg_examples[int(len(neg_examples) * 0.7):]
    train = pos_train + neg_train
    random.shuffle(train)
    test = pos_test + neg_test
    random.shuffle(test)

    return train, test


def load_example_paths(_dir: str):
    """
    Get file paths for examples
    :param _dir: dir
    :return: paths
    """
    print("Loading examples from " + _dir)
    files = os.listdir(_dir)
    examples = []
    for file in files:
        examples.append(_dir + "/" + file)

    return examples


def to_tensors_and_save(train_paths: list, test_paths: list, tokenizer: PreTrainedTokenizerBase):
    print("Converting input files to tensors")
    train_dir = "split_datasets/transformer_train"
    test_dir = "split_datasets/transformer_test"

    try:
        os.mkdir(train_dir)
    except OSError:
        print("Train directory already exists")
        print("Assuming sequences are preprocessed and saved")

    try:
        os.mkdir(test_dir)
    except OSError:
        print("Test directory already exists")

    tfrecord_train_ids = tf.io.TFRecordWriter(train_dir + "/ids.tfrecord")
    tfrecord_train_mask = tf.io.TFRecordWriter(train_dir + "/mask.tfrecord")
    tfrecord_train_tokens = tf.io.TFRecordWriter(train_dir + "/tokens.tfrecord")
    tfrecord_train_labels = tf.io.TFRecordWriter(train_dir + "/labels.tfrecord")
    train_dataset_size = len(train_paths)

    with open(train_dir + "/dataset_size", "w+", encoding='utf-8') as f:
        f.write(str(train_dataset_size))

    for i in range(train_dataset_size):
        with open(train_paths[i][1], "r", encoding='utf-8') as f:
            plaintext = f.read()

        x = tokenizer.encode_plus(plaintext, max_length=4096, pad_to_multiple_of=512, padding='max_length')
        x_serialized_ids = tf.io.serialize_tensor(x.data["input_ids"])
        x_serialized_types = tf.io.serialize_tensor(x.data["token_type_ids"])
        x_serialized_mask = tf.io.serialize_tensor(x.data["attention_mask"])
        x_serialized_label = tf.io.serialize_tensor(tf.fill((1, ), train_paths[i][0]))

        tfrecord_train_ids.write(x_serialized_ids.numpy())
        tfrecord_train_mask.write(x_serialized_mask.numpy())
        tfrecord_train_tokens.write(x_serialized_types.numpy())
        tfrecord_train_labels.write(x_serialized_label.numpy())

    tfrecord_test_ids = tf.io.TFRecordWriter(test_dir + "/ids.tfrecord")
    tfrecord_test_mask = tf.io.TFRecordWriter(test_dir + "/mask.tfrecord")
    tfrecord_test_tokens = tf.io.TFRecordWriter(test_dir + "/tokens.tfrecord")
    tfrecord_test_labels = tf.io.TFRecordWriter(test_dir + "/labels.tfrecord")

    test_dataset_size = len(test_paths)
    with open(test_dir + "/dataset_size", "w+", encoding='utf-8') as f:
        f.write(str(test_dataset_size))

    for i in range(test_dataset_size):
        with open(test_paths[i][1], "r", encoding='utf-8') as f:
            plaintext = f.read()

        x = tokenizer.encode_plus(plaintext, max_length=4096, pad_to_multiple_of=512, padding='max_length')
        x_serialized_ids = tf.io.serialize_tensor(x.data["input_ids"])
        x_serialized_types = tf.io.serialize_tensor(x.data["token_type_ids"])
        x_serialized_mask = tf.io.serialize_tensor(x.data["attention_mask"])
        x_serialized_label = tf.io.serialize_tensor(tf.fill((1, ), train_paths[i][0]))

        tfrecord_test_ids.write(x_serialized_ids.numpy())
        tfrecord_test_mask.write(x_serialized_mask.numpy())
        tfrecord_test_tokens.write(x_serialized_types.numpy())
        tfrecord_test_labels.write(x_serialized_label.numpy())


def main():
    tokenizer, model = load_model("UWB-AIR/Czert-B-base-cased")

    pos_paths = [(1, path) for path in load_example_paths("raw_datasets/merged_positive/relevant_with_p")]
    neg_paths = [(0, path) for path in load_example_paths("raw_datasets/merged_negative/relevant_with_p")]
    train_paths, test_paths = create_split(pos_paths, neg_paths)
    to_tensors_and_save(train_paths, test_paths, tokenizer)



main()

