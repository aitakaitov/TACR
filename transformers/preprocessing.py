import os
import random
from transformers import AutoTokenizer, TFAutoModel, PreTrainedTokenizerBase, LongformerTokenizer, TFLongformerPreTrainedModel
import tensorflow as tf


def load_model(model_name):
    """
    Loads tokenizer and model
    :param model_name: repo/model
    :return: tokenizer, model
    """
    print("Loading model")
    return AutoTokenizer.from_pretrained(model_name)


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
    # split positive and negative examples into train and test sets
    pos_train = pos_examples[:int(len(pos_examples) * ratio)]
    pos_test = pos_examples[int(len(pos_examples) * ratio):]
    neg_train = neg_examples[:int(len(neg_examples) * ratio)]
    neg_test = neg_examples[int(len(neg_examples) * ratio):]
    # merge train and test sets and shuffle them
    train = pos_train + neg_train
    random.shuffle(train)
    test = pos_test + neg_test
    random.shuffle(test)

    return train[:5], test[:3]


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


def to_example(ids, mask, tokens, label):
    """
    Converts the processed example into TF Example
    :param ids: ids
    :param mask: mask
    :param tokens: tokens
    :param label: label
    :return: Example
    """
    # define the example data
    # we need to first serialize the tensors (arrays), then convert them into ByteLists, which can then
    # be wrapped in a Feature object
    # then we place all the Features into a Dict
    data = {
        'ids': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(ids).numpy()])),
        'mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(mask).numpy()])),
        'tokens': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tokens).numpy()])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(label).numpy()]))
    }

    # create an Example based on the dict, the serialize it so it can be written into TFRecord file
    return tf.train.Example(features=tf.train.Features(feature=data)).SerializeToString()


def to_tensors_and_save(train_paths: list, test_paths: list, tokenizer: PreTrainedTokenizerBase):
    """
    Processes examples and saves them into TFRecord files
    :param train_paths: train paths
    :param test_paths: test paths
    :param tokenizer: tokenizer
    :return: None
    """
    print("Converting input files to tensors")
    print("Train")


    # where to save them
    _dir = "split_datasets/dataset_new_notrunc_small"
    try:
        os.mkdir(_dir)
    except OSError:
        print("Train directory already exists")

    # TFRecordWriter for training examples
    tfrecord_train = tf.io.TFRecordWriter(_dir + "/train.tfrecord")

    # for each example
    for i in range(len(train_paths)):
        # read example file - path is at [1]
        with open(train_paths[i][1], "r", encoding='utf-8') as f:
            plaintext = f.read()

        # tokenize the example
        x = tokenizer(plaintext)
        # convert the processed example into TF Example
        example = to_example(x.data['input_ids'], x.data['token_type_ids'], x.data['attention_mask'],
                             tf.fill((1, ), train_paths[i][0]))
        # write the Example into the TFRecord file
        tfrecord_train.write(example)

    # do the same for test
    print("Test")
    tfrecord_test = tf.io.TFRecordWriter(_dir + "/test.tfrecord")

    for i in range(len(test_paths)):
        with open(test_paths[i][1], "r", encoding='utf-8') as f:
            plaintext = f.read()

        x = tokenizer(plaintext)
        example = to_example(x.data['input_ids'], x.data['token_type_ids'], x.data['attention_mask'],\
                             tf.fill((1, ), test_paths[i][0]))
        tfrecord_test.write(example)


def main():
    tokenizer = load_model("UWB-AIR/Czert-B-base-cased")
    # positive example paths have y=1
    pos_paths = [(1, path) for path in load_example_paths("raw_datasets/merged_positive/relevant_with_p")]
    # negative example paths have y=0
    neg_paths = [(0, path) for path in load_example_paths("raw_datasets/merged_negative/relevant_with_p")]
    # split the paths into train and test datasets
    train_paths, test_paths = create_split(pos_paths, neg_paths)
    # convert to tensors and save examples
    to_tensors_and_save(train_paths, test_paths, tokenizer)


main()