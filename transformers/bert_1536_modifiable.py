from transformers import TFAutoModel
import tensorflow as tf


class LongBert2(tf.keras.Model):
    def __init__(self):
        super(LongBert2, self).__init__()
        self.bert = load_model("UWB-AIR/Czert-B-base-cased")
        self.bert.config.attention_probs_dropout_prob = ?
        self.bert.config.hidden_dropout_prob = ?
        self.lstm = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(?, dropout=?, recurrent_dropout=?))
        self.dropout_1 = tf.keras.layers.Dropout(?)
        self.dense_1 = tf.keras.layers.Dense(?, activation='?', name='classifier')
        self.dropout_2 = tf.keras.layers.Dropout(?)
        self.dense_output = tf.keras.layers.Dense(1, activation='?', name='output')

    def call(self, inputs, training=False, **kwargs):
        # inputs = (ids, mask, tokens), each shape (batch, 4096)
        # split data into 512 blocks
        x = self.split_data(inputs)
        # x = eight blocks of shape (batch, 512) for each of the three inputs

        # process each of the blocks
        # for each input (batch, 512) we get a (batch, 768) result
        results = []
        for block in x:
            results.append(self.bert(block, training=training).pooler_output)

        # concat blocks
        # we concatenate the blocks, creating a tensor (batch, 8, 768)
        concatenated = tf.stack(results, axis=1)

        x = self.lstm(concatenated)
        x = self.dropout_1(x, training=training)
        x = self.dense_1(x)
        x = self.dropout_2(x, training=training)
        return self.dense_output(x)

    def split_data(self, x) -> list:
        # split each tensor into 3 blocks of 512 tokens - we get 3 blocks with shape (batch, 512) for each input (ids, mask, tokens)
        new_ids = tf.split(x[0], [512, 512, 512], axis=-1)
        new_mask = tf.split(x[1], [512, 512, 512], axis=-1)
        new_tokens = tf.split(x[2], [512, 512, 512], axis=-1)

        # return list of tuples of (ids, mask, tokens), each 512 tokens
        out = []
        for i in range(len(new_ids)):
            out.append((new_ids[i], new_mask[i], new_tokens[i]))

        return out


def load_model(model_name):
    """
    Loads tokenizer and model
    :param model_name: repo/model
    :return: tokenizer, model
    """
    print("Loading model")
    return TFAutoModel.from_pretrained(model_name)


def load_data():
    """
    Creates TFRecordReaders for train and test
    :return: train, test
    """
    train_file = "split_datasets/dataset_new_small/train.tfrecord"
    test_file = "split_datasets/dataset_new_small/test.tfrecord"

    return tf.data.TFRecordDataset(train_file).map(parse_element), tf.data.TFRecordDataset(test_file).map(parse_element)


# feature descriptor for example parsing
fd = {
        "ids": tf.io.FixedLenFeature([], tf.string),
        "mask": tf.io.FixedLenFeature([], tf.string),
        "tokens": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.string)
    }


def parse_element(el):
    """
    For mapping dataset
    Parses serialized Example and returns example for the model
    :param el: element
    :return: (ids, mask, tokens), label
    """

    # parse example given element and feature descriptor
    content = tf.io.parse_single_example(el, fd)

    # convert the serialized tensors from the example into tensors
    ids = tf.io.parse_tensor(content['ids'], out_type=tf.int32)
    mask = tf.io.parse_tensor(content['mask'], out_type=tf.int32)
    tokens = tf.io.parse_tensor(content['tokens'], out_type=tf.int32)
    label = tf.io.parse_tensor(content['label'], out_type=tf.int32)

    return (ids, mask, tokens), label


def main():
    model = LongBert2()
    dataset_train, dataset_test = load_data()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    train_acc_metric = tf.keras.metrics.BinaryAccuracy()
    test_acc_metric = tf.keras.metrics.BinaryAccuracy()

    summary_writer_train = tf.summary.create_file_writer("logs-!/train")
    summary_writer_test = tf.summary.create_file_writer("logs-!/test")

    logfile = open("log-!", "w+", encoding='utf-8')

    epochs = 5
    batch_size = 1

    logfile.writelines(["Epochs = " + str(epochs), "Batch size = " + str(batch_size)])
    # Iterate over epochs.
    i = 0
    for epoch in range(epochs):

        logfile.writelines(["Starting epoch " + str(epoch + 1)])
        print("[START OF EPOCH " + str(epoch + 1) + "]")

        dataset_train_batch = dataset_train.batch(batch_size)

        # Iterate over the batches of the dataset.
        for batch in dataset_train_batch:
            #tf.profiler.experimental.start("profile-logs")

            with tf.GradientTape() as tape:
                if i % 200 == 0:
                    logfile.writelines(["epoch " + str(epoch + 1) + " - batch " + str(i / batch_size + 1)])
                result = model(batch[0], training=True)
                loss = loss_fn(batch[1], result)

            train_acc_metric.update_state(batch[1], result)
            print("--- loss: " + str(float(loss)))
            print("--- accuracy: " + str(float(train_acc_metric.result())))

            if i % 200 == 0:
                logfile.writelines(["-- loss: " + str(float(loss)), "-- accuracy: " + str(float(train_acc_metric.result()))])
                logfile.flush()

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            #tf.profiler.experimental.stop()

            with summary_writer_train.as_default():
                with tf.name_scope('loss'):
                    tf.summary.scalar('bin_ce', loss, step=i)
                with tf.name_scope('accuracy'):
                    tf.summary.scalar('accuracy', float(train_acc_metric.result()), step=i)
            i += 1

        train_acc_metric.reset_states()

        dataset_test_batch = dataset_test.batch(1)

        print("--- VALIDATION")
        for el in dataset_test_batch:
            result = model(el[0], training=False)
            test_acc_metric.update_state(el[1], result)

        with summary_writer_test.as_default():
            with tf.name_scope('accuracy'):
                tf.summary.scalar('accuracy', float(test_acc_metric.result()), step=i)

        print("--- accuracy: " + str(float(test_acc_metric.result())))

        logfile.writelines(["Epoch " + str(epoch + 1) + " validation", "-- accuracy:" + str(float(test_acc_metric.result()))])
        test_acc_metric.reset_states()


main()
