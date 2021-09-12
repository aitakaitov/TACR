from transformers import TFAutoModel
import tensorflow as tf


class LongBert2(tf.keras.Model):
    def __init__(self):
        super(LongBert2, self).__init__()
        self.bert = load_model("UWB-AIR/Czert-B-base-cased")
        #self.lstm = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(100))
        #self.pooling = tf.keras.layers.GlobalMaxPool1D(name='pool')
        #self.pooling = tf.keras.layers.GlobalMaxPooling1D()
        self.dense_1 = tf.keras.layers.Dense(24, activation='relu', name='classifier')
        self.dense_output = tf.keras.layers.Dense(1, activation='sigmoid', name='output', dtype='float32')

    def call(self, inputs, training=False, **kwargs):
        x = self.bert(inputs, training=training).pooler_output
        x = self.dense_1(x)
        return self.dense_output(x)


def load_model(model_name):
    """
    Loads tokenizer and model
    :param model_name: repo/model
    :return: tokenizer, model
    """
    print("Loading model")
    return TFAutoModel.from_pretrained(model_name)


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


def load_data():
    """
    Creates TFRecordReaders for train and test
    :return: train, test
    """
    train_file = "split_datasets/dataset_new_small/train.tfrecord"
    test_file = "split_datasets/dataset_new_small/test.tfrecord"

    return tf.data.TFRecordDataset(train_file).map(parse_element).cache(), tf.data.TFRecordDataset(test_file).map(parse_element).cache()


def main():
    model = LongBert2()
    dataset_train, dataset_test = load_data()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.000001)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    train_acc_metric = tf.keras.metrics.BinaryAccuracy()
    test_acc_metric = tf.keras.metrics.BinaryAccuracy()

    summary_writer_train = tf.summary.create_file_writer("logs-512/train")
    summary_writer_test = tf.summary.create_file_writer("logs-512/test")

    logfile = open("log-512", "w+", encoding='utf-8')

    epochs = 5
    batch_size = 1

    logfile.writelines(["Epochs = " + str(epochs), "Batch size = " + str(batch_size)])
    # Iterate over epochs.
    i = 0
    for epoch in range(epochs):

        logfile.writelines(["Starting epoch " + str(epoch + 1)])
        print("[START OF EPOCH " + str(epoch + 1) + "]")

        dataset_train_batch = dataset_train.batch(batch_size).prefetch(1)

        # Iterate over the batches of the dataset.
        for batch in dataset_train_batch:
            #tf.profiler.experimental.start("profile-logs")

            with tf.GradientTape() as tape:
                if i % 1000 == 0:
                    logfile.writelines(["epoch " + str(epoch + 1) + " - batch " + str(i / batch_size + 1)])
                result = model(batch[0], training=True)
                loss = loss_fn(batch[1], result)

            train_acc_metric.update_state(batch[1], result)
            print("--- loss: " + str(float(loss)))
            print("--- accuracy: " + str(float(train_acc_metric.result())))

            if i % 1000 == 0:
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

        dataset_test_batch = dataset_test.batch(1).prefetch(1)

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
