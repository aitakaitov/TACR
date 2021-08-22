from transformers import TFAutoModel
import tensorflow as tf


class LongBert2(tf.keras.Model):
    def __init__(self):
        super(LongBert2, self).__init__()
        self.bert = load_model("UWB-AIR/Czert-B-base-cased")
        self.lstm = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(100))
        self.pooling = tf.keras.layers.GlobalMaxPool1D(name='pool')
        self.dense_1 = tf.keras.layers.Dense(24, activation='relu', name='classifier')
        self.dense_output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')

    def call(self, inputs, training=False, **kwargs):
        # inputs = (ids, mask, tokens), each shape (batch, 4096)
        # split data into 512 blocks
        x = self.split_data(inputs)
        # x = eight blocks of shape (batch, 512) for each of the three inputs

        # process each of the blocks
        # for each input (batch, 512) we get a (batch, 768) result
        results = []
        for block in x:
            results.append(self.bert(block).pooler_output)

        # concat blocks
        # we concatenate the blocks, creating a tensor (batch, 8, 768)
        concatenated = tf.stack(results, axis=1)

        x = self.lstm(concatenated)
        x = self.dense_1(x)
        return self.dense_output(x)

    def split_data(self, x) -> list:
        # split each tensor into 8 blocks of 512 tokens - we get 8 blocks with shape (batch, 512) for each input (ids, mask, tokens)
        new_ids = tf.split(x[0], [512, 512, 512], axis=-1)
        new_mask = tf.split(x[1], [512, 512, 512], axis=-1)
        new_tokens = tf.split(x[2], [512, 512, 512], axis=-1)

        # return list of tuples of (ids, mask, tokens), each 512 tokens
        out = []
        #for i in range(len(new_ids)):
        for i in range(3):
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


class DataLoader:
    def __init__(self, ids_record, mask_record, tokens_record, labels_record, dataset_size):
        self.ids_dataset = ids_record
        self.ids = iter(ids_record)
        self.mask_dataset = mask_record
        self.mask = iter(mask_record)
        self.tokens_dataset = tokens_record
        self.tokens = iter(tokens_record)
        self.labels_dataset = labels_record
        self.labels = iter(labels_record)
        self.dataset_size = dataset_size
        self.number_taken = 0

    def reset(self):
        self.number_taken = 0
        self.ids = iter(self.ids_dataset)
        self.mask = iter(self.mask_dataset)
        self.tokens = iter(self.tokens_dataset)
        self.labels = iter(self.labels_dataset)

    def get_sample(self):
        #for record in self.ids.skip(self.number_taken).take(1):
        #    ids_tensor = tf.io.parse_tensor(record, tf.int32)
        #for record in self.mask.skip(self.number_taken).take(1):
        #    mask_tensor = tf.io.parse_tensor(record, tf.int32)
        #for record in self.tokens.skip(self.number_taken).take(1):
        #    tokens_tensor = tf.io.parse_tensor(record, tf.int32)
        #for record in self.labels.skip(self.number_taken).take(1):
        #    label_tensor = tf.io.parse_tensor(record, tf.int32)

        try:
            ids_tensor = tf.io.parse_tensor(self.ids.get_next(), tf.int32)
            mask_tensor = tf.io.parse_tensor(self.mask.get_next(), tf.int32)
            tokens_tensor = tf.io.parse_tensor(self.tokens.get_next(), tf.int32)
            label_tensor = tf.io.parse_tensor(self.labels.get_next(), tf.int32)
        except tf.errors.OutOfRangeError:
            return (None, None, None), None
        self.number_taken += 1

        return (ids_tensor, mask_tensor, tokens_tensor), label_tensor

    def get_batch(self, batch_size):
        samples_X_ids = []
        samples_X_mask = []
        samples_X_tokens = []
        samples_y = []
        for i in range(batch_size):
            if self.number_taken == self.dataset_size:
                break

            sample = self.get_sample()
            if sample[1] is None:
                break

            samples_X_ids.append(sample[0][0])
            samples_X_mask.append(sample[0][1])
            samples_X_tokens.append(sample[0][2])
            samples_y.append(sample[1])

        if len(samples_y) == 0:
            return (None, None, None), None
        else:
            return (tf.stack(samples_X_ids, axis=0), tf.stack(samples_X_mask, axis=0), tf.stack(samples_X_tokens, axis=0)),\
                    tf.stack(samples_y, axis=0)


def load_data():
    train_ids_file = "split_datasets/transformer_train_1536/ids.tfrecord"
    train_mask_file = "split_datasets/transformer_train_1536/mask.tfrecord"
    train_tokens_file = "split_datasets/transformer_train_1536/tokens.tfrecord"
    train_labels_file = "split_datasets/transformer_train_1536/labels.tfrecord"

    test_ids_file = "split_datasets/transformer_test_1536/ids.tfrecord"
    test_mask_file = "split_datasets/transformer_test_1536/mask.tfrecord"
    test_tokens_file = "split_datasets/transformer_test_1536/tokens.tfrecord"
    test_labels_file = "split_datasets/transformer_test_1536/labels.tfrecord"

    with open("split_datasets/transformer_train_1536/dataset_size", "r", encoding='utf-8') as f:
        train_size = int(f.read())
    loader_train = DataLoader(tf.data.TFRecordDataset(train_ids_file), tf.data.TFRecordDataset(train_mask_file),
                              tf.data.TFRecordDataset(train_tokens_file), tf.data.TFRecordDataset(train_labels_file),
                              train_size)

    with open("split_datasets/transformer_test_1536/dataset_size", "r", encoding='utf-8') as f:
        test_size = int(f.read())
    loader_test = DataLoader(tf.data.TFRecordDataset(test_ids_file), tf.data.TFRecordDataset(test_mask_file),
                             tf.data.TFRecordDataset(test_tokens_file), tf.data.TFRecordDataset(test_labels_file),
                             test_size)

    return loader_train, loader_test


def main():
    model = LongBert2()
    loader_train, loader_test = load_data()

    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    train_acc_metric = tf.keras.metrics.BinaryAccuracy()
    test_acc_metric = tf.keras.metrics.BinaryAccuracy()

    summary_writer_train = tf.summary.create_file_writer("logs/train")
    summary_writer_test = tf.summary.create_file_writer("logs/test")

    logfile = open("log", "w+", encoding='utf-8')

    epochs = 2
    batch_size = 2

    logfile.writelines(["Epochs = " + str(epochs), "Batch size = " + str(batch_size)])
    # Iterate over epochs.
    for epoch in range(epochs):
        logfile.writelines(["Starting epoch " + str(epoch + 1)])
        print("[START OF EPOCH " + str(epoch + 1) + "]")
        # Iterate over the batches of the dataset.
        for i in range(int(loader_train.dataset_size / batch_size)):
            with tf.GradientTape() as tape:
                X, y = loader_train.get_batch(batch_size)
                if i % 200 == 0:
                    logfile.writelines(["epoch " + str(epoch + 1) + " - batch " + str(i / batch_size + 1)])
                if y is None:
                    break
                result = model(X, training=True)
                loss = loss_fn(y, result)

            train_acc_metric.update_state(y, result)
            print("--- loss: " + str(float(loss)))
            print("--- accuracy: " + str(float(train_acc_metric.result())))

            if i % 200 == 0:
                logfile.writelines(["-- loss: " + str(float(loss)), "-- accuracy: " + str(float(train_acc_metric.result()))])

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            with summary_writer_train.as_default():
                with tf.name_scope('loss'):
                    tf.summary.scalar('bin_ce', loss, step=epoch * loader_train.dataset_size + i * batch_size)
                with tf.name_scope('accuracy'):
                    tf.summary.scalar('accuracy', float(train_acc_metric.result()), step=epoch * loader_train.dataset_size + i * batch_size)

        train_acc_metric.reset_states()

        print("--- VALIDATION")
        for i in range(int(loader_test.dataset_size / batch_size)):
            X, y = loader_test.get_batch(batch_size)
            if y is None:
                break
            result = model(X, training=False)
            test_acc_metric.update_state(y, result)

        loader_test.reset()
        loader_train.reset()
        with summary_writer_test.as_default():
            with tf.name_scope('accuracy'):
                tf.summary.scalar('accuracy', float(test_acc_metric.result()), step=(epoch+1) * loader_train.dataset_size)

        print("--- accuracy: " + str(float(test_acc_metric.result())))

        logfile.writelines(["Epoch " + str(epoch + 1) + " validation", "-- accuracy:" + str(float(test_acc_metric.result()))])
        test_acc_metric.reset_states()

        loader_test.reset()
        loader_train.reset()


main()
