import tensorflow as tf

def load_data():
    train_ids_file = "split_datasets/transformer_train_512_1-1/ids.tfrecord"
    train_mask_file = "split_datasets/transformer_train_512_1-1/mask.tfrecord"
    train_tokens_file = "split_datasets/transformer_train_512_1-1/tokens.tfrecord"
    train_labels_file = "split_datasets/transformer_train_512_1-1/labels.tfrecord"

    test_ids_file = "split_datasets/transformer_test_512_1-1/ids.tfrecord"
    test_mask_file = "split_datasets/transformer_test_512_1-1/mask.tfrecord"
    test_tokens_file = "split_datasets/transformer_test_512_1-1/tokens.tfrecord"
    test_labels_file = "split_datasets/transformer_test_512_1-1/labels.tfrecord"

    with open("split_datasets/transformer_train_512_1-1/dataset_size", "r", encoding='utf-8') as f:
        train_size = int(f.read())
    loader_train = DataLoader(tf.data.TFRecordDataset(train_ids_file), tf.data.TFRecordDataset(train_mask_file),
                              tf.data.TFRecordDataset(train_tokens_file), tf.data.TFRecordDataset(train_labels_file),
                              train_size)

    with open("split_datasets/transformer_test_512_1-1/dataset_size", "r", encoding='utf-8') as f:
        test_size = int(f.read())
    loader_test = DataLoader(tf.data.TFRecordDataset(test_ids_file), tf.data.TFRecordDataset(test_mask_file),
                             tf.data.TFRecordDataset(test_tokens_file), tf.data.TFRecordDataset(test_labels_file),
                             test_size)

    return loader_train, loader_test

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

