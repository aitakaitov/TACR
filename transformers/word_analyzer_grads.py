import operator
import json
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel, BertTokenizer, TFBertModel, BatchEncoding


class LongBert2(tf.keras.Model):
    def __init__(self):
        super(LongBert2, self).__init__()
        self.bert = TFAutoModel.from_pretrained("UWB-AIR/Czert-B-base-cased")
        self.bert.config.attention_probs_dropout_prob = 0.1
        self.bert.config.hidden_dropout_prob = 0.1
        self.lstm = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(100, dropout=0.0, recurrent_dropout=0.0))
        self.dropout_1 = tf.keras.layers.Dropout(0.0)
        self.dense_1 = tf.keras.layers.Dense(24, activation='relu', name='classifier')
        self.dropout_2 = tf.keras.layers.Dropout(0.0)
        self.dense_output = tf.keras.layers.Dense(1, activation='sigmoid', name='output', dtype='float32')

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

    def get_config(self):
        super(LongBert2, self).get_config()


def get_model():
    model = LongBert2()
    model.load_weights("saved-weights-1")
    #return LongBert2().load_weights(filepath=model_path), AutoTokenizer.from_pretrained("UWB-AIR/Czert-B-base-cased")
    return model, AutoTokenizer.from_pretrained("UWB-AIR/Czert-B-base-cased")


def get_baseline_acc(encoded, model):
    res = model(encoded, training=False)
    return float(res)


def classify_example(_input, model):
    el = (
        tf.expand_dims(tf.convert_to_tensor(_input[0]), axis=0),
        tf.expand_dims(tf.convert_to_tensor(_input[1]), axis=0),
        tf.expand_dims(tf.convert_to_tensor(_input[2]), axis=0)
    )
    with tf.GradientTape() as tape:
        res = model(el, training=False)
    return float(res), tape.gradient(res, model.trainable_weights)


def process_plaintext(plaintext: str, tokenizer: BertTokenizer, model: TFBertModel):
    tokenized = tokenizer.tokenize(plaintext)
    sequences = preprocess_tokenized(tokenized)
    encoded = tokenizer(plaintext, max_length=1536, truncation=True, padding='max_length')
    baseline_acc, grads = classify_example((encoded.data['input_ids'], encoded.data['attention_mask'], encoded.data['token_type_ids']), model)

    embeddings = grads[0].values.numpy().tolist()
    tokens = grads[1].values.numpy().tolist()
    attention = grads[2].values.numpy().tolist()

    print("Baseline: " + str(baseline_acc) + "\n")
    results = []
    for n in [1,2,3,4,5,6,7,8,9,10]:
        results = results + process_ngrams((embeddings, tokens, attention), sequences, baseline_acc, n)

    print("Processing results\n")
    print(len(results))
    with open("results-grads-pos-short-tokens.json", "w+", encoding='utf-8') as f:
        f.write(json.dumps(results))
    with open("tokenized-grads-pos-short-tokens.json", "w+", encoding='utf-8') as f:
        f.write(json.dumps(tokenized))
    return


def preprocess_tokenized(tokenized: list):
    word_indices = []
    sequences = []
    words = []
    word = ""
    for i in range(min(len(tokenized), 1534)):
        if "##" in tokenized[i]:
            word += tokenized[i][2:]
        else:
            word_indices.append(i + 1)
            words.append(word)
            word = tokenized[i]

    sequences.append(word_indices)
    return sequences


def process_ngrams(encoded, sequences: list, baseline_acc: float, n: int):
    # token type ids all 0
    # attention mask 1 where input ids not 0
    print("Processing " + str(n) + "-grams")
    # beg_index = 1
    # end_index = encoded.data['input_ids'].index(3)
    output = []

    total_len = 0
    for sequence in sequences:
        seq_length = len(sequence)
        total_len += sequence[len(sequence) - 1]
        for i in range(seq_length):
            if i + n >= seq_length:
                break
            if i + n == seq_length:
                try:
                    res = 0
                    res += sum_range(encoded[0], sequence[i], sequence[i + n])
                    #res += sum_range(encoded[1], sequence[i], sequence[i + n])
                    res += sum_range(encoded[2], sequence[i], sequence[i + n])
                except IndexError:
                    pass
                output.append([sequence[i], n, res])
                break
            else:
                res = 0
                res += sum_range(encoded[0], sequence[i], sequence[i + n])
                #res += sum_range(encoded[1], sequence[i], sequence[i + n])
                res += sum_range(encoded[2], sequence[i], sequence[i + n])
                output.append([sequence[i], n, res])

    return output


def sum_range(values, beg, end):
    res = 0
    for i in range(beg, end):
        for j in range(len(values[i])):
            if values[i][j] is None:
                pass
            res += values[i][j]
    return res
    # for i in range(beg_index, end_index):
    #     modified = [encoded.data['input_ids'].copy(), encoded.data['attention_mask'].copy(), encoded.data['token_type_ids'].copy()]
    #     if i + n < end_index:
    #         modified = mask(i, n, modified)
    #     else:
    #         modified = mask(i, end_index - i, modified)
    #         acc = classify_example(modified, model)
    #         output.append([i, end_index - i, acc - baseline_acc])
    #         break
    #
    #     acc = classify_example(modified, model)
    #     print(str(acc))
    #     output.append([i, n, acc - baseline_acc])
    #
    # return output


def test():
    #text = 'Říkáte, že se aplikace internetového bankovnictví. MojeBanka vyvíjí takřka nepřetržitě.'
    text = open("input_pos_short.txt", "r", encoding='utf-8').read()
    model, tokenizer = get_model()
    process_plaintext(text, tokenizer, model)

test()