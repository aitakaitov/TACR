import json
import operator
import threading
from abc import ABC

import PySimpleGUI
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel, BertTokenizer, TFBertModel
from sys import argv
import webview
import numpy as np
from alibi.explainers import IntegratedGradients


# -------------------------------------------------------------- NEURAL NETWORK FOR EMBEDDINGS


# -------------------------------------------------------------- NEURAL NETWORK FOR INTEGRAL GRADIENTS LIB IMPL.

class LongBert2ForIGLib(tf.keras.Model):
    def __init__(self):
        super(LongBert2ForIGLib, self).__init__()
        self.bert = TFAutoModel.from_pretrained("UWB-AIR/Czert-B-base-cased")
        self.bert.config.attention_probs_dropout_prob = 0.1
        self.bert.config.hidden_dropout_prob = 0.1
        self.lstm = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(100, dropout=0.0, recurrent_dropout=0.0))
        self.dropout_1 = tf.keras.layers.Dropout(0.0)
        self.dense_1 = tf.keras.layers.Dense(24, activation='relu', name='classifier')
        self.dropout_2 = tf.keras.layers.Dropout(0.0)
        self.dense_output = tf.keras.layers.Dense(1, activation='sigmoid', name='output', dtype='float32')

    def call(self, inputs, training=False, **kwargs):
        if isinstance(inputs, tf.Tensor):
            attention_mask = tf.expand_dims(tf.convert_to_tensor([1 for _ in range(1536)]), axis=0)
            token_type_ids = tf.expand_dims(tf.convert_to_tensor([0 for _ in range(1536)]), axis=0)
            inputs = (inputs, attention_mask, token_type_ids)

        if "attention_mask" in kwargs.keys() or "token_type_ids" in kwargs.keys():
            print("IN CALL CHECK")
            input_ids = inputs[0]
            attention_mask = kwargs['attention_mask']
            token_type_ids = kwargs['token_type_ids']
            inputs = [input_ids, attention_mask, token_type_ids]

        print("Input IDS: " + str(inputs[0].shape))
        print("Token IDS: " + str(inputs[2].shape))
        print("Attention mask: " + str(inputs[1].shape))

        x = self.split_data(inputs)
        print("Split OK: " + str(np.array(x).shape))
        print("Dtype: " + str(np.array(x).dtype))
        # x = eight blocks of shape (batch, 512) for each of the three inputs

        # process each of the blocks
        # for each input (batch, 512) we get a (batch, 768) result
        results = []
        i = 0
        for block in x:
            print(np.array(block).shape)
            print(block)
            results.append(self.bert(block, training=training).pooler_output)
            print("BERT iteration")
            i += 1
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


# ---------------------------------------------------------------------------- NEURAL NETWORK FOR CLASSIFICATION

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
        x = self.split_data(inputs)

        # process each of the blocks
        # for each input (batch, 512) we get a (batch, 768) result
        results = []
        i = 0
        for block in x:
            results.append(self.bert(block, training=training).pooler_output)
            i += 1
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


# -------------------------------------------------------------------------- NEURAL NETWORK FOR ATTENTION OUTPUT

class LongBert2ForAttention(tf.keras.Model, ABC):
    def __init__(self):
        super(LongBert2ForAttention, self).__init__()
        self.bert = TFAutoModel.from_pretrained("UWB-AIR/Czert-B-base-cased")
        self.bert.config.attention_probs_dropout_prob = 0.1
        self.bert.config.hidden_dropout_prob = 0.1
        self.lstm = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(100, dropout=0.0, recurrent_dropout=0.0))
        self.dropout_1 = tf.keras.layers.Dropout(0.0)
        self.dense_1 = tf.keras.layers.Dense(24, activation='relu', name='classifier')
        self.dropout_2 = tf.keras.layers.Dropout(0.0)
        self.dense_output = tf.keras.layers.Dense(1, activation='sigmoid', name='output', dtype='float32')

    def call(self, inputs, training=False, cls_token_index=None, **kwargs):
        # split data into 512 blocks
        x = self.split_data(inputs)
        # x = eight blocks of shape (batch, 512) for each of the three inputs

        # process each of the blocks
        # for each input (batch, 512) we get a (batch, 768) result
        results = []
        tokens_processed = 0
        passed_cls = False
        for block in x:
            res = self.bert(block, training=training,output_attentions=True).attentions[11]
            tokens_processed += 512
            res = tf.squeeze(res)
            res = tf.math.reduce_mean(res, axis=0)
            res = res.numpy().tolist()
            if tokens_processed <= cls_token_index or passed_cls:
                res = res[511]
                results.append(res)
            else:
                res = res[cls_token_index % 512]
                results.append(res)
                passed_cls = True
        r = results[0] + results[1] + results[2]

        return r

    def split_data(self, x) -> list:
        new_ids = tf.split(x[0], [512, 512, 512], axis=-1)
        new_mask = tf.split(x[1], [512, 512, 512], axis=-1)
        new_tokens = tf.split(x[2], [512, 512, 512], axis=-1)

        # return list of tuples of (ids, mask, tokens), each 512 tokens
        out = []
        for i in range(len(new_ids)):
            out.append((new_ids[i], new_mask[i], new_tokens[i]))

        return out


# --------------------------------------------------------------------- NEURAL NETWORK FOR UNPROCESSED ATTENTION

class LongBert2ForAttentionGradients(tf.keras.Model):
    def __init__(self):
        super(LongBert2ForAttentionGradients, self).__init__()
        self.bert = TFAutoModel.from_pretrained("UWB-AIR/Czert-B-base-cased")
        self.bert.config.attention_probs_dropout_prob = 0.1
        self.bert.config.hidden_dropout_prob = 0.1
        self.lstm = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(100, dropout=0.0, recurrent_dropout=0.0))
        self.dropout_1 = tf.keras.layers.Dropout(0.0)
        self.dense_1 = tf.keras.layers.Dense(24, activation='relu', name='classifier')
        self.dropout_2 = tf.keras.layers.Dropout(0.0)
        self.dense_output = tf.keras.layers.Dense(1, activation='sigmoid', name='output', dtype='float32')

    def call(self, inputs, training=False, **kwargs):
        # split data into 512 blocks
        x = self.split_data(inputs)
        # x = eight blocks of shape (batch, 512) for each of the three inputs

        # process each of the blocks
        # for each input (batch, 512) we get a (batch, 768) result
        results = []
        for block in x:
            res = self.bert(block, training=training, output_attentions=True).attentions[11]
            results.append(res)

        return results

    def split_data(self, x) -> list:
        new_ids = tf.split(x[0], [512, 512, 512], axis=-1)
        new_mask = tf.split(x[1], [512, 512, 512], axis=-1)
        new_tokens = tf.split(x[2], [512, 512, 512], axis=-1)

        # return list of tuples of (ids, mask, tokens), each 512 tokens
        out = []
        for i in range(len(new_ids)):
            out.append((new_ids[i], new_mask[i], new_tokens[i]))

        return out


# -------------------------------------------------------------------------------------------- WEBVIEW SENTENCES

class WebViewWindowSentences:
    def __init__(self, data):
        self.was_tag_start = False
        self.ns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        results = data['results']
        tokenized = data['tokenized']
        baseline = data['baseline']

        # words = self.get_word_list(tokenized)
        # word_count = len(words)
        #
        # word_values = [[[] for _ in self.ns] for _ in range(word_count)]
        # percentile = self.get_percentile(results, 0.9)
        #
        # last_c = results[0][1]
        # word_index = 0
        # for b, c, val in results:
        #     if c != last_c:
        #         word_index = 0
        #         last_c = c
        #     if abs(val) > percentile:
        #         continue
        #     if c not in [1]:
        #         continue
        #     for i in range(c):
        #         word_values[word_index + i][self.ns.index(c)].append(val)
        #     word_index += 1
        #
        # vals = self.use_average(word_values)
        # vals = self.scale(vals)

        words = self.get_word_list(tokenized)
        word_count = len(words)
        word_values = [0 for _ in range(word_count)]
        percentile = self.get_percentile(results, 0.9)

        last_c = results[0][1]
        word_index = 0
        for b, c, val in results:
            if c != last_c:
                word_index = 0
                last_c = c
            if abs(val) > percentile:
                continue
            if c not in [1]:
                continue
            for i in range(c):
                word_values[word_index + i] = val
            word_index += 1

        sentences, vals = self.get_sentences_and_values(words, word_values)
        vals = self.scale(vals)

        classification = ""
        if baseline < 0.5:
            classification = "Nekomerční článek"
        elif baseline >= 0.5:
            classification = "Komerční článek"

        html = '<html lang="cs"><head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8"></head><body style="font-size: 20px;font-family=arial">'
        html += '<div style="width:100%;text-align:center;font-size: 25px;padding-bottom:10px">Predikce sítě: {:0.4f}'.format(
            baseline) + ' -> ' + classification + '</div>'

        for i in range(len(vals)):
            html += self.get_word_html(sentences[i], vals[i])

        html += "</body></html>"

        webview.create_window("Vizualizace", html=html)
        webview.start()

    def get_sentences_and_values(self, words, word_values):
        sentences = []
        sentence_values = []
        temp_value = []
        sentence = ""
        for i in range(len(words)):
            word = words[i]
            if word in ["<", "p", "/", ">"]:
                if len(temp_value) != 0:
                    sentences.append(sentence)
                    sentence_values.append(sum(temp_value) / len(temp_value))
                sentences.append(word)
                sentence_values.append(0)
                sentence = ""
                temp_value = []
            elif word in [',', '.']:
                sentences.append(sentence + word)
                temp_value.append(word_values[i])
                sentence_values.append(sum(temp_value) / len(temp_value))
                sentence = ""
                temp_value = []
            else:
                if word in ["“", "„"]:
                    sentence += word
                else:
                    sentence += " " + word
                temp_value.append(word_values[i])

        return sentences, sentence_values

    def use_average(self, w_vals):
        res = [0 for _ in range(len(w_vals))]
        for i in range(len(w_vals)):
            _sum = 0
            _count = 0
            for j in range(len(w_vals[i])):
                for k in range(len(w_vals[i][j])):
                    _sum += w_vals[i][j][k]
                    _count += 1
            try:
                res[i] = _sum / _count
            except ZeroDivisionError:
                res[i] = 0
        return res

    def get_percentile(self, results, percentile):
        vals = []
        for b, c, v in results:
            vals.append(abs(v))

        vals.sort()
        return vals[int(percentile * (len(vals) - 1))]

    def use_max(self, w_vals):
        temp = [[] for _ in range(len(w_vals))]
        for i in range(len(w_vals)):
            for j in range(len(w_vals[i])):
                _max = 0
                real_value = 0
                for k in range(len(w_vals[i][j])):
                    if abs(w_vals[i][j][k]) > _max:
                        _max = abs(w_vals[i][j][k])
                        real_value = w_vals[i][j][k]
                temp[i].append(real_value)

        output = [0 for _ in range(len(w_vals))]
        for i in range(len(temp)):
            _max = 0
            real_val = 0
            for j in range(len(temp[i])):
                if abs(temp[i][j]) > _max:
                    _max = abs(temp[i][j])
                    real_val = temp[i][j]
            output[i] = real_val

        return output

    def get_word_html(self, word, val):
        if word == "<":
            # word = "&lt"
            self.was_tag_start = True
            return "<"
        elif word == ">":
            # word = "&gt"
            self.was_tag_start = False
            return ">"
        if self.was_tag_start:
            return word
        if val < 0:
            html = '<span style="background-color:rgb(' + str(int(255 + val * 255)) + ', ' + str(255) + ', ' + str(
                int(255 + val * 255)) + ');">'
        else:
            html = '<span style="background-color:rgb(' + str(255) + ", " + str(int(255 - val * 255)) + "," + str(
                int(255 - val * 255)) + ');">'
        if word not in ["!", "?", ".", ",", '"']:
            html = " " + html + word
        else:
            html = html + word
        html += "</span>"
        return html

    def scale(self, vals):
        _max = -1
        for val in vals:
            if abs(val) > _max:
                _max = abs(val)

        for i in range(len(vals)):
            vals[i] = vals[i] / _max

        return vals

    def process_data(self, w_vals):
        averages = [0 for _ in w_vals]
        for i in range(len(w_vals)):
            _sum = 0
            _count = 0
            for j in range(len(w_vals[i])):
                _sum += sum(w_vals[i][j])
                _count += len(w_vals[i][j])
            averages[i] = _sum / _count

        valid_values = [[] for _ in w_vals]
        for i in range(len(w_vals)):
            for j in range(len(w_vals[i])):
                for k in range(len(w_vals[i][j])):
                    # if w_vals[i][j][k] > averages[i]:
                    #    valid_values[i].append(w_vals[i][j][k])
                    if (averages[i] >= 0 and w_vals[i][j][k] >= 0) or (averages[i] < 0 and w_vals[i][j][k] < 0):
                        valid_values[i].append(w_vals[i][j][k])
        res = []
        for arr in valid_values:
            res.append(sum(arr) / len(arr))
        return res

    def max_fn(self, w_vals):
        res = []
        for i in range(len(w_vals)):
            res.append(max(w_vals[i]))
        return res

    def avg_fn(self, w_vals):
        res = []
        for i in range(len(w_vals)):
            res.append(sum(w_vals[i]) / len(w_vals[i]))
        return res

    def get_word_list(self, tokens):
        words = []
        word = tokens[0]
        for i in range(1, min(len(tokens), 1534)):
            if "##" in tokens[i]:
                word += tokens[i][2:]
            else:
                words.append(word)
                word = tokens[i]
        words.append(word)
        return words

    def get_text(self, start, count, tokens):
        text = ""
        for i in range(start - 1, start + count - 1):
            text += self.get_word(tokens, i) + " "
        return text

    def get_word(self, tokens, idx):
        word = tokens[idx]
        idx += 1
        if idx == len(tokens):
            return word
        while "##" in tokens[idx]:
            word += tokens[idx][2:]
            idx += 1
        return word


# ------------------------------------------------------------------------------------------------ WEBVIEW WORDS

class WebViewWindowWords:
    def __init__(self, data):
        self.was_tag_start = False
        self.ns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        results = data['results']
        tokenized = data['tokenized']
        baseline = data['baseline']

        words = self.get_word_list(tokenized)
        word_count = len(words)

        word_values = [[[] for _ in self.ns] for _ in range(word_count)]
        percentile = self.get_percentile(results, 0.9)

        last_c = results[0][1]
        word_index = 0
        for b, c, val in results:
            if c != last_c:
                word_index = 0
                last_c = c
            if abs(val) > percentile:
                continue
            if c not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                continue
            for i in range(c):
                word_values[word_index + i][self.ns.index(c)].append(val)
            word_index += 1

        vals = self.use_average(word_values)
        vals = self.scale(vals)

        classification = ""
        if baseline < 0.5:
            classification = "Nekomerční článek"
        elif baseline >= 0.5:
            classification = "Komerční článek"

        html = '<html lang="cs"><head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8"></head><body style="font-size: 20px;font-family=arial">'
        html += '<div style="width:100%;text-align:center;font-size: 25px;padding-bottom:10px">Predikce sítě: {:0.4f}'.format(
            baseline) + ' -> ' + classification + '</div>'

        for i in range(len(vals)):
            html += self.get_word_html(words[i], vals[i])

        html += "</body></html>"

        webview.create_window("Vizualizace", html=html)
        webview.start()

    def use_average(self, w_vals):
        res = [0 for _ in range(len(w_vals))]
        for i in range(len(w_vals)):
            _sum = 0
            _count = 0
            for j in range(len(w_vals[i])):
                for k in range(len(w_vals[i][j])):
                    _sum += w_vals[i][j][k]
                    _count += 1
            try:
                res[i] = _sum / _count
            except ZeroDivisionError:
                res[i] = 0
        return res

    def get_percentile(self, results, percentile):
        vals = []
        for b, c, v in results:
            vals.append(abs(v))

        vals.sort()
        return vals[int(percentile * (len(vals) - 1))]

    def use_max(self, w_vals):
        temp = [[] for _ in range(len(w_vals))]
        for i in range(len(w_vals)):
            for j in range(len(w_vals[i])):
                _max = 0
                real_value = 0
                for k in range(len(w_vals[i][j])):
                    if abs(w_vals[i][j][k]) > _max:
                        _max = abs(w_vals[i][j][k])
                        real_value = w_vals[i][j][k]
                temp[i].append(real_value)

        output = [0 for _ in range(len(w_vals))]
        for i in range(len(temp)):
            _max = 0
            real_val = 0
            for j in range(len(temp[i])):
                if abs(temp[i][j]) > _max:
                    _max = abs(temp[i][j])
                    real_val = temp[i][j]
            output[i] = real_val

        return output

    def get_word_html(self, word, val):
        if word == "<":
            # word = "&lt"
            self.was_tag_start = True
            return "<"
        elif word == ">":
            # word = "&gt"
            self.was_tag_start = False
            return ">"
        if self.was_tag_start:
            return word
        if val < 0:
            html = '<span style="background-color:rgb(' + str(int(255 + val * 255)) + ', ' + str(255) + ', ' + str(
                int(255 + val * 255)) + ');">'
        else:
            html = '<span style="background-color:rgb(' + str(255) + ", " + str(int(255 - val * 255)) + "," + str(
                int(255 - val * 255)) + ');">'
        if word not in ["!", "?", ".", ",", '"']:
            html = " " + html + word
        else:
            html = html + word
        html += "</span>"
        return html

    def scale(self, vals):
        _max = -1
        for val in vals:
            if abs(val) > _max:
                _max = abs(val)

        for i in range(len(vals)):
            vals[i] = vals[i] / _max

        return vals

    def process_data(self, w_vals):
        averages = [0 for _ in w_vals]
        for i in range(len(w_vals)):
            _sum = 0
            _count = 0
            for j in range(len(w_vals[i])):
                _sum += sum(w_vals[i][j])
                _count += len(w_vals[i][j])
            averages[i] = _sum / _count

        valid_values = [[] for _ in w_vals]
        for i in range(len(w_vals)):
            for j in range(len(w_vals[i])):
                for k in range(len(w_vals[i][j])):
                    # if w_vals[i][j][k] > averages[i]:
                    #    valid_values[i].append(w_vals[i][j][k])
                    if (averages[i] >= 0 and w_vals[i][j][k] >= 0) or (averages[i] < 0 and w_vals[i][j][k] < 0):
                        valid_values[i].append(w_vals[i][j][k])
        res = []
        for arr in valid_values:
            res.append(sum(arr) / len(arr))
        return res

    def max_fn(self, w_vals):
        res = []
        for i in range(len(w_vals)):
            res.append(max(w_vals[i]))
        return res

    def avg_fn(self, w_vals):
        res = []
        for i in range(len(w_vals)):
            res.append(sum(w_vals[i]) / len(w_vals[i]))
        return res

    def get_word_list(self, tokens):
        words = []
        word = tokens[0]
        for i in range(1, min(len(tokens), 1534)):
            if "##" in tokens[i]:
                word += tokens[i][2:]
            else:
                words.append(word)
                word = tokens[i]
        words.append(word)
        return words

    def get_text(self, start, count, tokens):
        text = ""
        for i in range(start - 1, start + count - 1):
            text += self.get_word(tokens, i) + " "
        return text

    def get_word(self, tokens, idx):
        word = tokens[idx]
        idx += 1
        if idx == len(tokens):
            return word
        while "##" in tokens[idx]:
            word += tokens[idx][2:]
            idx += 1
        return word


# ------------------------------------------------------------------------- PREPROCESSOR GRADIENTS FOR ATTENTION

class PreprocessorAttentionGradients:
    def __init__(self):
        model, tokenizer = self.get_model(LongBert2ForAttentionGradients)
        self.model = model
        self.tokenizer = tokenizer

    def get_model(self, model_class):
        model = model_class()
        model.load_weights("saved-weights-1")
        return model, AutoTokenizer.from_pretrained("UWB-AIR/Czert-B-base-cased")

    def get_baseline_acc(self, encoded, model):
        res = model(encoded, training=False)
        return float(res)

    def classify_example(self, _input, model, get_attentions=False, get_gradients=False, gradients_target=None):
        el = (
            tf.expand_dims(tf.convert_to_tensor(_input[0]), axis=0),
            tf.expand_dims(tf.convert_to_tensor(_input[1]), axis=0),
            tf.expand_dims(tf.convert_to_tensor(_input[2]), axis=0)
        )
        if get_attentions:
            results = model(el, training=False)
            return results
        elif get_gradients:
            with tf.GradientTape() as tape:
                tape.watch(gradients_target)
                res = model(el, training=False)
            grads = tape.gradient(res, gradients_target)
            return grads
        else:
            res = model(el, training=False)
            return float(res)



    def process_plaintext(self, plaintext: str):
        tokenized = self.tokenizer.tokenize(plaintext)
        sequences = self.preprocess_tokenized(tokenized)
        encoded = self.tokenizer(plaintext, max_length=1536, truncation=True, padding='max_length')
        cls_token_index = encoded.data['input_ids'].index(3)

        # first get the attention values for each block
        attentions = self.classify_example(
            (encoded.data['input_ids'], encoded.data['attention_mask'], encoded.data['token_type_ids']), self.model, get_attentions=True)
        # delete useless model
        del self.model

        # then get the gradients for those attentions
        model = self.get_model(LongBert2)[0]
        grads = []
        for i in range(3):
            grads.append(self.classify_example(
                (encoded.data['input_ids'], encoded.data['attention_mask'], encoded.data['token_type_ids']), model, get_gradients=True, gradients_target=attentions[i]))
        # delete the model
        del self.model

        # get model for prediction
        model = self.get_model(LongBert2)[0]
        baseline_acc = self.classify_example((encoded.data['input_ids'], encoded.data['attention_mask'], encoded.data['token_type_ids']), model)

        print("Baseline: " + str(baseline_acc) + "\n")
        results = []
        for n in [1]:  # [1, 2, 3, 5, 7, 10, 20, 30]:
            results = results + self.process_ngrams(None, sequences, baseline_acc, n)

        return {"tokenized": tokenized, "results": results, "baseline": baseline_acc}

    def preprocess_tokenized(self, tokenized: list):
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

    def process_ngrams(self, array, sequences: list, baseline_acc: float, n: int):
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
                        res += self.sum_range(array, sequence[i], sequence[i + n])
                    except IndexError:
                        pass
                    output.append([sequence[i], n, res])
                    break
                else:
                    res = 0
                    res += self.sum_range(array, sequence[i], sequence[i + n])
                    output.append([sequence[i], n, res])

        return output

    def sum_range(self, values, beg, end):
        res = 0
        for i in range(beg, end):
            res += values[i]
        return res

# --------------------------------------------------------------- PREPROCESSOR INTEGRATED GRADIENTS LIBRARY IMPL.

class PreprocessorIntegratedGradientsLib:
    def __init__(self):
        model, tokenizer = self.get_model(LongBert2ForIGLib)
        self.model = model
        self.tokenizer = tokenizer

    def get_model(self, model_class):
        model = model_class()
        model.load_weights("saved-weights-1")
        return model, AutoTokenizer.from_pretrained("UWB-AIR/Czert-B-base-cased")

    def get_baseline_acc(self, encoded, model):
        res = model(encoded, training=False)
        return float(res)

    def classify_example(self, _input, model):
        el = (
            tf.expand_dims(tf.convert_to_tensor(_input[0]), axis=0),
            tf.expand_dims(tf.convert_to_tensor(_input[1]), axis=0),
            tf.expand_dims(tf.convert_to_tensor(_input[2]), axis=0)
        )
        res = model(el, training=False)
        return res

    def process_plaintext(self, plaintext: str):
        tokenized = self.tokenizer.tokenize(plaintext)
        sequences = self.preprocess_tokenized(tokenized)
        encoded = self.tokenizer(plaintext, max_length=1536, truncation=True, padding='max_length')
        baseline_acc = self.classify_example(
            (encoded.data['input_ids'], encoded.data['attention_mask'], encoded.data['token_type_ids']), self.model)

        ig = IntegratedGradients(self.model, layer=self.model.layers[0].bert.embeddings
                                 , method='gausslegendre', internal_batch_size=1, n_steps=2)
        explanation = ig.explain(np.array([encoded.data['input_ids']]), target=np.array([round(float(baseline_acc))]),
                                 forward_kwargs={"token_type_ids": tf.expand_dims(tf.convert_to_tensor(encoded.data['token_type_ids']), axis=0),
                                                "attention_mask": tf.expand_dims(tf.convert_to_tensor(encoded.data['attention_mask']), axis=0)})

        integrated_gradients = None
        print("Baseline: " + str(float(baseline_acc)) + "\n")
        results = []
        for n in [1]:  # [1, 2, 3, 5, 7, 10, 20, 30]:
            results = results + self.process_ngrams(integrated_gradients, sequences, baseline_acc, n)

        return {"tokenized": tokenized, "results": results, "baseline": float(baseline_acc)}

    def get_baseline(self, ids):
        baseline = [0 for _ in ids]
        baseline[0] = 2
        cls = ids.index(3)
        baseline[cls] = 3
        _random = np.random.randint(low=5, high=30000, size=(cls - 1)).tolist()
        for i in range(len(_random)):
            baseline[i+1] = _random[i]

        with open("baseline.json", "w+", encoding='utf-8') as f:
            f.write(json.dumps(baseline))
        return baseline


    def get_integrated_gradients(self, example, prediction, num_steps=20):
        # create baseline
        baseline = self.get_baseline(example.data['input_ids'])
        interpolated_examples = self.get_interpolated_examples((example.data['input_ids'], example.data['attention_mask'], example.data['token_type_ids']), num_steps, baseline)

        grads = []
        for interpolated in interpolated_examples:
            grads.append(self.get_gradients(interpolated, prediction))
        for i in range(len(grads)):
            grad = tf.convert_to_tensor(grads[i])
            grad = tf.reduce_mean(grad, axis=1)
            grads[i] = grad

        grads = tf.convert_to_tensor(grads, dtype=tf.float32)

        grads = (grads[:-1] + grads[1:]) / 2.0
        avg_grads = tf.reduce_mean(grads, axis=0)

        integrated_grads = (np.array(example.data['input_ids']) - np.array(baseline)) * avg_grads
        return integrated_grads

    def get_interpolated_examples(self, example, num_steps, baseline=None):
        ids = example[0]
        interpolated_examples = []
        if baseline is None:
            baseline = [0 for _ in range(len(ids))]
        for step in range(num_steps + 1):
            interpolated = [0 for _ in range(len(ids))]
            for j in range(len(ids)):
                interpolated[j] = int(baseline[j] + (step / num_steps) * (ids[j] - baseline[j]))
            interpolated_examples.append((interpolated, example[1], example[2]))

        return interpolated_examples

    def get_gradients(self, example, prediction):
        example = (
        tf.expand_dims(tf.convert_to_tensor(example[0]), axis=0),
        tf.expand_dims(tf.convert_to_tensor(example[1]), axis=0),
        tf.expand_dims(tf.convert_to_tensor(example[2]), axis=0)
        )
        with tf.GradientTape() as tape:
            res = self.model(example, training=False)
        grads = tape.gradient(res, self.model.trainable_weights)
        print(float(res))
        return grads[0].values.numpy().tolist()

    def preprocess_tokenized(self, tokenized: list):
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

    def process_ngrams(self, array, sequences: list, baseline_acc: float, n: int):
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
                        res += self.sum_range(array, sequence[i], sequence[i + n])
                    except IndexError:
                        pass
                    output.append([sequence[i], n, res])
                    break
                else:
                    res = 0
                    res += self.sum_range(array, sequence[i], sequence[i + n])
                    output.append([sequence[i], n, res])

        return output

    def sum_range(self, values, beg, end):
        res = 0
        for i in range(beg, end):
            res += values[i]
        return res


# ----------------------------------------------------------------------------- PREPROCESSOR INTEGRATED GRADIENTS

class PreprocessorIntegratedGradients:
    def __init__(self):
        model, tokenizer = self.get_model(LongBert2)
        self.model = model
        self.tokenizer = tokenizer
        self.embeddings

    def get_model(self, model_class):
        model = model_class()
        model.load_weights("saved-weights-1")
        return model, AutoTokenizer.from_pretrained("UWB-AIR/Czert-B-base-cased")

    def get_baseline_acc(self, encoded, model):
        res = model(encoded, training=False)
        return float(res)

    def classify_example(self, _input, model):
        el = (
            tf.expand_dims(tf.convert_to_tensor(_input[0]), axis=0),
            tf.expand_dims(tf.convert_to_tensor(_input[1]), axis=0),
            tf.expand_dims(tf.convert_to_tensor(_input[2]), axis=0)
        )
        res = model(el, training=False)
        return res

    def process_plaintext(self, plaintext: str):
        tokenized = self.tokenizer.tokenize(plaintext)
        vocab = self.tokenizer.vocab
        # 29 999
        sequences = self.preprocess_tokenized(tokenized)
        encoded = self.tokenizer(plaintext, max_length=1536, truncation=True, padding='max_length')
        baseline_acc = self.classify_example(
            (encoded.data['input_ids'], encoded.data['attention_mask'], encoded.data['token_type_ids']), self.model)

        integrated_gradients = self.get_integrated_gradients(encoded, float(baseline_acc)).numpy().tolist()

        print("Baseline: " + str(float(baseline_acc)) + "\n")
        results = []
        for n in [1]:  # [1, 2, 3, 5, 7, 10, 20, 30]:
            results = results + self.process_ngrams(integrated_gradients, sequences, baseline_acc, n)

        return {"tokenized": tokenized, "results": results, "baseline": float(baseline_acc)}

    def get_baseline(self, ids):
        baseline = [0 for _ in ids]
        baseline[0] = 2
        cls = ids.index(3)
        baseline[cls] = 3
        _random = np.random.randint(low=5, high=30000, size=(cls - 1)).tolist()
        for i in range(len(_random)):
            baseline[i+1] = _random[i]

        with open("baseline.json", "w+", encoding='utf-8') as f:
            f.write(json.dumps(baseline))
        return baseline

    def get_integrated_gradients(self, example, prediction, num_steps=20):
        # create baseline
        baseline = self.get_baseline(example.data['input_ids'])

        # create interpolated examples - if baseline is None baseline is all 0
        interpolated_examples = self.get_interpolated_examples((example.data['input_ids'], example.data['attention_mask'], example.data['token_type_ids']), num_steps, baseline)

        grads = []
        # for each interpolation get embedding gradients
        for interpolated in interpolated_examples:
            grads.append(self.get_gradients(interpolated, prediction)) # a gradient has shape (1536, 768)
        for i in range(len(grads)):
            grad = tf.convert_to_tensor(grads[i])  # to tensor
            grad = tf.reduce_mean(grad, axis=1)    # average over axis 1 to shape (1536,)
            grads[i] = grad

        grads = tf.convert_to_tensor(grads, dtype=tf.float32) # create tensor of size (num_steps + 1, 1536)

        grads = (grads[:-1] + grads[1:]) / 2.0      # approximate with trapezoidal rule, shape = (num_steps, 1536)
        avg_grads = tf.reduce_mean(grads, axis=0)   # average over samples, shape = (1536)

        integrated_grads = (np.array(example.data['input_ids']) - np.array(baseline)) * avg_grads   # calculate integral, output shape = (1536)
        return integrated_grads

    def get_interpolated_examples(self, example, num_steps, baseline=None):
        ids = example[0]
        interpolated_examples = []
        if baseline is None:
            baseline = [0 for _ in range(len(ids))]
        for step in range(num_steps + 1):
            interpolated = [0 for _ in range(len(ids))]
            for j in range(len(ids)):
                interpolated[j] = int(baseline[j] + (step / num_steps) * (ids[j] - baseline[j]))
            interpolated_examples.append((interpolated, example[1], example[2]))

        return interpolated_examples

    def get_gradients(self, example, prediction):
        example = (
        tf.expand_dims(tf.convert_to_tensor(example[0]), axis=0),
        tf.expand_dims(tf.convert_to_tensor(example[1]), axis=0),
        tf.expand_dims(tf.convert_to_tensor(example[2]), axis=0)
        )
        with tf.GradientTape() as tape:
            res = self.model(example, training=False)
        grads = tape.gradient(res, self.model.trainable_weights)
        print(float(res))
        return grads[0].values.numpy().tolist()

    def preprocess_tokenized(self, tokenized: list):
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

    def process_ngrams(self, array, sequences: list, baseline_acc: float, n: int):
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
                        res += self.sum_range(array, sequence[i], sequence[i + n])
                    except IndexError:
                        pass
                    output.append([sequence[i], n, res])
                    break
                else:
                    res = 0
                    res += self.sum_range(array, sequence[i], sequence[i + n])
                    output.append([sequence[i], n, res])

        return output

    def sum_range(self, values, beg, end):
        res = 0
        for i in range(beg, end):
            res += values[i]
        return res


# ----------------------------------------------------------------------------------------- PREPROCESSOR ATTENTION

class PreprocessorAttention:
    def __init__(self):
        model, tokenizer = self.get_model(LongBert2ForAttention)
        self.model = model
        self.tokenizer = tokenizer

    def get_model(self, model_class):
        model = model_class()
        model.load_weights("saved-weights-1")
        return model, AutoTokenizer.from_pretrained("UWB-AIR/Czert-B-base-cased")

    def get_baseline_acc(self, encoded, model):
        res = model(encoded, training=False)
        return float(res)

    def classify_example(self, _input, model, cls_token_index=None):
        el = (
            tf.expand_dims(tf.convert_to_tensor(_input[0]), axis=0),
            tf.expand_dims(tf.convert_to_tensor(_input[1]), axis=0),
            tf.expand_dims(tf.convert_to_tensor(_input[2]), axis=0)
        )
        with tf.GradientTape() as tape:
            if cls_token_index is not None:
                res = model(el, training=False, cls_token_index=cls_token_index)
            else:
                res = model(el, training=False)
        if isinstance(model, LongBert2ForAttention):
            return res
        else:
            return float(res), tape.gradient(res, model.trainable_weights)

    def process_plaintext(self, plaintext: str):
        tokenized = self.tokenizer.tokenize(plaintext)
        sequences = self.preprocess_tokenized(tokenized)
        encoded = self.tokenizer(plaintext, max_length=1536, truncation=True, padding='max_length')
        cls_token_index = encoded.data['input_ids'].index(3)
        res = self.classify_example(
            (encoded.data['input_ids'], encoded.data['attention_mask'], encoded.data['token_type_ids']), self.model, cls_token_index)
        for el in res:
            if el < 0:
                print(str(el))
        del self.model
        model = self.get_model(LongBert2)[0]
        baseline_acc, grads = self.classify_example(
            (encoded.data['input_ids'], encoded.data['attention_mask'], encoded.data['token_type_ids']), model)

        embeddings = grads[0].values
        embeddings = tf.reduce_sum(embeddings, axis=1)
        res = tf.multiply(embeddings, tf.convert_to_tensor(res)).numpy().tolist()

        print("Baseline: " + str(baseline_acc) + "\n")
        results = []
        for n in [1]:  # [1, 2, 3, 5, 7, 10, 20, 30]:
            results = results + self.process_ngrams(res, sequences, baseline_acc, n)

        return {"tokenized": tokenized, "results": results, "baseline": baseline_acc}

    def preprocess_tokenized(self, tokenized: list):
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

    def process_ngrams(self, array, sequences: list, baseline_acc: float, n: int):
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
                        res += self.sum_range(array, sequence[i], sequence[i + n])
                    except IndexError:
                        pass
                    output.append([sequence[i], n, res])
                    break
                else:
                    res = 0
                    res += self.sum_range(array, sequence[i], sequence[i + n])
                    output.append([sequence[i], n, res])

        return output

    def sum_range(self, values, beg, end):
        res = 0
        for i in range(beg, end):
            res += values[i]
        return res


# ----------------------------------------------------------------------------------------- PREPROCESSOR GRADIENT

class PreprocessorGradient:
    def __init__(self):
        model, tokenizer = self.get_model()
        self.model = model
        self.tokenizer = tokenizer

    def get_model(self):
        model = LongBert2()
        model.load_weights("saved-weights-1")
        return model, AutoTokenizer.from_pretrained("UWB-AIR/Czert-B-base-cased")

    def get_baseline_acc(self, encoded, model):
        res = model(encoded, training=False)
        return float(res)

    def classify_example(self, _input, model):
        el = (
            tf.expand_dims(tf.convert_to_tensor(_input[0]), axis=0),
            tf.expand_dims(tf.convert_to_tensor(_input[1]), axis=0),
            tf.expand_dims(tf.convert_to_tensor(_input[2]), axis=0)
        )
        with tf.GradientTape() as tape:
            res = model(el, training=False)
        return float(res), tape.gradient(res, model.trainable_weights)

    def process_plaintext(self, plaintext: str):
        tokenized = self.tokenizer.tokenize(plaintext)
        sequences = self.preprocess_tokenized(tokenized)
        encoded = self.tokenizer(plaintext, max_length=1536, truncation=True, padding='max_length')
        baseline_acc, grads = self.classify_example(
            (encoded.data['input_ids'], encoded.data['attention_mask'], encoded.data['token_type_ids']), self.model)

        embeddings = grads[0].values.numpy().tolist()
        tokens = grads[1].values.numpy().tolist()
        attention = grads[2].values.numpy().tolist()

        print("Baseline: " + str(baseline_acc) + "\n")
        results = []
        for n in [1, 2, 3]:  # [1, 2, 3, 5, 7, 10, 20, 30]:
            results = results + self.process_ngrams((embeddings, tokens, attention), sequences, baseline_acc, n)

        # print("Processing results\n")
        # print(len(results))
        # with open("results-grads-pos-short-tokens.json", "w+", encoding='utf-8') as f:
        #    f.write(json.dumps(results))
        # with open("tokenized-grads-pos-short-tokens.json", "w+", encoding='utf-8') as f:
        #    f.write(json.dumps(tokenized))
        return {"tokenized": tokenized, "results": results, "baseline": baseline_acc}

    def preprocess_tokenized(self, tokenized: list):
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

    def process_ngrams(self, encoded, sequences: list, baseline_acc: float, n: int):
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
                        res += self.sum_range(encoded[0], sequence[i], sequence[i + n])
                        res += self.sum_range(encoded[1], sequence[i], sequence[i + n])
                        res += self.sum_range(encoded[2], sequence[i], sequence[i + n])
                    except IndexError:
                        pass
                    output.append([sequence[i], n, res])
                    break
                else:
                    res = 0
                    res += self.sum_range(encoded[0], sequence[i], sequence[i + n])
                    res += self.sum_range(encoded[1], sequence[i], sequence[i + n])
                    res += self.sum_range(encoded[2], sequence[i], sequence[i + n])
                    output.append([sequence[i], n, res])

        return output

    def sum_range(self, values, beg, end):
        res = 0
        for i in range(beg, end):
            for j in range(len(values[i])):
                if values[i][j] is None:
                    pass
                res += values[i][j]
        return res


# ----------------------------------------------------------------------------------------- PREPROCESSOR COVERING

class PreprocessorCovering:
    def __init__(self):
        model, tokenizer = self.get_model()
        self.model = model
        self.tokenizer = tokenizer

    def get_model(self):
        model = LongBert2()
        model.load_weights("saved-weights-1")
        # return LongBert2().load_weights(filepath=model_path), AutoTokenizer.from_pretrained("UWB-AIR/Czert-B-base-cased")
        return model, AutoTokenizer.from_pretrained("UWB-AIR/Czert-B-base-cased")

    def get_baseline_acc(self, encoded, model):
        res = model(encoded, training=False)
        return float(res)

    def classify_example(self, _input, model):
        el = (
            tf.expand_dims(tf.convert_to_tensor(_input[0]), axis=0),
            tf.expand_dims(tf.convert_to_tensor(_input[1]), axis=0),
            tf.expand_dims(tf.convert_to_tensor(_input[2]), axis=0)
        )
        res = model(el, training=False)
        return float(res)

    def process_plaintext(self, plaintext: str, tokenizer: BertTokenizer, model: TFBertModel):
        tokenized = tokenizer.tokenize(plaintext)
        sequences = self.preprocess_tokenized(tokenized)
        encoded = tokenizer(plaintext, max_length=1536, truncation=True, padding='max_length')
        baseline_acc = self.classify_example(
            (encoded.data['input_ids'], encoded.data['attention_mask'], encoded.data['token_type_ids']), model)

        print("Baseline: " + str(baseline_acc) + "\n")
        results = []
        for n in [180, 200]:
            results = results + self.process_ngrams(encoded, sequences, model, baseline_acc, n)

        print("Processing results\n")
        print(len(results))
        with open("results-test.json", "w+", encoding='utf-8') as f:
            f.write(json.dumps(results))
        with open("tokenized-test.json", "w+", encoding='utf-8') as f:
            f.write(json.dumps(tokenized))
        return

    def preprocess_tokenized(self, tokenized: list):
        word_indices = []
        sequences = []
        seq_end_str = ["!", "?", "."]

        words = []
        word = ""
        for i in range(min(len(tokenized), 1534)):
            if "##" in tokenized[i]:
                word += tokenized[i][2:]
            else:
                word_indices.append(i + 1)
                words.append(word)
                word = tokenized[i]
                # if tokenized[i] in seq_end_str:
                #    sequences.append(word_indices)
                #    word_indices = []

        sequences.append(word_indices)
        return sequences

    def process_ngrams(self, encoded, sequences: list, model: TFBertModel, baseline_acc: float, n: int):
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
                if i + n > seq_length:
                    break
                if i + n == seq_length:
                    modified = [encoded.data['input_ids'].copy(), encoded.data['attention_mask'].copy(),
                                encoded.data['token_type_ids'].copy()]
                    modified[0] = modified[0][0:sequence[i]] + modified[0][sequence[i + n - 1] + 1:] + [0 for _ in
                                                                                                        range(sequence[
                                                                                                                  seq_length - 1] -
                                                                                                              sequence[
                                                                                                                  i] + 1)]
                    modified[1] = modified[1][0:sequence[i]] + modified[0][sequence[i + n - 1] + 1:] + [0 for _ in
                                                                                                        range(sequence[
                                                                                                                  seq_length - 1] -
                                                                                                              sequence[
                                                                                                                  i] + 1)]
                    res = self.classify_example(modified, model)
                    output.append([sequence[i], n, res - baseline_acc])
                    break
                else:
                    modified = [encoded.data['input_ids'].copy(), encoded.data['attention_mask'].copy(),
                                encoded.data['token_type_ids'].copy()]
                    modified[0] = modified[0][0:sequence[i]] + modified[0][sequence[i + n]:] + [0 for _ in range(
                        sequence[i + n] - sequence[i])]
                    modified[1] = modified[1][0:sequence[i]] + modified[1][sequence[i + n]:] + [0 for _ in range(
                        sequence[i + n] - sequence[i])]
                    res = self.classify_example(modified, model)
                    output.append([sequence[i], n, res - baseline_acc])

        return output


# ------------------------------------------------------------------------------------------- APPLICATION CONTROL

def control_batch(text):
    preprocessor = PreprocessorGradient()
    data = preprocessor.process_plaintext(text)
    WebViewWindowSentences(data)


def control_interactive():
    layout = [
        [
            [PySimpleGUI.Text("Zadejte text pro klasifikaci")],
            [PySimpleGUI.Multiline(size=(60, 20), key='textbox')],
            [PySimpleGUI.Button("Zpracovat"), PySimpleGUI.Text("Zpracování vstupu chvíli zabere", key="statustext"),
             PySimpleGUI.Button("Smazat")]
        ]
    ]
    window = PySimpleGUI.Window(title="Demo", layout=layout)
    preprocessor = PreprocessorAttentionGradients()
    while True:
        event, values = window.read()
        if event is None:
            exit(0)
        elif "Zpracovat" in event:
            text = values['textbox']
            data = preprocessor.process_plaintext(insert_paragraphs(text))
            WebViewWindowWords(data)
        elif "Smazat" in event:
            window['textbox'].update("")


def insert_paragraphs(text):
    split = text.split("\n")
    text = "<p>"
    for i in range(len(split) - 1):
        if split[i] == "":
            continue
        text += split[i] + "</p><p>"
    text += split[len(split) - 1] + "</p>"
    return text


def run(file):
    if file is not None:
        try:
            text = open(file, "r", encoding='utf-8').read()
        except OSError:
            print("Soubor nelze otevřít")
            exit(1)
        control_batch(text)
    else:
        control_interactive()


if __name__ == "__main__":
    if len(argv) != 2:
        print("Nezadán vstupní soubor, program poběží v interaktivním režimu")
        run(None)
    run(argv[1])
