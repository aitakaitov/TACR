import os
import pickle
import random
import re
from typing import Union

import fasttext
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer


class PreprocessingConfig:
    """
    Contains stuff necessary for the model to work
    """
    def __init__(self, emb_matrix, tokenizer, doc_max_len):
        self.emb_matrix = emb_matrix
        self.tokenizer = tokenizer
        self.doc_max_len = doc_max_len


# noinspection PyMethodMayBeStatic
class Preprocessing:
    """
    Performs data preprocessing, defining a vocabulary, Tokenizer instance, and creating an embedding
    matrix
    """
    def __init__(self):
        self.tokenizer = Tokenizer(
            num_words=None,
            filters="!#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\"\'",
            lower=True,
            split=' ',
            char_level=False
        )
        self.vocabulary = None
        self.Tokenizer = None
        self.embedding_matrix = None
        self.doc_max_len = 10000
        self.urls_cookies = []
        self.urls_terms = []
        self.urls_irrelevant = []

    def create_dataset(self, relevant_pages_dir: str, irrelevant_pages_dir: str, train_dir: str, test_dir: str, fasttext_file: str, filter_keywords):
        """
        Preprocesses the pages and creates a dataset, then saves vocabulary in a file
        :param fasttext_file: File with fasttext embeddings
        :param relevant_pages_dir: Directory with relevant pages. Inside, a folder for each page is expected. In that folder,
         terms and cookies folders with pages are expected. Folder with no pages in it is ignored
        :param irrelevant_pages_dir: Directory with irrelevant pages. Inside it, files with page contents are expected.
        :param train_dir: Dir to write the training dataset into, contains a file for each page - the format is class text. 0 is irrelevant,
        1 is cookies and 2 is terms
        :param test_dir: Dir to write the testing dataset into, contains a file for each page - the format is class text, one page per line. 0 is irrelevant,
        1 is cookies and 2 is terms
        :return: None
        """

        # check if the directories exist
        os.makedirs(relevant_pages_dir, exist_ok=True)
        os.makedirs(irrelevant_pages_dir, exist_ok=True)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # contains (path, class, train/test)
        pages_info = []

        # collect all page paths
        print("Collecting all page paths")
        tcpaths = self.__collect_rel_pages_paths(relevant_pages_dir)
        terms_pages_paths = tcpaths[0]
        cookies_pages_paths = tcpaths[1]
        irrelevant_pages_paths = self.__collect_irr_pages_paths(irrelevant_pages_dir)

        print("{}\n{}\n{}".format(len(irrelevant_pages_paths), len(terms_pages_paths), len(cookies_pages_paths)))

        # split into train and test
        print("Splitting pages into train and test datasets")
        pgs = self.__split_pages(terms_pages_paths)

        # create info
        pinf = self.__create_page_info(pgs[0], 2, True)     # create info for train terms
        pinf2 = self.__create_page_info(pgs[1], 2, False)   # create info for test terms
        # add info to pages_info
        [pages_info.append(info) for info in pinf]
        [pages_info.append(info) for info in pinf2]

        # do that for other classes
        pgs = self.__split_pages(cookies_pages_paths)
        pinf = self.__create_page_info(pgs[0], 1, True)
        pinf2 = self.__create_page_info(pgs[1], 1, False)
        [pages_info.append(info) for info in pinf]
        [pages_info.append(info) for info in pinf2]

        pgs = self.__split_pages(irrelevant_pages_paths)
        pinf = self.__create_page_info(pgs[0], 0, True)
        pinf2 = self.__create_page_info(pgs[1], 0, False)
        [pages_info.append(info) for info in pinf]
        [pages_info.append(info) for info in pinf2]

        # create vocab, preprocess pages
        print("Creating vocabulary, writing pages into test and train directories")
        self.vocabulary = dict()
        for i in range(len(pages_info)):
            if pages_info[i][2]:
                target_dir = train_dir
            else:
                target_dir = test_dir
            page_vc = self.__preprocess_page(pages_info[i], target_dir, i)

            # if the page was a duplicate
            if page_vc is None:
                continue

            for key, value in page_vc.items():
                try:
                    self.vocabulary[key] += value
                except KeyError:
                    self.vocabulary[key] = value

        oov_dict, self.vocabulary = self.__process_vocabulary(self.vocabulary, fasttext_file, filter_keywords)

        text = ""
        for word in self.vocabulary:
            text += word + " "
        self.tokenizer.fit_on_texts([text], )

        self.__create_embedding_matrix(fasttext_file, oov_dict)

        print("{}\n{}\n{}".format(len(self.urls_irrelevant), len(self.urls_terms), len(self.urls_cookies)))

        return

    def __process_vocabulary(self, vocabulary: dict, fasttext_file, filter_keywords) -> tuple:
        """
        Processes vocabulary
        :param vocabulary: vocabulary
        :return: list of words
        """
        min_count = 15
        remove_numbers = True
        remove_stopwords = True

        # remove stopwords
        stopwords = []
        if remove_stopwords:
            if not filter_keywords:
                sw_file = open("neural_net/stopwords-cs", "r", encoding='utf-8')       # use stopwords without keywords
            else:
                sw_file = open("neural_net/stopwords-cs-keyw", "r", encoding='utf-8')     # use stopwords with keywords

            stopwords = sw_file.readlines()
            for i in range(len(stopwords)):
                stopwords[i] = stopwords[i][0:len(stopwords[i]) - 1]

        # load words we have embeddings for
        ft_words = []
        ft_file = open(fasttext_file, "r", encoding='utf-8')
        ft_file.readline()
        while True:
            line = ft_file.readline()
            if line == "":
                break
            ft_words.append(line.split()[0])

        ft_words = set(ft_words)

        new_vocab = []
        oov_words = []

        for key, value in vocabulary.items():
            add = True
            if value < min_count:
                add = False
            if remove_numbers and key.isdigit():
                add = False
            if remove_stopwords:
                if key in stopwords:
                    add = False
            if key not in ft_words and add:
                oov_words.append(key)

            if add:
                new_vocab.append(key)

        oov_dict = self.create_oov_vectors(oov_words, fasttext_file)

        return oov_dict, new_vocab

    def create_oov_vectors(self, oov_words, fasttext_path):
        """
        Uses fasttext to create embedding vectors for OOV words
        :param oov_words: list of oov words
        :return: None
        """

        print("Loading fasttext model and creating vectors for OOV words")
        split_path = fasttext_path.split("/")
        ft = fasttext.load_model(fasttext_path[0:len(fasttext_path) - len(split_path[-1])] + "cc.cs.300.bin")
        word_vec_dict = dict()

        for oov in oov_words:
            word_vec_dict[oov] = ft.get_word_vector(oov)

        return word_vec_dict

    def __create_page_info(self, pages: list, clss: int, is_train: bool) -> list:
        """
        creates info tuples for each page in pages
        :param pages: pages
        :param clss: 0,1,2
        :param is_train:
        :return: tuple (page path, class, is_train)
        """
        infos = []
        for page in pages:
            infos.append((page, clss, is_train))

        return infos

    def __preprocess_page(self, page_info: tuple, target_dir: str, index: int) -> Union[dict, None]:
        """
        Performs preprocessing and saves the result into the target directory. Returns a dictionary with words and counts
        :param page_info: tuple of (source path, class, train/test)
        :param target_dir: target directory
        :param index: index of the page
        :return: dict vocabulary or None if the page is a duplicate
        """
        with open(page_info[0], "r", encoding='utf-8') as src_file:
            page_lines = src_file.readlines()

        try:
            url = page_lines[0][5:len(page_lines[0]) - 1]             # extract the URL and add a / if necessary
            if url[-1] != "/":
                url += "/"
        except IndexError:
            return None

        # check for duplicates
        if page_info[1] == 0:
            if url in self.urls_irrelevant:
                return None
            else:
                self.urls_irrelevant.append(url)
        elif page_info[1] == 1:
            if url in self.urls_cookies:
                return None
            else:
                self.urls_cookies.append(url)
        elif page_info[1] == 2:
            if url in self.urls_terms:
                return None
            else:
                self.urls_terms.append(url)

        # remove URL and DEPTH
        page_text = ""
        for line in page_lines[2:]:
            page_text += line

        # remove all non-alpha numeric and replace them with space
        page_text = re.sub(r"[\W_]+", ' ', page_text)
        page_tgt_file = open(target_dir + "/" + str(index), "w+", encoding='utf-8')
        page_tgt_file.write(str(page_info[1]) + " " + page_text)
        page_tgt_file.close()

        # create vocabulary
        split_text = page_text.split()
        vc = dict()
        for word in split_text:
            try:
                vc[word.lower()] += 1
            except KeyError:
                vc[word.lower()] = 1

        return vc

    def __collect_rel_pages_paths(self, relevant_dir: str) -> tuple:
        """
        Collects all relevant page paths
        :param relevant_dir: dir
        :return: tuple of (terms paths, cookies paths)
        """
        pages_dirs = os.listdir(os.path.realpath(relevant_dir))
        pages_dirs.sort()
        cookies_paths = []
        terms_paths = []

        for pdir in pages_dirs:
            cookies_files = os.listdir(relevant_dir + "/" + pdir + "/cookies")
            cookies_files.sort()
            terms_files = os.listdir(relevant_dir + "/" + pdir + "/terms")
            terms_files.sort()

            if len(cookies_files) != 0:
                [cookies_paths.append(relevant_dir + "/" + pdir + "/cookies/" + file) for file in cookies_files]

            if len(terms_files) != 0:
                [terms_paths.append(relevant_dir + "/" + pdir + "/terms/" + file) for file in terms_files]

        return terms_paths, cookies_paths

    def __collect_irr_pages_paths(self, irrelevant_dir: str) -> list:
        """
        Collects irrelevant page paths
        :param irrelevant_dir: dir
        :return: list of irrelevant page paths
        """
        files = os.listdir(irrelevant_dir)
        files.sort()
        paths = []

        for i in range(len(files)):
            paths.append(irrelevant_dir + "/" + files[i])

        return paths

    def __split_pages(self, pages: list) -> tuple:
        """
        Splits pages list into training pages and testing pages
        :param pages: list of pages
        :return: tuple of training pages, testing pages
        """
        test_size = 0.3
        test_count = int(len(pages) * test_size)

        random.shuffle(pages)
        random.shuffle(pages)

        test_pages = pages[:test_count]
        train_pages = pages[test_count:]

        return train_pages, test_pages

    def __create_embedding_matrix(self, fasttext_path: str, oov_dict: dict):
        """
        Saves embedding vectors based on Tokenizer indexing
        :param fasttext_path: Path to fasttext pretrained embeddings
        :return: None
        """

        # initialize the embedding matrix
        ft_file = open(fasttext_path, "r+", encoding='utf-8')
        word_index = self.tokenizer.word_index
        self.embedding_matrix = np.zeros((len(word_index) + 1, 300))

        print("Parsing OOV words embeddings")
        # handle OOV tokens
        for word, vector in oov_dict.items():
            try:
                index = word_index[word]
            except KeyError:
                continue

            for i in range(300):
                self.embedding_matrix[index][i] = vector[i]

        print("Parsing fasttext embeddings")
        ft_file.readline()  # read dimensions
        while True:
            line = ft_file.readline()
            if line == "":
                break
            word = line.split()[0]
            try:
                if word_index[word] is not None:
                    vector = []
                    for e in line.split()[1:]:
                        vector.append(float(e))
                    self.embedding_matrix[word_index[word]] = vector
            except KeyError:
                continue

        ft_file.close()

    def load(self, config_path):
        """
        Loads the relevant objects into PreprocessingConfig
        :param config_path: File to load the config from
        :return:
        """
        with open(config_path, "rb") as f:
            return pickle.load(f)

    def save(self, config_path):
        """
        Dumps the relevant objects into a pickle
        :param config_path: File to save the config to
        :return:
        """
        config = PreprocessingConfig(self.embedding_matrix, self.tokenizer, self.doc_max_len)
        pickle.dump(config, open(config_path, "wb"))

