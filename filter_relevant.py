from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from neural_net.preprocessing import Preprocessing

import os
from shutil import copyfile

# define source directories
dataset_dir_plaintext = "datasets_to_filter/pages_plaintext"
dataset_dir_html = "datasets_to_filter/pages_html"

# list all domains (plaintext)
relevant_dirs = os.listdir(dataset_dir_plaintext)
terms_paths = []
cookies_paths = []

# collect all paths
print("Collecting paths...")
for _dir in relevant_dirs:
    dirs = os.listdir(dataset_dir_plaintext + "/" + _dir)
    if "terms" in dirs:
        for t in os.listdir(dataset_dir_plaintext + "/" + _dir + "/terms"):
            terms_paths.append(dataset_dir_plaintext + "/" + _dir + "/terms/" + t)
    if "cookies" in dirs:
        for c in os.listdir(dataset_dir_plaintext + "/" + _dir + "/cookies"):
            cookies_paths.append(dataset_dir_plaintext + "/" + _dir + "/cookies/" + c)

# define destination directories
filtered_dir_plaintext = "filtered_datasets/filtered_relevant_plaintext"
filtered_dir_html = "filtered_datasets/filtered_relevant_html"

# create directories
print("Creating directories...")
try:
    os.mkdir("filtered_datasets")
    os.mkdir(filtered_dir_plaintext)
    os.mkdir(filtered_dir_plaintext + "/cookies")
    os.mkdir(filtered_dir_plaintext + "/terms")
    os.mkdir(filtered_dir_html)
    os.mkdir(filtered_dir_html + "/cookies")
    os.mkdir(filtered_dir_html + "/terms")
except OSError:
    pass

# load model and tokenizer
print("Loading model...")
model = load_model("final_models/model-best")
tokenizer = Preprocessing().load("model_configurations/tokenizer_configurations/preprocessing-config-no-keyw").tokenizer

# process terms pages
print("Processing terms pages...")
for i in range(len(terms_paths)):
    terms_path = terms_paths[i]
    with open(terms_path, "r", encoding='utf-8') as f:
        tokenized = tokenizer.texts_to_sequences([f.read()])[0]
        tokenized = pad_sequences([tokenized], maxlen=10000, truncating="post", padding="post")

        # predict class
        res = model.predict(tokenized, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,
                            workers=1, use_multiprocessing=False)[0]

        # ignore irrelevant pages
        # NOTE:  the structures and file names of HTML and plaintext directories are the exact same - only the files differ in contents
        #        that way it is possible to just replace the plaintext dir in path with the html dir with no side effects
        maximum = max(res[0], res[1], res[2])
        if maximum == res[1]:
            copyfile(terms_path, filtered_dir_plaintext + "/cookies/" + terms_path.split("/")[len(terms_path.split("/")) - 1])

            # here we replace plaintext dir in terms_path with html dir - since the structure is the same, we can do this
            copyfile(terms_path.replace(dataset_dir_plaintext + "/", dataset_dir_html + "/"), filtered_dir_html + "/cookies/" + terms_path.split("/")[len(terms_path.split("/")) - 1])

        elif maximum == res[2]:
            copyfile(terms_path, filtered_dir_plaintext + "/terms/" + terms_path.split("/")[len(terms_path.split("/")) - 1])
            copyfile(terms_path.replace(dataset_dir_plaintext + "/", dataset_dir_html + "/"), filtered_dir_html + "/terms/" + terms_path.split("/")[len(terms_path.split("/")) - 1])

# process cookies pages
print("Processing cookies pages...")
for i in range(len(cookies_paths)):
    cookies_path = cookies_paths[i]
    with open(cookies_path, "r", encoding='utf-8') as f:
        tokenized = tokenizer.texts_to_sequences([f.read()])[0]
        tokenized = pad_sequences([tokenized], maxlen=10000, truncating="post", padding="post")

        res = model.predict(tokenized, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,
                            workers=1, use_multiprocessing=False)[0]

        # ignore irrelevant pages
        maximum = max(res[0], res[1], res[2])
        if maximum == res[1]:
            copyfile(cookies_path, filtered_dir_plaintext + "/cookies/" + cookies_path.split("/")[len(cookies_path.split("/")) - 1])
            copyfile(cookies_path.replace(dataset_dir_plaintext + "/", dataset_dir_html + "/"), filtered_dir_html + "/cookies/" + cookies_path.split("/")[len(cookies_path.split("/")) - 1])
        elif maximum == res[2]:
            copyfile(cookies_path, filtered_dir_plaintext + "/terms/" + cookies_path.split("/")[len(cookies_path.split("/")) - 1])
            copyfile(cookies_path.replace(dataset_dir_plaintext + "/", dataset_dir_html + "/"), filtered_dir_html + "/terms/" + cookies_path.split("/")[len(cookies_path.split("/")) - 1])


