import neural_net.preprocessing

"""
A script to create a new test/train split and a new tokenizer 
"""

# relevant pages from crawler, expects the same structure the crawler has generated
pages_dir = "raw_datasets/pages"

# irrelevant pages from crawler, expects a directory with page files in it
irrelevant_dir = "raw_datasets/irrelevant"

# directory the test dataset will be in
test_dir = "test_created"

# directory the train dataset will be in
train_dir = "train_created"

# path to text fasttext file for czech language
fasttext_file = "fasttext/cc.cs.300.vec"

# filter out keywords
filter_keywords = False

# created tokenizer file
tokenizer_file = "preprocessing-config"

# create instance of preprocessor
prep_conf = neural_net.preprocessing.Preprocessing()

# preprocess the data, create test/train split, fit the tokenizer on the vocabulary
prep_conf.create_dataset(pages_dir, irrelevant_dir, train_dir, test_dir, fasttext_file, filter_keywords)

# save the created tokenizer, embedding matrix and maximum length of a document
prep_conf.save(tokenizer_file)
