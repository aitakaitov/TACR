import os

plain_dir = "filtered_datasets/filtered_relevant_plaintext"
html_dir = "filtered_datasets/filtered_relevant_html"

domains_html = os.listdir(html_dir)
domains_plain = os.listdir(plain_dir)


terms_files_html = os.listdir(html_dir + "/terms")
cookies_files_html = os.listdir(html_dir + "/cookies")

terms_files_plain = os.listdir(plain_dir + "/terms")
cookies_files_plain = os.listdir(plain_dir + "/cookies")

for file in cookies_files_html:
    if file not in cookies_files_plain:
        print("MISSING")

for file in terms_files_html:
    if file not in terms_files_plain:
        print("MISSING")


