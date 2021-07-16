import os
from bs4 import BeautifulSoup
import bs4


src_terms_dir = "filtered_datasets/filtered_relevant_html/terms"
src_cookies_dir = "filtered_datasets/filtered_relevant_html/cookies"

dest_terms_dir = "filtered_datasets/filtered_relevant_plaintext_with_par/terms"
dest_cookies_dir = "filtered_datasets/filtered_relevant_plaintext_with_par/cookies"


def get_paths(_dir):
    """
    Returns paths if files in a directory
    :param _dir: directory
    :return: paths
    """
    files = os.listdir(_dir)
    for i in range(len(files)):
        files[i] = _dir + "/" + files[i]

    return files


def filter_files(_type: str, paths: list):
    """
    Filters file contents
    :param _type: cookies or terms
    :param paths: file paths
    :return:
    """
    dest_dir = ""
    if _type == "cookies":
        dest_dir = dest_cookies_dir
    else:
        dest_dir = dest_terms_dir

    for path in paths:
        # open source file and read contents
        s_file = open(path, "r", encoding='utf-8')
        res = process_file(s_file.read())
        s_file.close()

        # create new file
        filename = path.split("/")
        filename = filename[len(filename) - 1]
        # filters contents and write them into the new file
        d_file = open(dest_dir + "/" + filename, "w", encoding='utf-8')
        d_file.write(res)
        d_file.close()


def process_file(contents):
    """
    Filters file contents
    :param contents: contents
    :return: filtered contents
    """
    soup = BeautifulSoup(contents)

    # get all tags
    tags = soup.html.find_all()

    # remove HTML tag
    html = soup.find_all("html")
    html[0].unwrap()

    # for each tag
    for t in tags:
        # all non-paragraphs are unwrapped
        if t.name != "p":
            t.unwrap()
        else:
            # all paragraphs have their attributes removed
            t.attrs = dict()

    # find and remove all comments
    comments = soup.findAll(text=lambda text: isinstance(text, bs4.Comment))
    for comment in comments:
        comment.extract()

    # return html code
    text = soup.prettify()
    return text


def main():
    cookies_paths = get_paths(src_cookies_dir)
    terms_paths = get_paths(src_terms_dir)

    filter_files("cookies", cookies_paths)
    filter_files("terms", terms_paths)


main()