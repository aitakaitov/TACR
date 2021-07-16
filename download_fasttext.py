import requests
import shutil
import gzip
import os


def download_file(url):
    print("Downloading from " + url)
    filename = url.split('/')[-1]
    try:
        with requests.get(url, stream=True) as url_in:
            with open(filename, "wb") as file:
                for chunk in url_in.iter_content(chunk_size=4096):
                    file.write(chunk)
                file.flush()
    except OSError:
        print("Not enough space on drive")
        exit(1)
    return filename


def extract(file):
    print("Extracting " + file)
    try:
        with gzip.open(file, "rb") as f_in:
            with open("fasttext/" + file[0:-3], "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
                f_out.flush()
    except OSError:
        print("Not enough space on drive")
        exit(1)

    os.remove(file)


os.mkdir("fasttext")

text_ft_url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.cs.300.vec.gz"
text_ft_gz = download_file(text_ft_url)
extract("cc.cs.300.vec.gz")

bin_ft_url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.cs.300.bin.gz"
bin_ft_gz = download_file(bin_ft_url)
extract("cc.cs.300.bin.gz")





