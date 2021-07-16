import os

plain_dir = "filtered_datasets/filtered_relevant_plaintext"
html_dir = "filtered_datasets/filtered_relevant_html"



def delete_file(html_path: str):
    """
    Gets plaintext path and removes both files (html and plaintext)
    :param html_path: path to the HTML file
    :return: None
    """
    plain_path = html_path.replace(html_dir + "/", plain_dir + "/")
    os.remove(html_path)
    os.remove(plain_path)

    return


def delete_containing(html_paths: list, string: str):
    """
    Removes all files containing string in their name
    :param html_paths: List of HTML file paths
    :param string: string
    :return: None
    """
    removed = []
    for path in html_paths:
        if string in path:
            delete_file(path)
            removed.append(path)

    for r in removed:
        html_paths.remove(r)

    return


def remove_duplicates(html_paths: list):
    """
    Removes HTTP/HTTPS duplicates and URLs with and without / at the end
    :param html_paths: HTML file paths
    :return:
    """
    removed = []
    for path in html_paths:
        if path[-1] != "_":
            for cpath in html_paths:
                if cpath == path + "_":
                    delete_file(cpath)
                    removed.append(cpath)

    for r in removed:
        html_paths.remove(r)
    removed = []

    for path in html_paths:
        url = path.split("/")
        url = url[len(url) - 1]
        if url[0:5] == "https":
            for cpath in html_paths:
                curl = cpath.split("/")
                curl = curl[len(curl) - 1]
                if curl[0:4] == "http" and curl[0:4] == url[0:5]:
                    delete_file(cpath)
                    removed.append(cpath)

    for r in removed:
        html_paths.remove(r)

    return


def retain_one(html_paths: list, string: str):
    """
    Removes all files but one which contain a string
    :param html_paths: HTML file paths
    :param string: string
    :return: None
    """
    indices = []
    for i in range(len(html_paths)):
        if string in html_paths[i]:
            indices.append(i)

    removed = []
    for i in range(len(indices) - 1):
        path = html_paths[indices[i]]
        removed.append(path)
        delete_file(path)

    for r in removed:
        html_paths.remove(r)

    return


def clean_terms():
    """
    Cleans terms files
    :return:
    """

    # domains we want to remove
    domains = ["gdpr_spir_cz", "bylinkyprovsechny_cz", "rady_navody_cz", "zemedelske_potreby_cz"]

    paths = os.listdir(html_dir + "/terms")
    for i in range(len(paths)):
        paths[i] = html_dir + "/terms/" + paths[i]

    for domain in domains:
        delete_containing(paths, domain)

    remove_duplicates(paths)

    return


def clean_cookies():
    """
    Cleans cookies files
    :return:
    """

    # domains we want to remove
    domains = ["zemedelske_potreby_cz_zpracovani", "vodafone_cz_o_vodafonu_ke_stazeni", "uoou_cz",
               "officedepot_cz_ochrana", "ochranaprirody_cz", "mujeee_cz", "mpo_cz", "hledejceny_cz",
               "agromanual_cz_cz_clanky", "ceskykutil_cz_clanek", "gdpr_spir_cz"]

    # urls we want only one instance of
    retain = ["rrtv_cz_cz", "economia_cz_zpracovani_osobnich_udaju_", "economia_cz_prohlaseni_o_cookies_",
              "economia_cz_ochrana_osobnich_udaju"]

    paths = os.listdir(html_dir + "/cookies")
    for i in range(len(paths)):
        paths[i] = html_dir + "/cookies/" + paths[i]

    for domain in domains:
        delete_containing(paths, domain)

    for r in retain:
        retain_one(paths, r)

    remove_duplicates(paths)

    return


clean_cookies()
clean_terms()