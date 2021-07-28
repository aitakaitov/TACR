import os

root_folder = "ad_pages/merged"
words = ["komercni sdeleni", "komercni clanek", "komerční sdělení", "komerční článek", "komerční prezentace", "komercni prezentace", "sponzorovaný článek",
         "sponzorovany clanek"]

dirs = os.listdir(root_folder)

for _dir in dirs:
    files = os.listdir(root_folder + "/" + _dir)
    for file in files:
        with open(root_folder + '/' + _dir + '/' + file) as f:
            text = f.read()
            for word in words:
                if word in text:
                    print(word + " in " + root_folder + "/" + _dir + "/" + file)
