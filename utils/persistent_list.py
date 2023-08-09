import _pickle as pickle
import os.path
from .log import Log


class PersistentList:
    def __init__(self, file_path):
        self.path = 'persistence_temp/' + file_path
        self.list = []
        self.log = Log('log.txt')

        try:
            os.mkdir('persistence_temp')
        except OSError:
            self.log.log("[CRAWLER - PERSISTENT] Persistent folder already exists.")

        if os.path.exists(self.path):
            file = open(self.path, "rb")
            self.list = pickle.load(file)
            self.log.log("[CRAWLER - PERSISTENT] File {} found, loading the file.".format(self.path))

        else:
            file = open(self.path, "wb")
            pickle.dump(self.list, file)
            self.log.log("[CRAWLER - PERSISTENT] File {} not found, creating the file.".format(self.path))

        file.close()

    def append(self, object) -> None:
        file = open(self.path, "wb")
        self.list.append(object)
        pickle.dump(self.list, file)
        file.close()

    def remove(self, object) -> None:
        file = open(self.path, "wb")
        self.list.remove(object)
        pickle.dump(self.list, file)
        file.close()

    def extend(self, object) -> None:
        file = open(self.path, "wb")
        self.list.extend(object)
        pickle.dump(self.list, file)
        file.close()

    def __iter__(self):
        return iter(self.list)

    def __getitem__(self, item):
        return self.list[item]

    def __len__(self):
        return len(self.list)
