import _pickle as pickle
import os.path
from crawler.log import Log
from crawler.config import Config


class PersistentList:
    def __init__(self, file_path):
        self.path = Config.persistent_path + "/" + file_path
        self.list = []
        self.log = Log(Config.log_path)

        try:
            os.mkdir("./" + Config.persistent_path)
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
        self.log.log("[CRAWLER - PERSISTENT] Saving persistent state")
        file = open(self.path, "wb")
        self.list.append(object)
        pickle.dump(self.list, file)
        file.close()
        self.log.log("[CRAWLER - PERSISTENT] Finished saving persistent state")

    def remove(self, object) -> None:
        self.log.log("[CRAWLER - PERSISTENT] Saving persistent state")
        file = open(self.path, "wb")
        self.list.remove(object)
        pickle.dump(self.list, file)
        file.close()
        self.log.log("[CRAWLER - PERSISTENT] Finished saving persistent state")

    def __iter__(self):
        return iter(self.list)

    def __getitem__(self, item):
        return self.list[item]

    def __len__(self):
        return len(self.list)
