import os.path as path
import pickle

class Constructor:
    def __init__(self, f, cache_path: str):
        self.f = f
        self.cache_path = cache_path
    def construct(self):
        if path.exists(self.cache_path):
            print(f"path: {self.cache_path} exists")
            with open(self.cache_path, "rb") as f:
                return pickle.load(f)
        else:
            print(f"path: {self.cache_path} does not exist")
            result = self.f()
            print(f"saving to {self.cache_path}")
            with open(self.cache_path, "wb") as f:
                pickle.dump(result, f)
            return result
