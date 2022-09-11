import json


class Config:
    def __init__(self, path):
        self.config_path = path
        self.update()

    def update(self):
        config = json.load(open(self.config_path))
        self.__dict__.update(config)

    @property
    def dict(self):
        return self.__dict__
