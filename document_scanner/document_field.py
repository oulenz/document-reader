

from abc import ABC

class Document_field(ABC):

    def __init__(self, name: str):
        self.name = name
        self.coordinates = None
        self.image = None
        self.value = None

