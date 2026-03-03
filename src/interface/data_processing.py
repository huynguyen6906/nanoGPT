from abc import ABC, abstractmethod

class Processing(ABC):
    @abstractmethod
    def processedData(self) -> list[list[int]]:
        pass