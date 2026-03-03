import os
import sys
sys.path.append(os.path.abspath(".."))
from interface import data_processing

class Bigrams(data_processing.Processing):
    def __init__(self, data: list[str]):
        self.__data = data
        self.__processed_data, self.__itos, self.__stoi = self.__processing()

    def processedData(self) -> list[list[int]]:
        return self.__processed_data
    
    def stoi(self, ch: str):
        return self.__stoi[ch]
    
    def itos(self, i: int):
        return self.__itos[i]
        
    def __processing(self) -> tuple[list[list[int]], list[str], dict[str, int]]:
        chs: set[str] = set()
        for word in self.__data:
            for ch in word:
                chs.add(ch)
        itos = sorted(chs) + ['.']
        stoi = {ch: i for i, ch in enumerate(itos)}
        processed_data = [[0]*len(itos) for _ in range(len(itos))]
        for word in self.__data:
            word = '.' + word + '.'
            for ch1, ch2 in zip(word, word[1:]):
                processed_data[stoi[ch1]][stoi[ch2]] += 1
        print(processed_data)
        return processed_data, itos, stoi
    
a = Bigrams(["dcba"])