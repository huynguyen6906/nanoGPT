from interface.data_loader import Data

DATA = Data("data/raw/names.txt", 0.7, 0.15)
TRAIN_DATA = DATA.trainData()
