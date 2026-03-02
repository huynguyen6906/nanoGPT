import os
import random
import logging
from typing import List, Tuple

# Configure logging with both file and console handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('data_loader.log')
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

class Data:
    """Data loader class for managing text datasets with train/dev/test splits."""

    def __init__(self, file_path: str = "../data/raw/names.txt",
                 train_ratio: float = 0.7, dev_ratio: float = 0.15) -> None:
        """Initialize Data loader with file path and train/dev/test split ratios.

        Parameters
        ----------
        file_path : str
            Path to the text file containing data
        train_ratio : float
            Ratio of data to use for training (default: 0.7)
        dev_ratio : float
            Ratio of data to use for development (default: 0.15)
        """
        self.file_path = file_path
        self.train_ratio = train_ratio
        self.dev_ratio = dev_ratio

        try:
            # Load and clean data from file
            with open(file_path, 'r', encoding='utf-8') as f:
                self.rawData = [line.strip() for line in f if line.strip()]

            if not self.rawData:
                raise ValueError(f"No data found in {file_path}")

            # Calculate dataset statistics
            self.max_len = max(len(word) for word in self.rawData)
            self.min_len = min(len(word) for word in self.rawData)
            self.total_samples = len(self.rawData)

            logger.info(f"Successfully loaded {self.total_samples} samples")
            logger.info(f"Dataset statistics - Max length: {self.max_len}, Min length: {self.min_len}")

            # Create shuffled copy for train/dev/test splits
            self.shuffleData = self.rawData.copy()
            self.shuffle()

        except FileNotFoundError:
            logger.error(f"Data file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise

    def getData(self) -> List[str]:
        """Get the raw data as a list of strings.

        Returns
        -------
        List[str]
            List of all data samples (unshuffled)
        """
        return self.rawData

    def getStats(self) -> Tuple[int, int, int]:
        """Get dataset statistics.

        Returns
        -------
        Tuple[int, int, int]
            Total samples, maximum length, minimum length
        """
        return self.total_samples, self.max_len, self.min_len

    def shuffle(self) -> None:
        """Shuffle the dataset in place.

        Note
        ----
        This modifies the internal shuffleData list used for train/dev/test splits
        """
        random.shuffle(self.shuffleData)
        logger.debug("Dataset shuffled successfully")

    def getSplitIndices(self) -> Tuple[int, int]:
        """Get the split indices for train/dev/test sets.

        Returns
        -------
        Tuple[int, int]
            End index for training set, end index for development set
            Test set uses remaining samples after dev_end
        """
        train_end = int(self.total_samples * self.train_ratio)
        dev_end = train_end + int(self.total_samples * self.dev_ratio)
        return train_end, dev_end

    def trainData(self) -> List[str]:
        """Get the training data subset.

        Returns
        -------
        List[str]
            List of training samples (first train_ratio fraction of shuffled data)
        """
        train_end, _ = self.getSplitIndices()
        return self.shuffleData[:train_end]

    def devData(self) -> List[str]:
        """Get the development data subset.

        Returns
        -------
        List[str]
            List of development samples (middle dev_ratio fraction of shuffled data)
        """
        train_end, dev_end = self.getSplitIndices()
        return self.shuffleData[train_end:dev_end]

    def testData(self) -> List[str]:
        """Get the test data subset.

        Returns
        -------
        List[str]
            List of test samples (remaining samples after dev_end)
        """
        _, dev_end = self.getSplitIndices()
        return self.shuffleData[dev_end:]

    def getBatches(self, batch_size: int) -> List[List[str]]:
        """Get training data in batches.

        Parameters
        ----------
        batch_size : int
            Number of samples per batch

        Returns
        -------
        List[List[str]]
            List of batches, each containing batch_size samples
            Last batch may be smaller if total samples not divisible by batch_size
        """
        train_data = self.trainData()
        return [train_data[i:i + batch_size]
                for i in range(0, len(train_data), batch_size)]

    def validateData(self) -> bool:
        """Validate the dataset for common issues.

        Returns
        -------
        bool
            True if dataset passes validation, False otherwise

        Raises
        ------
        ValueError
            If validation fails due to critical issues
        """
        if any(len(word) == 0 for word in self.rawData):
            logger.warning("Validation failed: Empty strings found in dataset")
            return False

        if any(len(word) > 1000 for word in self.rawData):
            logger.warning("Validation failed: Very long strings found (possibly corrupted)")
            return False

        if not all(isinstance(word, str) for word in self.rawData):
            logger.warning("Validation failed: Non-string data types found")
            return False

        return True