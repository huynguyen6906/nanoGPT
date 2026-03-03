import os
import sys
sys.path.append(os.path.abspath(os.path.join("..", "interface")))

from interface import data_loader
import random
import logging

class Bigrams(data_loader.Data):
    """Data loader class for managing text datasets with train/dev/test splits."""

    def __init__(self, file_path: str, logger: logging.Logger,
                 train_ratio: float = 0.7, dev_ratio: float = 0.15) -> None:
        """Initialize Data loader with file path and train/dev/test split ratios.

        Parameters
        ----------
        file_path : str
            Path to the text file containing data
        logger : logging.Logger
            Logger instance for logging operations
        train_ratio : float
            Ratio of data to use for training (default: 0.7)
        dev_ratio : float
            Ratio of data to use for development (default: 0.15)
        """
        self.__file_path = file_path
        self.logger = logger
        self.__train_ratio = train_ratio
        self.__dev_ratio = dev_ratio

        try:
            # Load and clean data from file
            with open(file_path, 'r', encoding='utf-8') as f:
                self.__rawData = [line.strip() for line in f if line.strip()]

            if not self.__rawData:
                raise ValueError(f"No data found in {file_path}")

            # Calculate dataset statistics
            self.__max_len = max(len(word) for word in self.__rawData)
            self.__min_len = min(len(word) for word in self.__rawData)
            self.__total_samples = len(self.__rawData)

            logger.info(f"Successfully loaded {self.__total_samples} samples")
            logger.info(f"Dataset statistics - Max length: {self.__max_len}, Min length: {self.__min_len}")

            # Create shuffled copy for train/dev/test splits
            self.__shuffleData = self.__rawData.copy()
            self.shuffle()

        except FileNotFoundError:
            logger.error(f"Data file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise

    def getData(self) -> list[str]:
        """Get the raw data as a list of strings.

        Returns
        -------
        list[str]
            List of all data samples (unshuffled)
        """
        return self.__rawData

    def getStats(self) -> tuple[int, int, int]:
        """Get dataset statistics.

        Returns
        -------
        tuple[int, int, int]
            Total samples, maximum length, minimum length
        """
        return self.__total_samples, self.__max_len, self.__min_len

    def shuffle(self) -> None:
        """Shuffle the dataset in place.

        Note
        ----
        This modifies the internal shuffleData list used for train/dev/test splits
        """
        random.shuffle(self.__shuffleData)
        self.logger.debug("Dataset shuffled successfully")

    def getSplitIndices(self) -> tuple[int, int]:
        """Get the split indices for train/dev/test sets.

        Returns
        -------
        tuple[int, int]
            End index for training set, end index for development set
            Test set uses remaining samples after dev_end
        """
        train_end = int(self.__total_samples * self.__train_ratio)
        dev_end = train_end + int(self.__total_samples * self.__dev_ratio)
        return train_end, dev_end

    def trainData(self) -> list[str]:
        """Get the training data subset.

        Returns
        -------
        list[str]
            List of training samples (first train_ratio fraction of shuffled data)
        """
        train_end, _ = self.getSplitIndices()
        return self.__shuffleData[:train_end]

    def devData(self) -> list[str]:
        """Get the development data subset.

        Returns
        -------
        list[str]
            List of development samples (middle dev_ratio fraction of shuffled data)
        """
        train_end, dev_end = self.getSplitIndices()
        return self.__shuffleData[train_end:dev_end]

    def testData(self) -> list[str]:
        """Get the test data subset.

        Returns
        -------
        list[str]
            List of test samples (remaining samples after dev_end)
        """
        _, dev_end = self.getSplitIndices()
        return self.__shuffleData[dev_end:]

    def getBatches(self, batch_size: int) -> list[list[str]]:
        """Get training data in batches.

        Parameters
        ----------
        batch_size : int
            Number of samples per batch

        Returns
        -------
        list[list[str]]
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
        if any(len(word) == 0 for word in self.__rawData):
            self.logger.warning("Validation failed: Empty strings found in dataset")
            return False

        if any(len(word) > 1000 for word in self.__rawData):
            self.logger.warning("Validation failed: Very long strings found (possibly corrupted)")
            return False

        if not all(isinstance(word, str) for word in self.__rawData): # type: ignore
            self.logger.warning("Validation failed: Non-string data types found")
            return False

        return True