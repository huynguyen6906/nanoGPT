import os
import random
import logging
from typing import List, Tuple

# Configure logging with both file and console handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# File handler for persistent logging
file_handler = logging.FileHandler('data_loader.log')
file_handler.setLevel(logging.DEBUG)

# Console handler for real-time feedback
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Set formatter for both handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

class Data:
    """
    Data loader class for text processing tasks.

    Handles loading text data, splitting into train/dev/test sets,
    and provides basic statistics and shuffling functionality.
    """

    def __init__(self, file_path: str = "../data/raw/names.txt",
                 train_ratio: float = 0.7, dev_ratio: float = 0.15) -> None:
        """
        Initialize the Data loader with configuration options.

        Args:
            file_path: Path to the text file containing data (default: ../data/raw/names.txt)
            train_ratio: Proportion of data for training (0-1, default: 0.7)
            dev_ratio: Proportion of data for development (0-1, default: 0.15)

        Raises:
            FileNotFoundError: If the specified file does not exist
            ValueError: If the file is empty or contains invalid data
        """
        self.file_path = file_path
        self.train_ratio = train_ratio
        self.dev_ratio = dev_ratio

        try:
            # Load and preprocess data with UTF-8 encoding
            # Strip whitespace and filter out empty lines
            with open(file_path, 'r', encoding='utf-8') as f:
                self.rawData = [line.strip() for line in f if line.strip()]

            # Validate that data was loaded successfully
            if not self.rawData:
                raise ValueError(f"No data found in {file_path}")

            # Calculate basic statistics for the dataset
            # Max length: longest word in the dataset
            # Min length: shortest word in the dataset
            self.max_len = max(len(word) for word in self.rawData)
            self.min_len = min(len(word) for word in self.rawData)
            self.total_samples = len(self.rawData)

            # Log successful data loading with statistics
            logger.info(f"Successfully loaded {self.total_samples} samples")
            logger.info(f"Dataset statistics - Max length: {self.max_len}, Min length: {self.min_len}")

            # Create a shuffled copy for data augmentation
            # This preserves the original order while providing shuffled data
            self.shuffleData = self.rawData.copy()
            self.shuffle()

        except FileNotFoundError:
            # Log and re-raise file not found errors
            logger.error(f"Data file not found: {file_path}")
            raise
        except Exception as e:
            # Catch-all for any other exceptions during data loading
            logger.error(f"Error loading data from {file_path}: {e}")
            raise

    def getData(self) -> List[str]:
        """
        Return the complete raw dataset.

        Returns:
            List of all data samples in their original order
        """
        return self.rawData

    def getStats(self) -> Tuple[int, int, int]:
        """
        Return comprehensive dataset statistics.

        Returns:
            Tuple containing:
            - total_samples: Total number of data samples
            - max_length: Length of the longest sample
            - min_length: Length of the shortest sample
        """
        return self.total_samples, self.max_len, self.min_len

    def shuffle(self) -> None:
        """
        Shuffle the dataset in-place using Fisher-Yates algorithm.

        This method randomizes the order of samples in shuffleData
        while preserving the original rawData order.
        """
        random.shuffle(self.shuffleData)
        logger.debug("Dataset shuffled successfully")

    def getSplitIndices(self) -> Tuple[int, int]:
        """
        Calculate indices for train/dev/test split based on configured ratios.

        Uses integer arithmetic to ensure indices are valid list positions.
        The test set automatically receives any remaining samples.

        Returns:
            Tuple of (train_end_index, dev_end_index) where:
            - train_end_index: End index for training data (exclusive)
            - dev_end_index: End index for development data (exclusive)
        """
        train_end = int(self.total_samples * self.train_ratio)
        dev_end = train_end + int(self.total_samples * self.dev_ratio)
        return train_end, dev_end

    def trainData(self) -> List[str]:
        """
        Return the training data subset.

        Uses the configured train_ratio to determine the split point.
        Returns data in original order (not shuffled).

        Returns:
            List of training samples
        """
        train_end, _ = self.getSplitIndices()
        return self.rawData[:train_end]

    def devData(self) -> List[str]:
        """
        Return the development (validation) data subset.

        Uses the configured dev_ratio to determine the split point.
        Returns data in original order (not shuffled).

        Returns:
            List of development samples
        """
        train_end, dev_end = self.getSplitIndices()
        return self.rawData[train_end:dev_end]

    def testData(self) -> List[str]:
        """
        Return the test data subset.

        Automatically receives all remaining data after train/dev splits.
        Returns data in original order (not shuffled).

        Returns:
            List of test samples
        """
        _, dev_end = self.getSplitIndices()
        return self.rawData[dev_end:]

    def getBatches(self, batch_size: int) -> List[List[str]]:
        """
        Generate batches from the training data for efficient processing.

        Args:
            batch_size: Number of samples per batch (must be > 0)

        Returns:
            List of batches, each containing up to batch_size samples
            The last batch may contain fewer samples if total is not divisible

        Example:
            >>> data = Data()
            >>> batches = data.getBatches(32)  # Returns list of 32-sample batches
        """
        train_data = self.trainData()
        return [train_data[i:i + batch_size]
                for i in range(0, len(train_data), batch_size)]

    def validateData(self) -> bool:
        """
        Validate data quality and integrity.

        Checks for common data issues that could cause training problems:
        - Empty strings
        - Extremely long strings (possible corruption)
        - Non-string data types

        Returns:
            True if data passes all validation checks, False otherwise

        Side Effects:
            Logs warnings for any validation failures
        """
        # Check for empty strings
        if any(len(word) == 0 for word in self.rawData):
            logger.warning("Validation failed: Empty strings found in dataset")
            return False

        # Check for extremely long strings (possible data corruption)
        if any(len(word) > 1000 for word in self.rawData):
            logger.warning("Validation failed: Very long strings found (possibly corrupted)")
            return False

        # Check for non-string data types
        if not all(isinstance(word, str) for word in self.rawData):
            logger.warning("Validation failed: Non-string data types found")
            return False

        return True