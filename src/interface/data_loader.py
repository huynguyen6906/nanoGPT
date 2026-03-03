from abc import ABC, abstractmethod

class Data (ABC):
    """Abstract base class defining the interface for data loading operations."""

    @abstractmethod
    def getData(self) -> list[str]:
        """Get the raw data as a list of strings.

        Returns
        -------
        list[str]
            List of all data samples (unshuffled)
        """

    @abstractmethod
    def getStats(self) -> tuple[int, int, int]:
        """Get dataset statistics.

        Returns
        -------
        tuple[int, int, int]
            Total samples, maximum length, minimum length
        """

    @abstractmethod
    def shuffle(self) -> None:
        """Shuffle the dataset in place.

        Note
        ----
        This modifies the internal data structure used for train/dev/test splits
        """

    @abstractmethod
    def getSplitIndices(self) -> tuple[int, int]:
        """Get the split indices for train/dev/test sets.

        Returns
        -------
        tuple[int, int]
            End index for training set, end index for development set
            Test set uses remaining samples after dev_end
        """

    @abstractmethod
    def trainData(self) -> list[str]:
        """Get the training data subset.

        Returns
        -------
        list[str]
            List of training samples (first train_ratio fraction of shuffled data)
        """

    @abstractmethod
    def devData(self) -> list[str]:
        """Get the development data subset.

        Returns
        -------
        list[str]
            List of development samples (middle dev_ratio fraction of shuffled data)
        """

    @abstractmethod
    def testData(self) -> list[str]:
        """Get the test data subset.

        Returns
        -------
        list[str]
            List of test samples (remaining samples after dev_end)
        """

    @abstractmethod
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

    @abstractmethod
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