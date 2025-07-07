from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    All agent analyze methods should expect a 'ticker' key in their input data.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def analyze(self, data):
        """
        Analyze the provided data for a given stock ticker.
        Args:
            data (dict): Must include a 'ticker' key.
        Returns:
            dict: Analysis result.
        """
        pass 