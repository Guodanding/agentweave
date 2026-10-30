from abc import ABC, abstractmethod
from typing import List
from agentweave.scheduler.state import RequestState, RoundTask


class BaseAgent(ABC):

    @abstractmethod
    async def process_batch(self, tasks: List[RoundTask], states: List[RequestState]):
        pass

    @abstractmethod
    def get_initial_state_config(self, doc: str, config: dict) -> dict:
        return config
