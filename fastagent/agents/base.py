"""
Base Agent Module
Provides the foundational BaseAgent class and AgentRegistry for managing different agent types.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Type, Any

from fastagent.llm import LLMClient
from fastagent.memory import Memory
from fastagent.utils.logging import Logger

if TYPE_CHECKING:
    from fastagent.agents.coordinator import AgentCoordinator
    from fastagent.agents.agent_data_manager import AgentDataManager
    from fastagent.grounding.core.grounding_client import GroundingClient

logger = Logger.get_logger(__name__)


class BaseAgent(ABC):
    def __init__(
        self,
        name: str,
        backend_scope: Optional[List[str]] = None,
        llm_client: Optional[LLMClient] = None,
        coordinator: Optional[AgentCoordinator] = None,
    ) -> None:
        """
        Initialize the BaseAgent.
        
        Args:
            name: Unique name for the agent
            backend_scope: List of backend types this agent can access (e.g., ["gui", "shell", "mcp"])
            llm_client: LLM client for agent reasoning (optional, can be set later)
            coordinator: Reference to AgentCoordinator for resource access and communication
        """
        self._name = name
        self._coordinator: Optional[AgentCoordinator] = coordinator
        self._backend_scope = backend_scope or []
        self._llm_client = llm_client
        self._step = 0
        self._status = AgentStatus.ACTIVE
        
        from fastagent.agents.agent_data_manager import AgentDataManager
        
        if not coordinator or not hasattr(coordinator, 'storage_manager'):
            raise ValueError(
                f"{name}: Coordinator with storage_manager is required. "
                f"Please ensure AgentCoordinator is properly initialized."
            )
        
        self._data_manager = AgentDataManager(
            agent_name=name,
            storage_manager=coordinator.storage_manager,
            max_memory_items=50,
            compression_threshold=30
        )
        
        self._memory = self._data_manager.memory
        
        self._register_self()
        
        if coordinator:
            coordinator.register_agent(self)
        
        logger.info(f"Initialized {self.__class__.__name__}: {name}")

    @property
    def name(self) -> str:
        return self._name

    @property
    def memory(self) -> Memory:
        """
        Get the agent's memory.
        Memory is agent-local storage for conversation/interaction history.
        """
        return self._memory
    
    @property
    def data_manager(self) -> "AgentDataManager":
        """
        Get the agent's data manager.
        
        AgentDataManager provides unified management of:
        - Memory (short-term working memory)
        - Content processing (classification, filtering)
        - Automatic compression
        - LLM context building
        """
        return self._data_manager

    @property
    def coordinator(self) -> Optional[AgentCoordinator]:
        return self._coordinator
    
    def get_grounding_client(self) -> Optional[GroundingClient]:
        if self._coordinator:
            return self._coordinator.grounding_client
        return None

    @property
    def backend_scope(self) -> List[str]:
        return self._backend_scope

    @property
    def llm_client(self) -> Optional[LLMClient]:
        return self._llm_client

    @llm_client.setter
    def llm_client(self, client: LLMClient) -> None:
        self._llm_client = client

    @property
    def step(self) -> int:
        return self._step

    @property
    def status(self) -> str:
        return self._status

    @abstractmethod
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def construct_messages(self, **kwargs) -> List[Dict[str, Any]]:
        pass

    async def get_llm_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List] = None,
        **kwargs
    ) -> Dict[str, Any]:
        if not self._llm_client:
            raise ValueError(f"LLM client not initialized for agent {self.name}")
        
        try:
            response = await self._llm_client.complete(
                messages=messages,
                tools=tools,
                **kwargs
            )
            return response
        except Exception as e:
            logger.error(f"{self.name}: LLM call failed: {e}", exc_info=True)
            raise

    def response_to_dict(self, response: str) -> Dict[str, Any]:
        try:
            if response.strip().startswith("```json") or response.strip().startswith("```"):
                lines = response.strip().split('\n')
                if lines and lines[0].startswith('```'):
                    lines = lines[1:]
                end_idx = len(lines)
                for i, line in enumerate(lines):
                    if line.strip() == '```':
                        end_idx = i
                        break
                response = '\n'.join(lines[:end_idx])
            
            return json.loads(response)
        except json.JSONDecodeError as e:
            # If parsing fails, try to find and extract just the JSON object/array
            if "Extra data" in str(e):
                try:
                    decoder = json.JSONDecoder()
                    obj, idx = decoder.raw_decode(response)
                    logger.warning(
                        f"{self.name}: Successfully extracted JSON but found extra text after position {idx}. "
                        f"Extra text: {response[idx:idx+100]}..."
                    )
                    return obj
                except Exception as e2:
                    logger.error(f"{self.name}: Failed to extract JSON even with raw_decode: {e2}")
            
            logger.error(f"{self.name}: Failed to parse response: {e}")
            logger.error(f"{self.name}: Response content: {response[:500]}")
            return {"error": "Failed to parse response", "raw": response}

    def increment_step(self) -> None:
        self._step += 1

    @classmethod
    def _register_self(cls) -> None:
        """Register the agent class in the registry upon instantiation."""
        # Get the actual instance class, not BaseAgent
        if cls.__name__ != "BaseAgent" and cls.__name__ not in AgentRegistry._registry:
            AgentRegistry.register(cls.__name__, cls)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, step={self.step}, status={self.status})>"


class AgentStatus:
    """Constants for agent status."""
    ACTIVE = "active"
    IDLE = "idle"
    WAITING = "waiting"


class AgentRegistry:
    """
    Registry for managing agent classes.
    Allows dynamic registration and retrieval of agent types.
    """

    _registry: Dict[str, Type[BaseAgent]] = {}

    @classmethod
    def register(cls, name: str, agent_cls: Type[BaseAgent]) -> None:
        if name in cls._registry:
            logger.warning(f"Agent class '{name}' already registered, overwriting")
        cls._registry[name] = agent_cls
        logger.debug(f"Registered agent class: {name}")

    @classmethod
    def get_cls(cls, name: str) -> Type[BaseAgent]:
        if name not in cls._registry:
            raise ValueError(f"No agent class registered under '{name}'")
        return cls._registry[name]

    @classmethod
    def list_registered(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()
        logger.debug("Agent registry cleared")