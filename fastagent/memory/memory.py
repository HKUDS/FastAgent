"""
Memory Module for Agent-Local State

"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastagent.utils.logging import Logger

logger = Logger.get_logger(__name__)


@dataclass
class MemoryItem:
    """
    Represents a single memory item for an agent at one step.
    
    This is a flexible data structure that can store arbitrary key-value pairs.
    Typically contains:
    - type: "llm_interaction" or "execution" or "decision"
    - role: "user" or "assistant" or "system" (for LLM interactions)
    - content: The actual content
    - timestamp: When this happened
    - step: Step number
    """
    
    _memory_attributes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for key, value in self.__dict__.items():
            if key != "_memory_attributes" and key in self._memory_attributes:
                if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                    result[key] = value
                else:
                    result[key] = str(value)
        return result
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            self.set_value(key, value)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    def set_value(self, key: str, value: Any) -> None:
        setattr(self, key, value)
        if key not in self._memory_attributes:
            self._memory_attributes.append(key)
    
    def get_value(self, key: str) -> Optional[Any]:
        return getattr(self, key, None)
    
    def get_values(self, keys: List[str]) -> Dict[str, Any]:
        return {key: self.get_value(key) for key in keys}
    
    def add_values_from_dict(self, values: Dict[str, Any]) -> None:
        for key, value in values.items():
            self.set_value(key, value)
    
    @property
    def attributes(self) -> List[str]:
        return self._memory_attributes.copy()

    def __setattr__(self, key: str, value: Any) -> None:
        super().__setattr__(key, value)
        
        should_track = (
            not key.startswith("_") and      
            not key.startswith("tmp_") and   
            key != "_memory_attributes"     
        )
        
        if should_track:
            if not hasattr(self, "_memory_attributes"):
                super().__setattr__("_memory_attributes", [])
            
            if key not in self._memory_attributes:
                self._memory_attributes.append(key)


@dataclass
class Memory:
    """
    Stores a sequence of MemoryItem objects representing the agent's history.
    """
    
    _content: List[MemoryItem] = field(default_factory=list)
    
    def load(self, content: List[MemoryItem]) -> None:
        """Load memory content."""
        self._content = content
    
    def add_memory_item(
        self, 
        memory_item: MemoryItem,
        agent_name: Optional[str] = None,
        step: Optional[int] = None,
        auto_timestamp: bool = True,
    ) -> None:
        """
        Add a memory item with automatic metadata filling.
        
        Args:
            memory_item: The memory item to add
            agent_name: Agent name (auto-filled if not present)
            step: Step number (auto-filled if not present)
            auto_timestamp: Whether to auto-fill timestamp
        """
        # Auto-fill step
        if step is not None and memory_item.get_value("step") is None:
            memory_item.set_value("step", step)
        
        # Auto-fill agent_name
        if agent_name is not None and memory_item.get_value("agent_name") is None:
            memory_item.set_value("agent_name", agent_name)
        
        # Auto-fill timestamp
        if auto_timestamp and memory_item.get_value("timestamp") is None:
            memory_item.set_value("timestamp", datetime.now().isoformat())
        
        self._content.append(memory_item)
    
    def clear(self) -> None:
        """Clear all memory."""
        self._content.clear()
    
    @property
    def length(self) -> int:
        """Get memory length."""
        return len(self._content)
    
    def delete_memory_item(self, step: int) -> None:
        """Delete memory item at specific step."""
        self._content = [
            item for item in self._content
            if item.get_value("step") != step
        ]
    
    def to_json(self) -> str:
        return json.dumps([item.to_dict() for item in self._content if item])
    
    def to_list_of_dicts(self) -> List[Dict[str, Any]]:
        return [item.to_dict() for item in self._content]
    
    def from_list_of_dicts(self, data: List[Dict[str, Any]]) -> None:
        self._content.clear()
        for item_dict in data:
            memory_item = MemoryItem()
            memory_item.from_dict(item_dict)
            self._content.append(memory_item)
    
    def get_latest_item(self) -> Optional[MemoryItem]:
        """Get the most recent memory item."""
        if self.length == 0:
            return None
        return self._content[-1]
    
    @property
    def content(self) -> List[MemoryItem]:
        return self._content
    
    @property
    def list_content(self) -> List[Dict[str, Any]]:
        return [item.to_dict() for item in self._content]
    
    def is_empty(self) -> bool:
        return self.length == 0
    
    def filter_memory_from_steps(self, steps: List[int]) -> List[Dict[str, Any]]:
        return [
            item.to_dict() for item in self._content
            if item.get_value("step") in steps
        ]
    
    def filter_memory_from_keys(self, keys: List[str]) -> List[Dict[str, Any]]:
        result = []
        for item in self._content:
            filtered = {k: item.get_value(k) for k in keys if item.get_value(k) is not None}
            if filtered:
                result.append(filtered)
        return result

