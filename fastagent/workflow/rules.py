"""
Workflow Rule Definitions
Define workflow rules to control task triggering conditions and execution flow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Any, Callable

from fastagent.kanban import CardType, CardStatus
from fastagent.utils.logging import Logger

if TYPE_CHECKING:
    from fastagent.kanban import KanbanCard

logger = Logger.get_logger(__name__)


class WorkflowRule:
    """
    Workflow Rule: Define which agent to trigger when specific conditions are met
    
    A rule consists of:
    1. Trigger conditions: card type, status, additional condition function
    2. Target agent: name of the agent to trigger
    3. Priority: determines execution order of rules
    4. Hook functions: callbacks before and after execution
    """
    
    def __init__(
        self,
        name: str,
        card_type: CardType,
        card_status: CardStatus,
        agent_name: str,
        priority: int = 0,
        condition: Optional[Callable[[KanbanCard], bool]] = None,
        pre_hook: Optional[Callable[[KanbanCard], Any]] = None,
        post_hook: Optional[Callable[[KanbanCard, Any], None]] = None,
        timeout: Optional[float] = None,
    ):
        """
        Initialize workflow rule
        
        Args:
            name: Rule name (unique identifier)
            card_type: Card type that triggers this rule
            card_status: Card status that triggers this rule
            agent_name: Name of agent to trigger
            priority: Priority level (higher number = higher priority, default 0)
            condition: Additional condition function, triggers only when returns True
            pre_hook: Hook function to execute before agent (sync/async)
            post_hook: Hook function to execute after agent (sync/async)
            timeout: Task timeout in seconds, None means no limit
        """
        self.name = name
        self.card_type = card_type
        self.card_status = card_status
        self.agent_name = agent_name
        self.priority = priority
        self.condition = condition
        self.pre_hook = pre_hook
        self.post_hook = post_hook
        self.timeout = timeout
        
        self._validate()
    
    def _validate(self) -> None:
        """Validate rule parameters"""
        if not self.name:
            raise ValueError("Rule name cannot be empty")
        
        # agent_name can be None (rule only executes hooks without triggering agent)
        if self.agent_name is not None and not self.agent_name:
            raise ValueError("Agent name cannot be empty string")
        
        if not isinstance(self.card_type, CardType):
            raise ValueError(f"Invalid card_type: {self.card_type}")
        
        if not isinstance(self.card_status, CardStatus):
            raise ValueError(f"Invalid card_status: {self.card_status}")
        
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError(f"Timeout must be positive, got {self.timeout}")
    
    def matches(self, card: KanbanCard) -> bool:
        """
        Check if a card matches this rule
        
        Args:
            card: Card to check
            
        Returns:
            True if matched, False otherwise
        """
        if card.card_type != self.card_type:
            return False
        
        if card.status != self.card_status:
            return False
        
        if self.condition:
            try:
                return self.condition(card)
            except Exception as e:
                logger.warning(f"Rule '{self.name}': condition check failed: {e}")
                return False
        
        return True
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"WorkflowRule(name='{self.name}', "
            f"trigger={self.card_type.value}/{self.card_status.value}, "
            f"agent={self.agent_name}, priority={self.priority})"
        )
    
    def __eq__(self, other: Any) -> bool:
        """Compare rules (based on name)"""
        if not isinstance(other, WorkflowRule):
            return False
        return self.name == other.name
    
    def __hash__(self) -> int:
        """Hash (based on name)"""
        return hash(self.name)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "card_type": self.card_type.value,
            "card_status": self.card_status.value,
            "agent_name": self.agent_name,
            "priority": self.priority,
            "has_condition": self.condition is not None,
            "has_pre_hook": self.pre_hook is not None,
            "has_post_hook": self.post_hook is not None,
            "timeout": self.timeout,
        }