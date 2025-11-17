"""
Kanban Board for Multi-Agent Coordination

This module provides a Kanban-style task management system for coordinating
work across multiple agents. It is separate from agent memory, while Memory
stores agent-local execution history. Kanban tracks system-wide task state.
"""

import json
import os
import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime

from fastagent.utils.logging import Logger
from fastagent.recording.utils import (
    load_trajectory_from_jsonl,
    analyze_trajectory
)
from fastagent.kanban.enums import CardType, CardStatus, KanbanEvent

logger = Logger.get_logger(__name__)


@dataclass
class KanbanCard:
    """
    Represents a card on the Kanban board.
    
    Each card tracks a high-level task or activity in the system.
    Cards are lightweight and should NOT contain detailed execution results
    (those belong in Recording system or Agent Memory).
    """
    
    card_id: str
    agent_name: str
    card_type: CardType
    status: CardStatus
    title: str
    description: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    step: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "card_id": self.card_id,
            "agent_name": self.agent_name,
            "card_type": self.card_type.value,
            "status": self.status.value,
            "title": self.title,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "step": self.step,
            "metadata": self.metadata,
        }
        
        # Include result if it exists (dynamically added attribute)
        # Important for RESPONSE and EVALUATION cards
        if hasattr(self, 'result') and self.result is not None:
            data["result"] = self.result
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KanbanCard":
        """Create from dictionary."""
        card_type = data.get("card_type", CardType.EXECUTION.value)
        status = data.get("status", CardStatus.TODO.value)
        
        if isinstance(card_type, str):
            card_type = CardType(card_type)
        if isinstance(status, str):
            status = CardStatus(status)
        
        return cls(
            card_id=data.get("card_id", ""),
            agent_name=data.get("agent_name", ""),
            card_type=card_type,
            status=status,
            title=data.get("title", ""),
            description=data.get("description", ""),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            step=data.get("step"),
            metadata=data.get("metadata", {}),
        )
    
    def update_status(self, status: CardStatus) -> None:
        """Update card status."""
        self.status = status
        self.updated_at = datetime.now().isoformat()


class KanbanColumn:
    """
    Represents a column on the Kanban board.
    Columns organize cards by status: TODO, In Progress, Done, etc.
    """
    
    def __init__(self, name: str, status_filter: Optional[List[CardStatus]] = None):
        """
        Initialize a Kanban column.
        
        Args:
            name: Column name
            status_filter: List of statuses that belong in this column
        """
        self.name = name
        self.status_filter = status_filter or []
        self._cards: List[KanbanCard] = []
    
    def add_card(self, card: KanbanCard) -> None:
        """Add a card to this column."""
        if not self.status_filter or card.status in self.status_filter:
            self._cards.append(card)
    
    def remove_card(self, card_id: str) -> Optional[KanbanCard]:
        """Remove and return a card by ID."""
        for i, card in enumerate(self._cards):
            if card.card_id == card_id:
                return self._cards.pop(i)
        return None
    
    def get_cards(self) -> List[KanbanCard]:
        """Get all cards in this column."""
        return self._cards.copy()
    
    def count(self) -> int:
        """Count cards in this column."""
        return len(self._cards)
    
    def clear(self) -> None:
        """Clear all cards."""
        self._cards.clear()


class Kanban:
    """
    Kanban board for multi-agent coordination.
    
    Tracks high-level tasks across all agents using a board metaphor.
    This is separate from agent Memory:
    
    - Kanban: System-wide task coordination (what needs to be done)
    - Memory: Agent-local execution history (what was done, for LLM context)
    - Recording: Detailed execution logs (complete history for analysis)
    """
    
    def __init__(self, coordinator: Optional[Any] = None):
        """
        Initialize an empty Kanban board.
        
        Args:
            coordinator: Optional AgentCoordinator for dynamic agent memory access
        """
        self._cards: Dict[str, KanbanCard] = {}  # card_id -> card
        self._coordinator = coordinator
        self._global_step = 0
        
        self._columns: Dict[CardStatus, KanbanColumn] = {
            CardStatus.TODO: KanbanColumn("TODO", [CardStatus.TODO]),
            CardStatus.IN_PROGRESS: KanbanColumn("In Progress", [CardStatus.IN_PROGRESS]),
            CardStatus.DONE: KanbanColumn("Done", [CardStatus.DONE]),
            CardStatus.BLOCKED: KanbanColumn("Blocked", [CardStatus.BLOCKED]),
        }
        
        # Event hooks for real-time monitoring
        self._event_hooks: Dict[KanbanEvent, List[Callable[[Dict[str, Any]], None]]] = {
            KanbanEvent.ON_CARD_ADDED: [],
            KanbanEvent.ON_CARD_UPDATED: [],
            KanbanEvent.ON_CARD_DELETED: [],
            KanbanEvent.ON_STEP_RECORDED: [],
        }
        
        self._hook_lock = asyncio.Lock()
        
        logger.info("Kanban board initialized")
    
    async def register_hook(
        self,
        event_type: Union[KanbanEvent, str],
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Register event hooks for real-time monitoring.
        
        Args:
            event_type: Event type
            callback: Callback function (can be sync or async)
        """
        if isinstance(event_type, str):
            try:
                event_type = KanbanEvent(event_type)
            except ValueError:
                logger.warning(f"Unknown event type: {event_type}")
                return
        
        if event_type not in self._event_hooks:
            logger.warning(f"Unknown event type: {event_type}")
            return
        
        async with self._hook_lock:
            self._event_hooks[event_type].append(callback)
        logger.debug(f"Registered hook for event: {event_type.value}")
    
    async def unregister_hook(
        self,
        event_type: Union[KanbanEvent, str],
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Unregister event hooks"""
        if isinstance(event_type, str):
            try:
                event_type = KanbanEvent(event_type)
            except ValueError:
                logger.warning(f"Unknown event type: {event_type}")
                return
        
        async with self._hook_lock:
            if event_type in self._event_hooks and callback in self._event_hooks[event_type]:
                self._event_hooks[event_type].remove(callback)
                logger.debug(f"Unregistered hook for event: {event_type.value}")
    
    async def _trigger_event(self, event_type: KanbanEvent, event_data: Dict[str, Any]) -> None:
        """Trigger event hooks (support sync and async callbacks)"""
        if event_type not in self._event_hooks:
            return
        
        async with self._hook_lock:
            callbacks = list(self._event_hooks[event_type])
        
        for callback in callbacks:
            try:
                if inspect.iscoroutinefunction(callback):
                    await callback(event_data)
                else:
                    callback(event_data)
            except Exception as e:
                logger.error(f"Error in event hook {event_type.value}: {e}")
    
    async def add_card(
        self,
        agent_name: str,
        card_type: Union[CardType, str],
        title: str,
        description: str = "",
        status: Union[CardStatus, str] = CardStatus.TODO,
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        card_id: Optional[str] = None,
    ) -> KanbanCard:
        """
        Add a new card to the board.
        
        Args:
            agent_name: Name of the agent
            card_type: Type of card 
            title: Card title
            description: Card description
            status: Initial status
            step: Associated step number
            metadata: Additional metadata (should be lightweight!)
            card_id: Optional card_id (auto-generated if not provided)
        """
        if isinstance(card_type, str):
            card_type = CardType(card_type)
        if isinstance(status, str):
            status = CardStatus(status)
        
        if card_id is None:
            import uuid
            card_id = f"{agent_name}_{card_type.value}_{uuid.uuid4().hex[:12]}"
        
        if card_id in self._cards:
            raise ValueError(f"Card ID '{card_id}' already exists")
        
        card = KanbanCard(
            card_id=card_id,
            agent_name=agent_name,
            card_type=card_type,
            status=status,
            title=title,
            description=description,
            step=step,
            metadata=metadata or {},
        )
        
        self._cards[card_id] = card
        self._add_card_to_column(card)
        
        await self._trigger_event(KanbanEvent.ON_CARD_ADDED, {
            "card": card.to_dict(),
            "card_id": card_id,
        })
        
        logger.debug(f"Kanban: Added card {card_id}")
        return card
    
    def _add_card_to_column(self, card: KanbanCard) -> None:
        """Add card to appropriate column."""
        column = self._columns.get(card.status)
        if column:
            column.add_card(card)
    
    async def update_card_status(
        self,
        card_id: str,
        new_status: Union[CardStatus, str]
    ) -> bool:
        """Update card status and move between columns."""
        if isinstance(new_status, str):
            new_status = CardStatus(new_status)
        
        card = self._cards.get(card_id)
        if not card:
            return False
        
        old_status = card.status
        
        # Remove from old column
        old_column = self._columns.get(old_status)
        if old_column:
            old_column.remove_card(card_id)
        
        # Update status
        card.update_status(new_status)
        
        # Add to new column
        self._add_card_to_column(card)
        
        await self._trigger_event(KanbanEvent.ON_CARD_UPDATED, {
            "card": card.to_dict(),
            "card_id": card_id,
            "old_status": old_status.value,
            "new_status": new_status.value,
        })
        
        logger.debug(f"Kanban: Updated card {card_id} to status {new_status.value}")
        return True
    
    def get_card(self, card_id: str) -> Optional[KanbanCard]:
        """Get a card by ID."""
        return self._cards.get(card_id)
    
    def get_cards_by_agent(self, agent_name: str) -> List[KanbanCard]:
        """Get all cards for a specific agent."""
        return [card for card in self._cards.values() if card.agent_name == agent_name]
    
    def get_cards_by_status(self, status: str) -> List[KanbanCard]:
        """Get all cards with a specific status."""
        return [card for card in self._cards.values() if card.status == status]
    
    def get_cards_by_type(self, card_type: Union[CardType, str]) -> List[KanbanCard]:
        """Get all cards of a specific type."""
        if isinstance(card_type, str):
            card_type = CardType(card_type)
        return [card for card in self._cards.values() if card.card_type == card_type]
    
    def set_coordinator(self, coordinator: Any) -> None:
        """Set Coordinator for dynamic agent memory access"""
        self._coordinator = coordinator
        logger.debug("Kanban: Coordinator set")
    
    @property
    def global_step(self) -> int:
        """Get global step number"""
        return self._global_step
    
    def set_global_step(self, step: int) -> None:
        self._global_step = step
        logger.debug(f"Kanban: Global step set to {step}")
    
    def get_agent_memory(self, agent_name: str) -> Optional[Any]:
        """Get agent memory (requires coordinator)"""
        if self._coordinator:
            agent = self._coordinator.get_agent(agent_name)
            if agent:
                return agent.memory
        return None
    
    def _extract_card_params_from_step(
        self, 
        step_data: Dict[str, Any], 
        agent_name: str = "GroundingAgent"
    ) -> Dict[str, Any]:
        """Extract parameters for creating a card from step data"""
        step_num = step_data.get("step")
        backend = step_data.get("backend", "unknown")
        tool = step_data.get("tool", "unknown")
        result = step_data.get("result", {})
        status_str = result.get("status", "unknown")
        
        return {
            "agent_name": agent_name,
            "card_type": CardType.EXECUTION,
            "status": CardStatus.DONE if status_str == "success" else CardStatus.BLOCKED,
            "title": f"{backend}.{tool}",
            "description": f"Step {step_num}: {step_data.get('command', '')}",
            "step": step_num,
            "metadata": step_data,
        }
    
    async def update_from_trajectory(
        self,
        trajectory_path: str,
        agent_name: str = "GroundingAgent"
    ) -> bool:
        """
        Update Kanban from trajectory record (for offline analysis).
        """
        try:
            if os.path.isdir(trajectory_path):
                jsonl_path = os.path.join(trajectory_path, "traj.jsonl")
            else:
                jsonl_path = trajectory_path
            
            if not os.path.exists(jsonl_path):
                logger.warning(f"Trajectory file not found: {jsonl_path}")
                return False
            
            trajectory = load_trajectory_from_jsonl(jsonl_path)
            if not trajectory:
                return False
            
            for step_data in trajectory:
                card_params = self._extract_card_params_from_step(step_data, agent_name)
                await self.add_card(**card_params)
            
            analysis = analyze_trajectory(trajectory)
            logger.info(
                f"Kanban: Loaded {analysis['total_steps']} steps from trajectory "
                f"(success rate: {analysis['success_rate']:.1%})"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Kanban: Failed to update from trajectory: {e}")
            return False
    
    def get_summary(self) -> str:
        """Get a summary of the Kanban board state."""
        from fastagent.utils.display import Box, BoxStyle
        
        box = Box(width=62, style=BoxStyle.ROUNDED, color='bl')
        lines = [
            box.top_line(0),
            box.text_line("Kanban Board Summary", align='center', indent=0, text_color='c'),
            box.separator_line(0),
            box.text_line(f"Total cards: {len(self._cards)}", indent=0),
            box.separator_line(0),
        ]
        
        for col_name, column in self._columns.items():
            count = column.count()
            lines.append(box.text_line(f"{column.name}: {count} cards", indent=0))
            
            cards = column.get_cards()[-3:]
            for card in cards:
                max_title_len = 50
                title = card.title[:max_title_len] + "..." if len(card.title) > max_title_len else card.title
                lines.append(box.text_line(f"  - [{card.agent_name}] {title}", indent=0))
        
        lines.append(box.separator_line(0))
        
        agents = set(card.agent_name for card in self._cards.values())
        lines.append(box.text_line("Activity by agent:", indent=0, text_color='c'))
        for agent in agents:
            agent_cards = self.get_cards_by_agent(agent)
            lines.append(box.text_line(f"  {agent}: {len(agent_cards)} cards", indent=0))
        
        lines.append(box.bottom_line(0))
        return "\n".join(lines)
    
    def get_blocked_details(self) -> List[Dict[str, Any]]:
        """Get detailed information about BLOCKED cards."""
        blocked_cards = self.get_cards_by_status(CardStatus.BLOCKED)
        details = []
        
        for card in blocked_cards:
            card_info = {
                "card_id": card.card_id,
                "card_type": card.card_type.value,
                "title": card.title,
                "description": card.description,
                "agent_name": card.agent_name,
                "created_at": card.created_at,
                "updated_at": card.updated_at,
                "step": card.step,
                "error": card.metadata.get("error", "Unknown error"),
            }
            
            if hasattr(card, 'result') and card.result:
                result = card.result
                if isinstance(result, dict):
                    card_info["result"] = {
                        "status": result.get("status", "unknown"),
                        "response": result.get("response", ""),
                        "iterations": result.get("iterations", 0),
                    }
                    
                    tool_executions = result.get("tool_executions", [])
                    if tool_executions:
                        card_info["tool_executions"] = [
                            {
                                "tool_name": exec_info.get("tool_name", "unknown"),
                                "backend": exec_info.get("backend", "unknown"),
                                "status": exec_info.get("status", "unknown"),
                                "error": exec_info.get("error"),
                            }
                            for exec_info in tool_executions[-5:]
                        ]
            
            planning_card_id = card.metadata.get("planning_card_id")
            if planning_card_id:
                card_info["planning_card_id"] = planning_card_id
                
                related_cards = [
                    c for c in self._cards.values()
                    if c.metadata.get("planning_card_id") == planning_card_id
                    and c.card_id != card.card_id
                ]
                
                card_info["related_cards"] = [
                    {
                        "card_id": c.card_id,
                        "title": c.title,
                        "status": c.status.value,
                        "card_type": c.card_type.value,
                    }
                    for c in related_cards
                ]
                
                planning_card = self.get_card(planning_card_id)
                if planning_card:
                    card_info["original_task"] = planning_card.description or planning_card.title
            
            details.append(card_info)
        
        return details
    
    def get_agent_summary(self, agent_name: str) -> str:
        """Get summary for a specific agent."""
        cards = self.get_cards_by_agent(agent_name)
        if not cards:
            return f"No cards found for agent {agent_name}"
        
        from fastagent.utils.display import Box, BoxStyle
        
        box = Box(width=66, style=BoxStyle.ROUNDED, color='bl')
        lines = [
            box.top_line(0),
            box.text_line(f"{agent_name} Summary", align='center', indent=0, text_color='c'),
            box.separator_line(0),
            box.text_line(f"Total cards: {len(cards)}", indent=0),
            box.separator_line(0),
        ]
        
        by_status = {}
        for card in cards:
            if card.status not in by_status:
                by_status[card.status] = []
            by_status[card.status].append(card)
        
        for status, status_cards in by_status.items():
            lines.append(box.text_line(f"{status.upper()}: {len(status_cards)}", indent=0))
            for card in status_cards[-5:]:
                max_title_len = 54
                title = card.title[:max_title_len] + "..." if len(card.title) > max_title_len else card.title
                lines.append(box.text_line(f"  - {title}", indent=0))
        
        lines.append(box.bottom_line(0))
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Kanban board to dictionary."""
        return {
            "cards": {card_id: card.to_dict() for card_id, card in self._cards.items()},
            "global_step": self._global_step,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load from dictionary."""
        self._cards.clear()
        for col in self._columns.values():
            col.clear()
        
        self._global_step = data.get("global_step", 0)
        
        for card_id, card_data in data.get("cards", {}).items():
            card = KanbanCard.from_dict(card_data)
            self._cards[card_id] = card
            self._add_card_to_column(card)
    
    async def on_step_recorded(self, step_data: Dict[str, Any]) -> None:
        """
        Process recorded steps (updates global step count only).
        
        Separation of duties:
        - Kanban: High-level task management
        - Recording: Low-level execution record
        """
        step_num = step_data.get("step")
        if step_num is not None and step_num > self._global_step:
            self._global_step = step_num
        
        await self._trigger_event(KanbanEvent.ON_STEP_RECORDED, {
            "step_data": step_data,
        })
        
        logger.debug(f"Kanban: Processed recorded step {step_num}")
    
    async def delete_card(self, card_id: str) -> bool:
        """Delete a card from the board."""
        card = self._cards.get(card_id)
        if not card:
            return False
        
        column = self._columns.get(card.status)
        if column:
            column.remove_card(card_id)
        
        del self._cards[card_id]
        
        await self._trigger_event(KanbanEvent.ON_CARD_DELETED, {
            "card_id": card_id,
            "card": card.to_dict(),
        })
        
        logger.debug(f"Kanban: Deleted card {card_id}")
        return True
    
    def clear(self) -> None:
        """Clear all data from the board."""
        self._cards.clear()
        self._global_step = 0
        for column in self._columns.values():
            column.clear()
        logger.info("Kanban board cleared")