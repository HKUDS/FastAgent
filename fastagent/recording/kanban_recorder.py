"""
Kanban Event Recorder
Records kanban board events (card additions, updates, status changes).
"""

import datetime
import json
from typing import Any, Dict, Optional
from pathlib import Path

from fastagent.utils.logging import Logger

logger = Logger.get_logger(__name__)


class KanbanEventRecorder:
    """
    Records kanban board events.
    
    This recorder captures task management activities:
    - Card additions
    - Card updates
    - Status changes
    - Card deletions
    """
    
    def __init__(self, trajectory_dir: Path):
        """
        Initialize kanban event recorder.
        
        Args:
            trajectory_dir: Directory to save event records
        """
        self.trajectory_dir = trajectory_dir
        self.events_file = trajectory_dir / "kanban_events.jsonl"
        self.event_counter = 0
        
        # Ensure directory exists
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        
        # Track previous card states for change detection
        self._card_states: Dict[str, Dict[str, Any]] = {}
    
    async def record_event(
        self,
        event_type: str,
        agent_name: str,
        card_id: str,
        card_data: Dict[str, Any],
        changes: Optional[Dict[str, Any]] = None,
        related_action_step: Optional[int] = None,
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Record a kanban event.
        
        Args:
            event_type: Type of event (card_added | card_updated | card_deleted | status_changed)
            agent_name: Name of the agent that triggered the event
            card_id: ID of the affected card
            card_data: Current card data
            changes: Dictionary of changes (for update events)
            related_action_step: Agent action step that triggered this event (for correlation)
            correlation_id: Optional correlation ID to link related events
        """
        self.event_counter += 1
        timestamp = datetime.datetime.now().isoformat()
        
        # Infer agent type
        agent_type = self._infer_agent_type(agent_name)
        
        event_info = {
            "event_id": self.event_counter,
            "timestamp": timestamp,
            "event_type": event_type,
            "agent_name": agent_name,
            "agent_type": agent_type, 
            "card_id": card_id,
            "correlation_id": correlation_id or f"kanban_{self.event_counter}_{timestamp}",  
        }
        
        # Add card info (essential fields only)
        event_info["card"] = {
            "title": card_data.get("title", ""),
            "card_type": card_data.get("card_type", ""),
            "status": card_data.get("status", ""),
            "priority": card_data.get("priority", "medium"),
        }
        
        # Add description only if not too long
        if card_data.get("description"):
            desc = card_data["description"]
            if len(desc) <= 200:
                event_info["card"]["description"] = desc
            else:
                event_info["card"]["description"] = desc[:200] + "... [truncated]"
        
        # Add changes if provided
        if changes:
            event_info["changes"] = changes
        elif event_type == "card_updated":
            # Automatically detect changes if not provided
            event_info["changes"] = self._detect_changes(card_id, card_data)
        
        # Add related action step for correlation
        if related_action_step:
            event_info["related_action_step"] = related_action_step
        
        # Update card state tracking
        self._card_states[card_id] = card_data.copy()
        
        # Append to JSONL file
        await self._append_to_file(event_info)
        
        logger.debug(
            f"Recorded kanban event: {event_type} for card {card_id} by {agent_name}"
        )
        
        return event_info
    
    def _infer_agent_type(self, agent_name: str) -> str:
        name_lower = agent_name.lower()
        
        if "host" in name_lower:
            return "host"
        elif "grounding" in name_lower:
            return "grounding"
        elif "eval" in name_lower:
            return "eval"
        elif "coordinator" in name_lower:
            return "coordinator"
        elif "system" in name_lower:
            return "system"
        else:
            return "unknown"
    
    def _detect_changes(
        self,
        card_id: str,
        new_card_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect changes between previous and current card state.
        """
        changes = {}
        
        if card_id not in self._card_states:
            return changes
        
        old_card = self._card_states[card_id]
        
        # Check key fields for changes
        for field in ["status", "priority", "card_type", "title", "description"]:
            old_value = old_card.get(field)
            new_value = new_card_data.get(field)
            
            if old_value != new_value:
                changes[field] = {
                    "from": old_value,
                    "to": new_value
                }
        
        return changes
    
    async def _append_to_file(self, event_info: Dict[str, Any]):
        """Append event to JSONL file."""
        with open(self.events_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event_info, ensure_ascii=False))
            f.write("\n")
    
    def get_event_count(self) -> int:
        """Get current event count."""
        return self.event_counter
    
    async def record_snapshot(self, kanban_state: Dict[str, Any]):
        """
        Record a full kanban snapshot.
        """
        snapshots_dir = self.trajectory_dir / "kanban_snapshots"
        snapshots_dir.mkdir(exist_ok=True)
        
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        snapshot_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "state": kanban_state
        }
        
        # Save timestamped snapshot
        snapshot_file = snapshots_dir / f"snapshot_{timestamp_str}.json"
        with open(snapshot_file, 'w', encoding='utf-8') as f:
            json.dump(snapshot_data, f, indent=2, ensure_ascii=False)
        
        # Also save latest snapshot (overwrite)
        latest_file = snapshots_dir / "latest_snapshot.json"
        with open(latest_file, 'w', encoding='utf-8') as f:
            json.dump(snapshot_data, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved kanban snapshot: {snapshot_file.name}")


def load_kanban_events(trajectory_dir: str) -> list:
    events_file = Path(trajectory_dir) / "kanban_events.jsonl"
    
    if not events_file.exists():
        logger.warning(f"Kanban events file not found: {events_file}")
        return []
    
    events = []
    try:
        with open(events_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        
        logger.info(f"Loaded {len(events)} kanban events from {events_file}")
        return events
    
    except Exception as e:
        logger.error(f"Failed to load kanban events from {events_file}: {e}")
        return []


def analyze_kanban_events(events: list) -> Dict[str, Any]:
    if not events:
        return {
            "total_events": 0,
            "by_type": {},
            "by_agent": {},
            "cards_affected": 0,
        }
    
    # Count by type and agent
    by_type = {}
    by_agent = {}
    affected_cards = set()
    
    for event in events:
        event_type = event.get("event_type", "unknown")
        agent_name = event.get("agent_name", "unknown")
        card_id = event.get("card_id", "")
        
        by_type[event_type] = by_type.get(event_type, 0) + 1
        by_agent[agent_name] = by_agent.get(agent_name, 0) + 1
        
        if card_id:
            affected_cards.add(card_id)
    
    return {
        "total_events": len(events),
        "by_type": by_type,
        "by_agent": by_agent,
        "cards_affected": len(affected_cards),
    }


def format_kanban_events(events: list, format_type: str = "compact") -> str:
    if not events:
        return "No kanban events recorded"
    
    if format_type == "compact":
        lines = []
        for event in events:
            event_id = event.get("event_id", "?")
            event_type = event.get("event_type", "?")
            agent = event.get("agent_name", "?")
            card_title = event.get("card", {}).get("title", "?")
            
            lines.append(f"Event {event_id}: [{agent}] {event_type} - {card_title}")
        
        return "\n".join(lines)
    
    elif format_type == "detailed":
        lines = []
        for event in events:
            lines.append(f"\n{'='*60}")
            lines.append(f"Event {event.get('event_id', '?')}: {event.get('event_type', '?')}")
            lines.append(f"Agent: {event.get('agent_name', '?')}")
            lines.append(f"Time: {event.get('timestamp', '?')}")
            lines.append(f"Card ID: {event.get('card_id', '?')}")
            
            if event.get("card"):
                lines.append("\nCard:")
                lines.append(json.dumps(event["card"], indent=2, ensure_ascii=False))
            
            if event.get("changes"):
                lines.append("\nChanges:")
                lines.append(json.dumps(event["changes"], indent=2, ensure_ascii=False))
        
        return "\n".join(lines)
    
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def reconstruct_kanban_timeline(events: list) -> Dict[str, list]:
    timeline = {}
    
    for event in events:
        card_id = event.get("card_id", "")
        if not card_id:
            continue
        
        if card_id not in timeline:
            timeline[card_id] = []
        
        timeline[card_id].append(event)
    
    return timeline