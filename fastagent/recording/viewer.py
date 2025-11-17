"""
Recording Viewer
Convenient tools for viewing and analyzing recording sessions.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastagent.utils.logging import Logger
from .utils import load_recording_session, generate_summary_report
from .action_recorder import load_agent_actions, analyze_agent_actions, format_agent_actions
from .kanban_recorder import load_kanban_events, analyze_kanban_events, format_kanban_events, reconstruct_kanban_timeline

logger = Logger.get_logger(__name__)


class RecordingViewer:
    """
    Viewer for analyzing recording sessions.
    
    Provides convenient methods to:
    - Load and display recordings
    - Analyze agent behaviors
    - Track kanban changes
    - Generate reports
    """
    
    def __init__(self, recording_dir: str):
        """
        Initialize viewer with a recording directory.
        
        Args:
            recording_dir: Path to recording directory
        """
        self.recording_dir = Path(recording_dir)
        
        if not self.recording_dir.exists():
            raise ValueError(f"Recording directory not found: {recording_dir}")
        
        # Load session data
        self.session = load_recording_session(str(self.recording_dir))
        
        logger.info(f"Loaded recording from {recording_dir}")
    
    def show_summary(self) -> str:
        """
        Display a summary of the recording.
        
        Returns:
            Formatted summary string
        """
        if not self.session.get("metadata"):
            return "No metadata available"
        
        metadata = self.session["metadata"]
        stats = self.session.get("statistics", {})
        
        lines = []
        lines.append("=" * 70)
        lines.append("RECORDING SUMMARY")
        lines.append("=" * 70)
        lines.append(f"Task ID: {metadata.get('task_id', 'N/A')}")
        lines.append(f"Start: {metadata.get('start_time', 'N/A')}")
        lines.append(f"End: {metadata.get('end_time', 'N/A')}")
        lines.append(f"Total Steps: {metadata.get('total_steps', 0)}")
        lines.append("")
        
        lines.append("Statistics:")
        lines.append(f"  - Success Rate: {stats.get('success_rate', 0):.2%}")
        lines.append(f"  - Success Count: {stats.get('success_count', 0)}/{stats.get('total_steps', 0)}")
        lines.append("")
        
        if stats.get("backends"):
            lines.append("Backend Usage:")
            for backend, count in sorted(stats["backends"].items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  - {backend}: {count}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def show_agent_actions(self, format_type: str = "compact", agent_name: Optional[str] = None) -> str:
        actions = load_agent_actions(str(self.recording_dir))
        
        if agent_name:
            actions = [a for a in actions if a.get("agent_name") == agent_name]
        
        if not actions:
            return f"No agent actions found{' for ' + agent_name if agent_name else ''}"
        
        # Add header
        header = f"\nAGENT ACTIONS ({len(actions)} total)"
        if agent_name:
            header += f" - {agent_name}"
        header += "\n" + "=" * 70
        
        # Format actions
        formatted = format_agent_actions(actions, format_type)
        
        return header + "\n" + formatted
    
    def show_kanban_events(self, format_type: str = "compact", event_type: Optional[str] = None) -> str:
        events = load_kanban_events(str(self.recording_dir))
        
        if event_type:
            events = [e for e in events if e.get("event_type") == event_type]
        
        if not events:
            return f"No kanban events found{' for type ' + event_type if event_type else ''}"
        
        # Add header
        header = f"\nKANBAN EVENTS ({len(events)} total)"
        if event_type:
            header += f" - {event_type}"
        header += "\n" + "=" * 70
        
        # Format events
        formatted = format_kanban_events(events, format_type)
        
        return header + "\n" + formatted
    
    def show_card_timeline(self, card_id: str) -> str:
        events = load_kanban_events(str(self.recording_dir))
        timeline = reconstruct_kanban_timeline(events)
        
        if card_id not in timeline:
            return f"No events found for card {card_id}"
        
        card_events = timeline[card_id]
        
        lines = []
        lines.append(f"\nTIMELINE FOR CARD: {card_id}")
        lines.append("=" * 70)
        
        for event in card_events:
            lines.append(f"\n[{event.get('timestamp', 'N/A')}]")
            lines.append(f"Event: {event.get('event_type', 'unknown')}")
            lines.append(f"Agent: {event.get('agent_name', 'unknown')}")
            
            if event.get('card'):
                card = event['card']
                lines.append(f"Title: {card.get('title', 'N/A')}")
                lines.append(f"Status: {card.get('status', 'N/A')}")
            
            if event.get('changes'):
                lines.append("Changes:")
                for field, change in event['changes'].items():
                    lines.append(f"  {field}: {change.get('from')} → {change.get('to')}")
        
        return "\n".join(lines)
    
    def analyze_agents(self) -> str:
        actions = load_agent_actions(str(self.recording_dir))
        stats = analyze_agent_actions(actions)
        
        lines = []
        lines.append("\nAGENT ANALYSIS")
        lines.append("=" * 70)
        lines.append(f"Total Actions: {stats.get('total_actions', 0)}")
        lines.append("")
        
        lines.append("By Agent:")
        for agent, count in sorted(stats.get('by_agent', {}).items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats['total_actions'] * 100) if stats['total_actions'] > 0 else 0
            lines.append(f"  - {agent}: {count} ({percentage:.1f}%)")
        lines.append("")
        
        lines.append("By Action Type:")
        for action_type, count in sorted(stats.get('by_type', {}).items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats['total_actions'] * 100) if stats['total_actions'] > 0 else 0
            lines.append(f"  - {action_type}: {count} ({percentage:.1f}%)")
        
        return "\n".join(lines)
    
    def analyze_kanban(self) -> str:
        events = load_kanban_events(str(self.recording_dir))
        stats = analyze_kanban_events(events)
        
        lines = []
        lines.append("\nKANBAN ANALYSIS")
        lines.append("=" * 70)
        lines.append(f"Total Events: {stats.get('total_events', 0)}")
        lines.append(f"Cards Affected: {stats.get('cards_affected', 0)}")
        lines.append("")
        
        lines.append("By Event Type:")
        for event_type, count in sorted(stats.get('by_type', {}).items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats['total_events'] * 100) if stats['total_events'] > 0 else 0
            lines.append(f"  - {event_type}: {count} ({percentage:.1f}%)")
        lines.append("")
        
        lines.append("By Agent:")
        for agent, count in sorted(stats.get('by_agent', {}).items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats['total_events'] * 100) if stats['total_events'] > 0 else 0
            lines.append(f"  - {agent}: {count} ({percentage:.1f}%)")
        
        return "\n".join(lines)
    
    def generate_full_report(self, output_file: Optional[str] = None) -> str:
        return generate_summary_report(str(self.recording_dir), output_file)
    
    def export_to_json(self, output_file: str):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.session, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported session to {output_file}")
    
    def show_timeline(self, max_events: int = 50) -> str:
        # Load all events
        actions = load_agent_actions(str(self.recording_dir))
        events = load_kanban_events(str(self.recording_dir))
        trajectory = self.session.get("trajectory", [])
        
        # Combine all events with unified format
        timeline = []
        
        # Add agent actions
        for action in actions:
            timeline.append({
                "timestamp": action.get("timestamp", ""),
                "type": "agent_action",
                "agent_name": action.get("agent_name", ""),
                "agent_type": action.get("agent_type", "unknown"),
                "action_type": action.get("action_type", ""),
                "step": action.get("step"),
                "correlation_id": action.get("correlation_id", ""),
                "description": f"[{action.get('agent_type', '?').upper()}] {action.get('action_type', '?')}",
                "related_tool_steps": action.get("related_tool_steps", []),
                "related_kanban_events": action.get("related_kanban_events", []),
            })
        
        # Add kanban events
        for event in events:
            timeline.append({
                "timestamp": event.get("timestamp", ""),
                "type": "kanban_event",
                "agent_name": event.get("agent_name", ""),
                "agent_type": event.get("agent_type", "unknown"),
                "event_type": event.get("event_type", ""),
                "event_id": event.get("event_id"),
                "card_id": event.get("card_id", ""),
                "correlation_id": event.get("correlation_id", ""),
                "description": f"[KANBAN] {event.get('event_type', '?')} - {event.get('card', {}).get('title', '?')}",
                "related_action_step": event.get("related_action_step"),
            })
        
        # Add tool executions
        for traj_step in trajectory:
            timeline.append({
                "timestamp": traj_step.get("timestamp", ""),
                "type": "tool_execution",
                "backend": traj_step.get("backend", ""),
                "tool": traj_step.get("tool", ""),
                "step": traj_step.get("step"),
                "agent_name": traj_step.get("agent_name", ""),
                "description": f"[TOOL:{traj_step.get('backend', '?').upper()}] {traj_step.get('tool', '?')}",
                "status": traj_step.get("result", {}).get("status", ""),
            })
        
        # Sort by timestamp
        timeline.sort(key=lambda x: x.get("timestamp", ""))
        
        # Format output
        lines = []
        lines.append("\nUNIFIED TIMELINE")
        lines.append("=" * 100)
        lines.append(f"Total events: {len(timeline)} (showing first {max_events})")
        lines.append("")
        
        for i, item in enumerate(timeline[:max_events]):
            timestamp = item.get("timestamp", "N/A")
            time_str = timestamp.split("T")[1][:8] if "T" in timestamp else timestamp[-8:]
            
            # Format line with type indicator
            type_marker = {
                "agent_action": "🤖",
                "kanban_event": "📋",
                "tool_execution": "🔧"
            }.get(item.get("type"), "•")
            
            desc = item.get("description", "")
            agent = item.get("agent_name", "")
            agent_type = item.get("agent_type", "")
            
            line = f"{time_str} {type_marker} {desc}"
            
            # Add agent info if available
            if agent and agent_type:
                line += f" (by {agent}/{agent_type})"
            elif agent:
                line += f" (by {agent})"
            
            lines.append(line)
            
            # Show correlations
            correlations = []
            if item.get("related_tool_steps"):
                correlations.append(f"→ tool steps: {item['related_tool_steps']}")
            if item.get("related_kanban_events"):
                correlations.append(f"→ kanban events: {item['related_kanban_events']}")
            if item.get("related_action_step"):
                correlations.append(f"→ action step: {item['related_action_step']}")
            
            if correlations:
                for corr in correlations:
                    lines.append(f"         {corr}")
        
        if len(timeline) > max_events:
            lines.append(f"\n... and {len(timeline) - max_events} more events")
        
        return "\n".join(lines)
    
    def show_agent_flow(self, agent_name: Optional[str] = None) -> str:
        """
        Show the flow of a specific agent's actions and related events.
        """
        actions = load_agent_actions(str(self.recording_dir))
        events = load_kanban_events(str(self.recording_dir))
        
        if agent_name:
            actions = [a for a in actions if a.get("agent_name") == agent_name]
            events = [e for e in events if e.get("agent_name") == agent_name]
        
        lines = []
        lines.append(f"\nAGENT FLOW{' - ' + agent_name if agent_name else ''}")
        lines.append("=" * 100)
        
        # Combine and sort by timestamp
        all_items = []
        for action in actions:
            all_items.append(("action", action))
        for event in events:
            all_items.append(("event", event))
        
        all_items.sort(key=lambda x: x[1].get("timestamp", ""))
        
        for item_type, item in all_items:
            timestamp = item.get("timestamp", "N/A").split("T")[1][:8] if "T" in item.get("timestamp", "") else "N/A"
            
            if item_type == "action":
                agent_type = item.get("agent_type", "?").upper()
                action_type = item.get("action_type", "?")
                step = item.get("step", "?")
                lines.append(f"{timestamp} [{agent_type}] Action #{step}: {action_type}")
                
                # Show reasoning if available
                if item.get("reasoning"):
                    thought = item["reasoning"].get("thought", "")
                    if thought:
                        lines.append(f"         💭 {thought[:80]}...")
                
                # Show output
                if item.get("output"):
                    output = item["output"]
                    if isinstance(output, dict):
                        for key in ["message", "status", "evaluation"]:
                            if key in output:
                                lines.append(f"         📤 {key}: {str(output[key])[:60]}")
                
            else:  # kanban event
                event_type = item.get("event_type", "?")
                card_title = item.get("card", {}).get("title", "?")
                lines.append(f"{timestamp} [KANBAN] {event_type}: {card_title}")
                
                # Show changes
                if item.get("changes"):
                    for field, change in item["changes"].items():
                        lines.append(f"         🔄 {field}: {change.get('from')} → {change.get('to')}")
            
            lines.append("")
        
        return "\n".join(lines)


def view_recording(recording_dir: str):
    """
    Quick interactive viewer for a recording.
    """
    try:
        viewer = RecordingViewer(recording_dir)
        
        print(viewer.show_summary())
        print("\n")
        
        print(viewer.analyze_agents())
        print("\n")
        
        print(viewer.analyze_kanban())
        print("\n")
        
        print("Agent Actions (compact):")
        print(viewer.show_agent_actions(format_type="compact"))
        print("\n")
        
        print("Kanban Events (compact):")
        print(viewer.show_kanban_events(format_type="compact"))
        
    except Exception as e:
        logger.error(f"Failed to view recording: {e}")
        print(f"Error: {e}")


def compare_recordings(recording_dir1: str, recording_dir2: str) -> str:
    """
    Compare two recordings side by side.
    """
    try:
        viewer1 = RecordingViewer(recording_dir1)
        viewer2 = RecordingViewer(recording_dir2)
        
        lines = []
        lines.append("=" * 70)
        lines.append("RECORDING COMPARISON")
        lines.append("=" * 70)
        lines.append("")
        
        # Compare metadata
        meta1 = viewer1.session.get("metadata", {})
        meta2 = viewer2.session.get("metadata", {})
        
        lines.append("Recording 1:")
        lines.append(f"  Task: {meta1.get('task_id', 'N/A')}")
        lines.append(f"  Steps: {meta1.get('total_steps', 0)}")
        lines.append("")
        
        lines.append("Recording 2:")
        lines.append(f"  Task: {meta2.get('task_id', 'N/A')}")
        lines.append(f"  Steps: {meta2.get('total_steps', 0)}")
        lines.append("")
        
        # Compare statistics
        stats1 = viewer1.session.get("statistics", {})
        stats2 = viewer2.session.get("statistics", {})
        
        lines.append("Differences:")
        lines.append(f"  Steps: {meta2.get('total_steps', 0) - meta1.get('total_steps', 0):+d}")
        lines.append(f"  Success Rate: {stats2.get('success_rate', 0) - stats1.get('success_rate', 0):+.2%}")
        
        return "\n".join(lines)
        
    except Exception as e:
        logger.error(f"Failed to compare recordings: {e}")
        return f"Error: {e}"


# CLI interface
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m fastagent.recording.viewer <recording_dir>")
        sys.exit(1)
    
    recording_dir = sys.argv[1]
    view_recording(recording_dir)