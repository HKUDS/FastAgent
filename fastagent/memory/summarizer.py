from __future__ import annotations

import json
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from fastagent.memory.memory import Memory, MemoryItem
from fastagent.utils.logging import Logger

if TYPE_CHECKING:
    from fastagent.llm import LLMClient

logger = Logger.get_logger(__name__)


class MemorySummarizer:
    def __init__(
        self,
        llm_client: Optional["LLMClient"] = None,
        max_memory_items: int = 100,
        compression_threshold: int = 50,
        summary_window: int = 10,
    ):
        """
        Initialize memory summarizer
        
        Args:
            llm_client: LLM client, for generating summaries
            max_memory_items: Maximum number of memory items to retain
            compression_threshold: Threshold for triggering compression
            summary_window: Summary window size (how many memory items to summarize at once)
        """
        self.llm_client = llm_client
        self.max_memory_items = max_memory_items
        self.compression_threshold = compression_threshold
        self.summary_window = summary_window
        self._summaries: List[Dict[str, Any]] = []
    
    async def compress_memory(
        self,
        memory: Memory,
        force: bool = False
    ) -> Dict[str, Any]:
        if not force and memory.length < self.compression_threshold:
            logger.debug(f"Memory size ({memory.length}) below threshold, skipping compression")
            return {
                "compressed": False,
                "original_size": memory.length,
                "new_size": memory.length,
            }
        
        if not self.llm_client:
            logger.warning("No LLM client available for compression")
            return {
                "compressed": False,
                "error": "No LLM client",
                "original_size": memory.length,
                "new_size": memory.length,
            }
        
        try:
            items_to_compress = memory.length - self.max_memory_items + self.summary_window
            if items_to_compress <= 0:
                return {
                    "compressed": False,
                    "original_size": memory.length,
                    "new_size": memory.length,
                }
            
            old_items = memory.content[:items_to_compress]
            
            summary = await self._generate_summary(old_items)
            
            self._summaries.append({
                "timestamp": old_items[-1].get_value("timestamp") if old_items else None,
                "step_range": (
                    old_items[0].get_value("step"),
                    old_items[-1].get_value("step")
                ),
                "summary": summary,
                "compressed_items": len(old_items),
            })
            
            memory._content = memory.content[items_to_compress:]
            
            logger.info(f"Compressed {items_to_compress} memory items into summary")
            
            return {
                "compressed": True,
                "original_size": memory.length + items_to_compress,
                "new_size": memory.length,
                "compressed_items": items_to_compress,
                "summary": summary,
            }
            
        except Exception as e:
            logger.error(f"Failed to compress memory: {e}")
            return {
                "compressed": False,
                "error": str(e),
                "original_size": memory.length,
                "new_size": memory.length,
            }
    
    async def _generate_summary(self, items: List[MemoryItem]) -> str:
        if not items:
            return ""
        
        # Format items as clean JSON (no double-encoding thanks to improved to_dict())
        items_text = "\n".join([
            f"Step {item.get_value('step')}: {json.dumps(item.to_dict(), ensure_ascii=False, indent=2)}"
            for item in items
        ])
        
        messages = [
            {
                "role": "system",
                "content": """You are an expert Agent Memory Compressor. Your task is to compress agent execution history into structured, information-dense summaries that preserve critical context for future decision-making.

COMPRESSION REQUIREMENTS:
1. **Preserve Key Decisions**: Record important decisions the agent made and their rationale
2. **Preserve State Changes**: Track critical changes in system state, task status, and context
3. **Preserve Causal Relationships**: Document actions and their outcomes, especially failures and errors
4. **Preserve Tool Usage**: Record which tools/methods were used and their effectiveness
5. **Extract Patterns**: Identify recurring patterns, successful strategies, or learned lessons
6. **Maintain Temporal Order**: Keep chronological sequence of events
7. **Distinguish Roles**: Separate user requests, agent reasoning, tool execution, and system feedback

OUTPUT FORMAT: Use concise but information-dense language. Organize into clear sections."""
            },
            {
                "role": "user",
                "content": f"""Compress the following agent execution history into a structured summary:

{items_text}

Structure your summary as follows:
1. **Goal & Context**: What was the agent trying to accomplish during this period?
2. **Key Actions**: What critical operations were performed (tool calls, decision points)?
3. **Results & State Changes**: What outcomes were achieved? How did the state evolve?
4. **Problems & Solutions**: What issues were encountered? How were they resolved?
5. **Lessons & Patterns**: What notable patterns or insights emerged?

Use bullet points. Be concise but preserve critical details that future reasoning will need."""
            }
        ]
        
        try:
            response = await self.llm_client.complete(messages=messages)
            
            if isinstance(response, dict):
                content = response.get("content", "")
                if isinstance(content, list):
                    text_parts = [
                        block.get("text", "") for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                    return "\n".join(text_parts)
                return str(content)
            
            return str(response)
            
        except Exception as e:
            logger.error(f"Failed to generate summary via LLM: {e}")
            return f"Compressed {len(items)} operations from step {items[0].get_value('step')} to {items[-1].get_value('step')}"
    
    async def get_work_summary(
        self,
        memory: Memory,
        include_summaries: bool = True,
        recent_items: int = 20,
    ) -> str:
        
        from fastagent.utils.display import Box, BoxStyle
        
        box = Box(width=66, style=BoxStyle.ROUNDED, color='bl')
        lines = [
            box.top_line(0),
            box.text_line("Work Summary", align='center', indent=0, text_color='c'),
            box.separator_line(0),
        ]
        
        # Historical Summary
        if include_summaries and self._summaries:
            lines.append(box.text_line("Historical Summary", indent=0, text_color='c'))
            for i, summary_info in enumerate(self._summaries, 1):
                step_range = summary_info.get("step_range", (0, 0))
                lines.append(box.text_line(f"Summary {i} (Step {step_range[0]}-{step_range[1]})", indent=0))
                summary_text = summary_info.get("summary", "No summary")
                # Split long summary into multiple lines
                for line in summary_text.split('\n'):
                    if line.strip():
                        lines.append(box.text_line(f"  {line.strip()}", indent=0))
            lines.append(box.separator_line(0))
        
        # Recent Activity
        lines.append(box.text_line("Recent Activity", indent=0, text_color='c'))
        recent = memory.content[-recent_items:] if memory.length > 0 else []
        
        if not recent:
            lines.append(box.text_line("No recent activity records", indent=0))
        else:
            for item in recent:
                step = item.get_value("step")
                role = item.get_value("role", "unknown")
                content = item.get_value("content", "")
                
                # Handle dict/list content
                if isinstance(content, (dict, list)):
                    content_str = json.dumps(content, ensure_ascii=False)
                else:
                    content_str = str(content)
                
                # Limit content length
                if len(content_str) > 60:
                    content_str = content_str[:60] + "..."
                
                lines.append(box.text_line(f"Step {step} [{role}]: {content_str}", indent=0))
        
        lines.append(box.separator_line(0))
        lines.append(box.text_line(f"Total steps: {memory.length + sum(s.get('compressed_items', 0) for s in self._summaries)}", indent=0))
        lines.append(box.text_line(f"Current memory items: {memory.length}", indent=0))
        lines.append(box.text_line(f"Compressed summaries: {len(self._summaries)}", indent=0))
        lines.append(box.bottom_line(0))
        
        return "\n".join(lines)
    
    def get_summaries(self) -> List[Dict[str, Any]]:
        return self._summaries.copy()
    
    def clear_summaries(self) -> None:
        self._summaries.clear()
        logger.debug("Cleared all summaries")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summaries": self._summaries,
            "config": {
                "max_memory_items": self.max_memory_items,
                "compression_threshold": self.compression_threshold,
                "summary_window": self.summary_window,
            }
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        self._summaries = data.get("summaries", [])
        config = data.get("config", {})
        self.max_memory_items = config.get("max_memory_items", self.max_memory_items)
        self.compression_threshold = config.get("compression_threshold", self.compression_threshold)
        self.summary_window = config.get("summary_window", self.summary_window)