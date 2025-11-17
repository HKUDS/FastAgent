"""
Unified Content Processor Module
Responsible for unified content management across Memory, Context, and Response:

Design Philosophy:
1. Memory (Agent-Local): Short-term working memory, keep recent detailed information for LLM use
2. Context (Task-Level): Task knowledge accumulator, provide dependency information for subsequent steps
3. Response (User-Facing): User-friendly replies, polished by LLM
"""

from __future__ import annotations

import json
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from enum import Enum
from dataclasses import dataclass, field

from fastagent.utils.logging import Logger

if TYPE_CHECKING:
    from fastagent.llm import LLMClient

logger = Logger.get_logger(__name__)


class ContentType(Enum):
    """Content type classification"""
    # Operation type: only need to keep operation description and status, not detailed content
    OPERATION = "operation"  # e.g., click, input, scroll, GUI operations
    
    # Data type: need to keep detailed content, this is key knowledge
    DATA_RETRIEVAL = "data_retrieval"  # e.g., web search, read file, API calls
    
    # Verification type: only need to keep success/failure status
    VERIFICATION = "verification"  # e.g., check, verify, test
    
    # Transformation type: need to keep input/output
    TRANSFORMATION = "transformation"  # e.g., format, convert, generate


class ContentLevel(Enum):
    """Content detail level"""
    FULL = "full"  # Full content (for Memory and data type content)
    SUMMARY = "summary"  # Summary level (for Context passing)
    MINIMAL = "minimal"  # Minimal level (for operation type content)


class ImportanceLevel(Enum):
    """Content importance level (for fine-grained filtering)"""
    CRITICAL = "critical"      # Critical data, keep fully
    IMPORTANT = "important"    # Important content, keep main parts
    NORMAL = "normal"          # Normal content, keep summary
    MINIMAL = "minimal"        # Minimal content, keep status only


@dataclass
class ProcessedContent:
    content_type: ContentType
    level: ContentLevel
    
    # Basic information
    title: str
    status: str  # success/error/partial
    
    # Content data
    structured_data: Optional[Dict[str, Any]] = None  # Structured extracted data 
    entities: Dict[str, List[Any]] = field(default_factory=dict)  # Extracted entities 
    
    # Different level content representations
    full_data: Optional[Dict[str, Any]] = None  # Full raw data
    summary: str = ""  # Human-readable summary
    operations: List[str] = field(default_factory=list)  # Operation list
    
    # Metadata
    source_card_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self, level: Optional[ContentLevel] = None) -> Dict[str, Any]:
        use_level = level or self.level
        
        result = {
            "title": self.title,
            "status": self.status,
            "content_type": self.content_type.value,
            "level": use_level.value,
        }
        
        if self.structured_data:
            result["structured_data"] = self.structured_data
        if self.entities:
            result["entities"] = self.entities
        
        # Level-specific content
        if use_level == ContentLevel.FULL:
            # Full level: return all information
            result["full_data"] = self.full_data
            result["summary"] = self.summary
            result["operations"] = self.operations
            
        elif use_level == ContentLevel.SUMMARY:
            # Summary level: summary + optional data based on content type
            result["summary"] = self.summary
            if self.content_type == ContentType.DATA_RETRIEVAL:
                # Data type: also include full_data
                result["full_data"] = self.full_data
            elif self.content_type == ContentType.OPERATION:
                # Operation type: summary only
                if not result["summary"]:
                    result["summary"] = f"Executed {len(self.operations)} operations"
                
        elif use_level == ContentLevel.MINIMAL:
            # Minimal level: summary/title only
            result["summary"] = self.summary or self.title
        
        if self.source_card_id:
            result["source_card_id"] = self.source_card_id
            
        return result


class ContentProcessor:
    def __init__(
        self, 
        llm_client: Optional[LLMClient] = None,
        small_size_threshold: int = 1000,      # < 1KB
        large_size_threshold: int = 100000,    # < 100KB
    ):
        """
        Initialize content processor
        
        Args:
            llm_client: LLM client for generating summaries and polish
            small_size_threshold: Small content threshold (bytes)
            large_size_threshold: Large content threshold (bytes)
        """
        self.llm_client = llm_client
        self.small_size_threshold = small_size_threshold
        self.large_size_threshold = large_size_threshold
        logger.info("ContentProcessor initialized")
    
    def infer_content_type(
        self,
        title: str,
        metadata: Dict[str, Any]
    ) -> ContentType:
        """
        Infer content type
        Mainly based on backend and actual content characteristics
        """
        backend = metadata.get("backend", "").lower()
        
        # Basic judgment based on backend
        if backend == "gui":
            return ContentType.OPERATION
        elif backend in ["web", "mcp"]:
            return ContentType.DATA_RETRIEVAL
        elif backend == "shell":
            # shell needs to check specific tool
            tool = metadata.get("tool", "").lower()
            # Simple rule: file reading is data type, others are operation/transformation type
            if "read" in tool or "cat" in tool or "query" in tool:
                return ContentType.DATA_RETRIEVAL
            else:
                return ContentType.TRANSFORMATION
        else:
            # Default to operation type
            return ContentType.OPERATION
    
    def infer_importance_for_tool(
        self,
        backend: str,
        tool_name: str,
        content: Any,
        content_size: int
    ) -> ImportanceLevel:
        """
        Infer importance of a single tool execution (fine-grained)
        
        Judgment logic:
        1. GUI backend -> MINIMAL (operation type)
        2. Contains structured data -> IMPORTANT/CRITICAL
        3. Large content -> downgrade based on type
        4. Others -> NORMAL
        """
        backend_lower = backend.lower()
        tool_lower = tool_name.lower()
        
        # GUI operations are always minimal
        if backend_lower == "gui":
            return ImportanceLevel.MINIMAL
        
        # Check if contains structured data
        has_structured_data = isinstance(content, (dict, list)) or (
            isinstance(content, str) and self._looks_like_json(content)
        )
        
        if has_structured_data:
            # Judge structured data based on size
            if content_size < self.small_size_threshold:
                return ImportanceLevel.CRITICAL  # Small structured data
            elif content_size < self.large_size_threshold:
                return ImportanceLevel.IMPORTANT  # Medium structured data
            else:
                return ImportanceLevel.IMPORTANT  # Large structured data also keeps IMPORTANT
        
        # Web/MCP backend usually data retrieval
        if backend_lower in ["web", "mcp"]:
            return ImportanceLevel.IMPORTANT
        
        # Other cases
        if content_size < self.small_size_threshold:
            return ImportanceLevel.NORMAL
        else:
            return ImportanceLevel.NORMAL
    
    def _looks_like_json(self, content: str) -> bool:
        if not isinstance(content, str):
            return False
        stripped = content.strip()
        return (stripped.startswith('{') and stripped.endswith('}')) or \
               (stripped.startswith('[') and stripped.endswith(']'))
    
    def filter_execution_result(
        self,
        result: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Filter execution result (fine-grained, at tool_execution level)
        """
        tool_executions = result.get("tool_executions", [])
        
        if not tool_executions:
            return {
                "filtered_result": result,
                "filter_stats": {
                    "original_size": self._estimate_size(result),
                    "filtered_size": self._estimate_size(result),
                    "reduction_ratio": 0.0
                }
            }
        
        # Filter at tool_execution level
        filtered_tool_executions = []
        original_total_size = 0
        filtered_total_size = 0
        
        for tool_exec in tool_executions:
            tool_name = tool_exec.get("tool_name", "unknown")
            backend = tool_exec.get("backend", "unknown")
            content = tool_exec.get("content")
            
            # Calculate original size
            original_size = self._estimate_size(content)
            original_total_size += original_size
            
            # Judge importance
            importance = self.infer_importance_for_tool(
                backend=backend,
                tool_name=tool_name,
                content=content,
                content_size=original_size
            )
            
            # Filter content based on importance
            filtered_content = self._filter_content_by_importance(
                content=content,
                importance=importance,
                original_size=original_size
            )
            
            # Calculate filtered size
            filtered_size = self._estimate_size(filtered_content)
            filtered_total_size += filtered_size
            
            # Save filtered tool execution
            filtered_tool_exec = {
                **tool_exec,
                "content": filtered_content,
                "_filter_metadata": {
                    "importance": importance.value,
                    "original_size": original_size,
                    "filtered_size": filtered_size
                }
            }
            filtered_tool_executions.append(filtered_tool_exec)
        
        # Build complete filtered result
        filtered_result = {
            **result,
            "tool_executions": filtered_tool_executions
        }
        
        # Statistics
        filter_stats = {
            "original_size": original_total_size,
            "filtered_size": filtered_total_size,
            "reduction_ratio": 1 - (filtered_total_size / original_total_size) if original_total_size > 0 else 0
        }
        
        logger.debug(
            f"Filtering completed: {len(tool_executions)} tools, "
            f"{original_total_size} -> {filtered_total_size} bytes "
            f"({filter_stats['reduction_ratio']:.1%} reduction)"
        )
        
        return {
            "filtered_result": filtered_result,
            "filter_stats": filter_stats
        }
    
    def _filter_content_by_importance(
        self,
        content: Any,
        importance: ImportanceLevel,
        original_size: int
    ) -> Any:
        """Filter content based on importance"""
        if importance == ImportanceLevel.CRITICAL:
            # Critical content: keep fully
            return content
        
        elif importance == ImportanceLevel.MINIMAL:
            # Minimal content: keep brief description only
            if isinstance(content, str):
                return content[:100] if len(content) > 100 else content
            else:
                return f"<{type(content).__name__}>"
        
        elif importance == ImportanceLevel.IMPORTANT:
            # Important content: intelligent truncation
            if original_size > self.large_size_threshold:
                # Large content, keep 10%
                return self._truncate_content(content, keep_ratio=0.1)
            else:
                # Medium/small content, keep fully
                return content
        
        else:  # NORMAL
            # Normal content: decide based on size
            if original_size < self.small_size_threshold:
                return content
            elif original_size < self.large_size_threshold:
                return self._truncate_content(content, keep_ratio=0.5)
            else:
                return self._truncate_content(content, keep_ratio=0.1)
    
    def _truncate_content(self, content: Any, keep_ratio: float) -> Any:
        """Intelligently truncate content"""
        if isinstance(content, str):
            target_length = int(len(content) * keep_ratio)
            if target_length < len(content):
                return content[:target_length] + f"...(truncated)"
            return content
        
        elif isinstance(content, dict):
            if keep_ratio >= 0.5:
                return content
            else:
                keys = list(content.keys())
                keep_count = max(1, int(len(keys) * keep_ratio))
                return {k: content[k] for k in keys[:keep_count]}
        
        elif isinstance(content, list):
            keep_count = max(1, int(len(content) * keep_ratio))
            return content[:keep_count]
        
        else:
            return content
    
    def _estimate_size(self, content: Any) -> int:
        """Estimate content size (bytes)"""
        if content is None:
            return 0
        elif isinstance(content, str):
            return len(content.encode('utf-8'))
        elif isinstance(content, (dict, list)):
            try:
                return len(json.dumps(content, ensure_ascii=False).encode('utf-8'))
            except:
                return len(str(content).encode('utf-8'))
        else:
            return len(str(content).encode('utf-8'))
    
    def process_execution_result(
        self,
        card_id: str,
        title: str,
        result: Dict[str, Any],
        metadata: Dict[str, Any],
        target_level: ContentLevel = ContentLevel.SUMMARY
    ) -> ProcessedContent:
        # Infer content type
        content_type = self.infer_content_type(title, metadata)
        
        # Extract status
        status = result.get("status", "unknown")
        
        # Extract operation list
        operations = []
        tool_executions = result.get("tool_executions", [])
        for exec_info in tool_executions:
            tool_name = exec_info.get("tool_name", "unknown")
            backend = exec_info.get("backend", "unknown")
            operations.append(f"{backend}.{tool_name}")
        
        extracted_knowledge = result.get("extracted_knowledge", {})
        
        summary = extracted_knowledge.get("what_was_done", "")
        if not summary:
            # Fallback to old summary generation
            summary = self._generate_summary(
                content_type=content_type,
                title=title,
                result=result,
                operations=operations,
                data=None
            )
        
        full_data = None
        structured_data = None
        if extracted_knowledge.get("has_artifact"):
            artifact = extracted_knowledge.get("artifact", {})
            full_data = {
                "type": artifact.get("type", "unknown"),
                "content": artifact.get("content"),
                "description": artifact.get("description", ""),
                "metadata": artifact.get("metadata", {})
            }
            
            structured_data = artifact.get("metadata", {})
        
        entities = {}
        if tool_executions:
            for exec_info in tool_executions:
                if exec_info.get("status") == "success":
                    exec_metadata = exec_info.get("metadata", {})
                    
                    # Extract entities
                    if "entities" in exec_metadata:
                        exec_entities = exec_metadata["entities"]
                        for entity_type, entity_list in exec_entities.items():
                            if entity_type not in entities:
                                entities[entity_type] = []
                            # Deduplicate and merge
                            for entity in entity_list:
                                if entity not in entities[entity_type]:
                                    entities[entity_type].append(entity)
        
        return ProcessedContent(
            content_type=content_type,
            level=target_level,
            title=title,
            status=status,
            structured_data=structured_data,
            entities=entities,
            full_data=full_data,
            summary=summary,
            operations=operations,
            source_card_id=card_id,
            metadata=metadata
        )
    
    def _generate_summary(
        self,
        content_type: ContentType,
        title: str,
        result: Dict[str, Any],
        operations: List[str],
        data: Optional[Dict[str, Any]]
    ) -> str:
        """
        Generate appropriate summary based on content type
        """
        status = result.get("status", "unknown")
        
        if content_type == ContentType.OPERATION:
            # Operation type: describe what operations were performed
            if operations:
                return f"Executed {len(operations)} operations: {', '.join(operations[:3])}"
            else:
                return f"Completed operation: {title}"
                
        elif content_type == ContentType.DATA_RETRIEVAL:
            # Data type: describe what data was retrieved
            if data and "sources" in data:
                source_count = data.get("count", 0)
                return f"Retrieved {source_count} data items: {title}"
            else:
                return f"Data retrieval: {title}"
                
        elif content_type == ContentType.VERIFICATION:
            # Verification type: describe verification result
            return f"Verification result: {status}"
            
        elif content_type == ContentType.TRANSFORMATION:
            # Transformation type: describe transformation content
            return f"Transformation completed: {title}"
            
        return title
    
    async def polish_response(
        self,
        executions: List[ProcessedContent],
        original_task: str,
        execution_plan: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Polish Response content using LLM
        
        Args:
            executions: List of processed execution content
            original_task: Original task description
            execution_plan: Execution plan (optional)
        """
        if not self.llm_client:
            # No LLM, return simple summary
            return self._simple_response_summary(executions, original_task)
        
        # Prepare input data
        execution_summaries = []
        important_data = []
        
        for i, exec_content in enumerate(executions, 1):
            exec_dict = exec_content.to_dict(ContentLevel.SUMMARY)
            execution_summaries.append(f"{i}. {exec_dict.get('title')}: {exec_dict.get('summary', 'N/A')}")
            
            # Collect important data
            if exec_content.content_type == ContentType.DATA_RETRIEVAL and exec_content.full_data:
                important_data.append({
                    "source": exec_content.title,
                    "data": exec_content.full_data
                })
        
        # Build prompt
        messages = [
            {
                "role": "system",
                "content": """You are a professional task result summarizer. Your role is to integrate results from multiple execution steps into a clear, friendly, and informative user response.

Requirements:
1. Answer the user's original question in natural, friendly language
2. Data-type content (web search, file reading, etc.): Display key data completely
3. Operation-type content (clicking, typing, etc.): Briefly explain what operations were completed
4. If there are multiple steps, organize with clear structure
5. Highlight key findings and results
6. If there are errors, state them clearly

Output format:
{
    "answer": "Natural language answer for the user",
    "key_findings": ["Key finding 1", "Key finding 2"],
    "data_summary": "Brief summary of data (if any)",
    "operations_performed": "Summary of operations performed (if any)"
}"""
            },
            {
                "role": "user",
                "content": f"""Original task: {original_task}

Execution steps:
{chr(10).join(execution_summaries)}

Important data:
{json.dumps(important_data, ensure_ascii=False, indent=2) if important_data else "None"}

Please integrate the above information and generate a clear, complete user response."""
            }
        ]
        
        try:
            response = await self.llm_client.complete(messages=messages)
            
            # Extract content
            if isinstance(response, dict):
                content = response.get("content", "")
                if isinstance(content, list):
                    text_parts = [
                        block.get("text", "") for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                    content_str = "\n".join(text_parts)
                else:
                    content_str = str(content)
            else:
                content_str = str(response)
            
            # Try to parse JSON
            try:
                polished = json.loads(content_str)
            except json.JSONDecodeError:
                # If not JSON, use entire content as answer
                polished = {"answer": content_str}
            
            # Add original data reference
            polished["important_data"] = important_data
            polished["execution_count"] = len(executions)
            
            logger.info(f"Successfully polished response with {len(important_data)} important data items")
            return polished
            
        except Exception as e:
            logger.error(f"Polish response failed: {e}")
            return self._simple_response_summary(executions, original_task)
    
    def _simple_response_summary(
        self,
        executions: List[ProcessedContent],
        original_task: str
    ) -> Dict[str, Any]:
        """
        Simple response summary (fallback)
        """
        lines = [f"Task: {original_task}", "", "Execution results:"]
        
        important_data = []
        operations = []
        
        for i, exec_content in enumerate(executions, 1):
            if exec_content.content_type == ContentType.DATA_RETRIEVAL:
                lines.append(f"{i}. ✓ {exec_content.title}")
                if exec_content.full_data:
                    important_data.append({
                        "source": exec_content.title,
                        "data": exec_content.full_data
                    })
            else:
                operations.append(exec_content.summary or exec_content.title)
                lines.append(f"{i}. ✓ {exec_content.summary}")
        
        return {
            "answer": "\n".join(lines),
            "important_data": important_data,
            "operations_performed": operations,
            "execution_count": len(executions)
        }
    
    def compress_for_memory(
        self,
        executions: List[ProcessedContent],
        max_items: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Compress content for Memory
        
        Strategy:
        1. Data-type content: keep most recent max_items
        2. Operation-type content: merge into summary
        """
        data_items = []
        operation_items = []
        other_items = []
        
        for exec_content in executions:
            if exec_content.content_type == ContentType.DATA_RETRIEVAL:
                data_items.append(exec_content.to_dict(ContentLevel.SUMMARY))
            elif exec_content.content_type == ContentType.OPERATION:
                operation_items.append(exec_content.to_dict(ContentLevel.MINIMAL))
            else:
                other_items.append(exec_content.to_dict(ContentLevel.SUMMARY))
        
        # Keep most recent data-type content
        compressed = data_items[-max_items:] if data_items else []
        
        # If there are operation-type contents, add a summary
        if operation_items:
            compressed.append({
                "title": "Operations Summary",
                "summary": f"Executed {len(operation_items)} operations",
                "content_type": ContentType.OPERATION.value,
                "level": ContentLevel.MINIMAL.value
            })
        
        # Add other types
        compressed.extend(other_items[-max_items:])
        
        return compressed