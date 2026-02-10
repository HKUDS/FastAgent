"""
Data types for tool quality tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar, Dict, List, Optional, Any


@dataclass
class ExecutionRecord:
    """Single execution record."""
    timestamp: datetime
    success: bool
    execution_time_ms: float
    error_message: Optional[str] = None


@dataclass
class DescriptionQuality:
    """LLM-evaluated description quality."""
    clarity: float  # 0-1: Is the purpose and usage clear?
    completeness: float  # 0-1: Are inputs/outputs documented?
    evaluated_at: datetime
    reasoning: str = ""  # LLM's reasoning for the scores
    
    @property
    def overall_score(self) -> float:
        """Computed overall score (average of all dimensions)."""
        return (self.clarity + self.completeness) / 2


@dataclass
class ToolQualityRecord:
    """
    Complete quality record for a tool.
    
    Key: "{backend}:{server}:{tool_name}"
    """
    tool_key: str
    backend: str
    server: str
    tool_name: str
    
    # Execution stats
    total_calls: int = 0
    success_count: int = 0
    total_execution_time_ms: float = 0.0
    
    # Recent execution history (rolling window)
    recent_executions: List[ExecutionRecord] = field(default_factory=list)
    
    # Description quality (LLM-evaluated)
    description_quality: Optional[DescriptionQuality] = None
    
    # Metadata
    description_hash: Optional[str] = None
    first_seen: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Keep only recent N executions
    MAX_RECENT_EXECUTIONS: ClassVar[int] = 100
    
    # Penalty threshold: only penalize tools with success rate below this value
    # Tools with success rate >= this threshold get penalty = 1.0 (no penalty)
    PENALTY_THRESHOLD: ClassVar[float] = 0.4
    
    @property
    def success_rate(self) -> float:
        """Overall success rate."""
        if self.total_calls == 0:
            return 0.0
        return self.success_count / self.total_calls
    
    @property
    def avg_execution_time_ms(self) -> float:
        """Average execution time."""
        if self.total_calls == 0:
            return 0.0
        return self.total_execution_time_ms / self.total_calls
    
    @property
    def recent_success_rate(self) -> float:
        """Success rate from recent executions."""
        if not self.recent_executions:
            return self.success_rate
        successes = sum(1 for e in self.recent_executions if e.success)
        return successes / len(self.recent_executions)
    
    @property
    def consecutive_failures(self) -> int:
        """Count consecutive failures from the most recent execution."""
        count = 0
        for exec_record in reversed(self.recent_executions):
            if not exec_record.success:
                count += 1
            else:
                break
        return count
    
    @property
    def penalty(self) -> float:
        """
        Compute penalty factor based on failure rate.
        
        Design principles:
        - Only penalize tools with success rate < PENALTY_THRESHOLD (default 40%)
        - New tools (< 3 calls) get no penalty to allow fair evaluation
        
        Returns value between 0.2-1.0:
        - 1.0: No penalty (success rate >= threshold or insufficient data)
        - 0.2: Maximum penalty (consistently failing tool)
        """
        if self.total_calls < 3:
            return 1.0
        
        success_rate = self.recent_success_rate
        threshold = self.PENALTY_THRESHOLD
        
        if success_rate >= threshold:
            return 1.0
        
        # Linear mapping: penalty = 0.3 + (success_rate / threshold) * 0.7
        base_penalty = 0.3 + (success_rate / threshold) * 0.7
        
        # Extra penalty for consecutive failures (indicates systematic issues)
        consec = self.consecutive_failures
        if consec >= 3:
            # 3 consecutive → extra 0.1, 5 consecutive → extra 0.3
            extra_penalty = min(0.3, (consec - 2) * 0.1)
            base_penalty -= extra_penalty
        
        # Clamp to [0.2, 1.0]
        return max(0.2, min(1.0, base_penalty))
    
    @property
    def quality_score(self) -> float:
        """
        Legacy quality score for backward compatibility.
        Now delegates to penalty property.
        """
        return self.penalty
    
    def add_execution(self, record: ExecutionRecord) -> None:
        """Add execution record and update stats."""
        self.total_calls += 1
        self.total_execution_time_ms += record.execution_time_ms
        
        if record.success:
            self.success_count += 1
        
        self.recent_executions.append(record)
        
        # Trim to max size
        if len(self.recent_executions) > self.MAX_RECENT_EXECUTIONS:
            self.recent_executions = self.recent_executions[-self.MAX_RECENT_EXECUTIONS:]
        
        self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for persistence."""
        return {
            "tool_key": self.tool_key,
            "backend": self.backend,
            "server": self.server,
            "tool_name": self.tool_name,
            "total_calls": self.total_calls,
            "success_count": self.success_count,
            "total_execution_time_ms": self.total_execution_time_ms,
            "recent_executions": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "success": e.success,
                    "execution_time_ms": e.execution_time_ms,
                    "error_message": e.error_message,
                }
                for e in self.recent_executions
            ],
            "description_quality": {
                "clarity": self.description_quality.clarity,
                "completeness": self.description_quality.completeness,
                "evaluated_at": self.description_quality.evaluated_at.isoformat(),
                "reasoning": self.description_quality.reasoning,
            } if self.description_quality else None,
            "description_hash": self.description_hash,
            "first_seen": self.first_seen.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolQualityRecord":
        """Deserialize from dict."""
        record = cls(
            tool_key=data["tool_key"],
            backend=data["backend"],
            server=data["server"],
            tool_name=data["tool_name"],
            total_calls=data.get("total_calls", 0),
            success_count=data.get("success_count", 0),
            total_execution_time_ms=data.get("total_execution_time_ms", 0.0),
            description_hash=data.get("description_hash"),
            first_seen=datetime.fromisoformat(data["first_seen"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
        )
        
        # Parse recent executions
        for e in data.get("recent_executions", []):
            record.recent_executions.append(ExecutionRecord(
                timestamp=datetime.fromisoformat(e["timestamp"]),
                success=e["success"],
                execution_time_ms=e["execution_time_ms"],
                error_message=e.get("error_message"),
            ))
        
        # Parse description quality
        dq = data.get("description_quality")
        if dq:
            record.description_quality = DescriptionQuality(
                clarity=dq.get("clarity", 0.5),  # Fallback for old data
                completeness=dq.get("completeness", 0.5),
                evaluated_at=datetime.fromisoformat(dq["evaluated_at"]),
                reasoning=dq.get("reasoning", ""),  # Optional field
            )
        
        return record

