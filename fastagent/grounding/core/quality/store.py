"""
Persistent storage for tool quality data.
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional

from .types import ToolQualityRecord
from fastagent.utils.logging import Logger
from fastagent.config.constants import PROJECT_ROOT

logger = Logger.get_logger(__name__)


class QualityStore:
    """
    Persistent storage for tool quality records.
    
    Storage structure:
    <project_root>/.fastagent/tool_quality/
    ├── records.json          # All quality records
    └── records_backup.json   # Backup on save
    """
    
    VERSION = 1
    
    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            cache_dir = PROJECT_ROOT / ".fastagent" / "tool_quality"
        
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._records_file = self._cache_dir / "records.json"
        self._backup_file = self._cache_dir / "records_backup.json"
        
        self._write_lock = asyncio.Lock()
        
        logger.debug(f"QualityStore initialized at {self._cache_dir}")
    
    def load_all(self) -> tuple[Dict[str, ToolQualityRecord], int]:
        """Load all quality records and global execution count from disk.
        
        Returns:
            Tuple of (records_dict, global_execution_count)
        """
        if not self._records_file.exists():
            return {}, 0
        
        try:
            with open(self._records_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Version check
            if data.get("version") != self.VERSION:
                logger.warning(f"Cache version mismatch, clearing cache")
                return {}, 0
            
            records = {}
            for key, record_data in data.get("records", {}).items():
                try:
                    records[key] = ToolQualityRecord.from_dict(record_data)
                except Exception as e:
                    logger.warning(f"Failed to load record {key}: {e}")
            
            global_count = data.get("global_execution_count", 0)
            logger.info(f"Loaded {len(records)} quality records from cache (global_count={global_count})")
            return records, global_count
            
        except Exception as e:
            logger.error(f"Failed to load quality cache: {e}")
            return {}, 0
    
    async def save_all(self, records: Dict[str, ToolQualityRecord], global_execution_count: int = 0) -> None:
        """Save all quality records and global execution count to disk."""
        async with self._write_lock:
            try:
                # Backup existing file
                if self._records_file.exists():
                    import shutil
                    shutil.copy(self._records_file, self._backup_file)
                
                data = {
                    "version": self.VERSION,
                    "global_execution_count": global_execution_count,
                    "records": {
                        key: record.to_dict()
                        for key, record in records.items()
                    }
                }
                
                with open(self._records_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                logger.debug(f"Saved {len(records)} quality records to cache (global_count={global_execution_count})")
                
            except Exception as e:
                logger.error(f"Failed to save quality cache: {e}")
    
    async def save_record(self, record: ToolQualityRecord, all_records: Dict[str, ToolQualityRecord], global_execution_count: int = 0) -> None:
        """Save a single record (saves all for simplicity)."""
        all_records[record.tool_key] = record
        await self.save_all(all_records, global_execution_count)
    
    def clear(self) -> None:
        """Clear all cached data."""
        if self._records_file.exists():
            self._records_file.unlink()
        if self._backup_file.exists():
            self._backup_file.unlink()
        logger.info("Quality cache cleared")

