#!/usr/bin/env python3
"""
📦 ARTIFACT MANAGER 📦
Comprehensive artifact management system for evaluation results

This module implements the ArtifactManager class for evaluation result storage and versioning,
episode-level data export in CSV/JSON formats, video and trace file management with compression,
evaluation history tracking and champion progression, and utilities for artifact cleanup and archival.
"""

import os
import sys
import json
import gzip
import shutil
import hashlib
import sqlite3
import zipfile
import tarfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging
import pandas as pd
import numpy as np
from enum import Enum
import threading
import tempfile

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

class ArtifactType(Enum):
    """Types of artifacts managed by the system."""
    EVALUATION_RESULT = "evaluation_result"
    EPISODE_DATA = "episode_data"
    VIDEO = "video"
    TRACE = "trace"
    REPORT = "report"
    MODEL_CHECKPOINT = "model_checkpoint"
    CONFIGURATION = "configuration"
    LOG = "log"
    PLOT = "plot"
    HEATMAP = "heatmap"

class CompressionType(Enum):
    """Compression types for artifacts."""
    NONE = "none"
    GZIP = "gzip"
    ZIP = "zip"
    TAR_GZ = "tar.gz"

class StoragePolicy(Enum):
    """Storage policies for artifact retention."""
    KEEP_ALL = "keep_all"
    KEEP_TOP_K = "keep_top_k"
    KEEP_RECENT = "keep_recent"
    KEEP_CHAMPIONS = "keep_champions"

@dataclass
class ArtifactMetadata:
    """Metadata for stored artifacts."""
    artifact_id: str
    artifact_type: ArtifactType
    file_path: str
    original_size: int
    compressed_size: Optional[int] = None
    compression_type: CompressionType = CompressionType.NONE
    checksum: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    model_id: Optional[str] = None
    evaluation_id: Optional[str] = None
    suite_name: Optional[str] = None
    map_name: Optional[str] = None
    episode_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationHistory:
    """History entry for evaluation runs."""
    evaluation_id: str
    model_id: str
    timestamp: str
    global_score: float
    success_rate: float
    is_champion: bool = False
    champion_rank: Optional[int] = None
    artifacts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChampionProgression:
    """Champion progression tracking."""
    champion_id: str
    model_id: str
    timestamp: str
    global_score: float
    success_rate: float
    previous_champion: Optional[str] = None
    improvement_metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)

@dataclass
class ArtifactManagerConfig:
    """Configuration for artifact manager."""
    base_path: str = "artifacts"
    database_path: str = "artifacts/artifact_registry.db"
    compression_enabled: bool = True
    default_compression: CompressionType = CompressionType.GZIP
    storage_policy: StoragePolicy = StoragePolicy.KEEP_TOP_K
    max_artifacts_per_type: int = 100
    max_age_days: int = 30
    auto_cleanup_enabled: bool = True
    backup_enabled: bool = True
    backup_interval_hours: int = 24

class ArtifactManager:
    """
    Comprehensive artifact management system for evaluation results.
    
    Provides functionality for:
    - Evaluation result storage and versioning
    - Episode-level data export in CSV/JSON formats
    - Video and trace file management with compression
    - Evaluation history tracking and champion progression
    - Utilities for artifact cleanup and archival
    """
    
    def __init__(self, config: Optional[ArtifactManagerConfig] = None):
        """Initialize artifact manager."""
        self.config = config or ArtifactManagerConfig()
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        
        # Create base directory structure
        self.base_path = Path(self.config.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.subdirs = {
            ArtifactType.EVALUATION_RESULT: self.base_path / "evaluations",
            ArtifactType.EPISODE_DATA: self.base_path / "episodes",
            ArtifactType.VIDEO: self.base_path / "videos",
            ArtifactType.TRACE: self.base_path / "traces",
            ArtifactType.REPORT: self.base_path / "reports",
            ArtifactType.MODEL_CHECKPOINT: self.base_path / "models",
            ArtifactType.CONFIGURATION: self.base_path / "configs",
            ArtifactType.LOG: self.base_path / "logs",
            ArtifactType.PLOT: self.base_path / "plots",
            ArtifactType.HEATMAP: self.base_path / "heatmaps"
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        self.logger.info(f"ArtifactManager initialized with base path: {self.base_path}")
    
    def _init_database(self):
        """Initialize SQLite database for artifact registry."""
        db_path = Path(self.config.database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(str(db_path)) as conn:
            # Artifacts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    artifact_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    original_size INTEGER NOT NULL,
                    compressed_size INTEGER,
                    compression_type TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    model_id TEXT,
                    evaluation_id TEXT,
                    suite_name TEXT,
                    map_name TEXT,
                    episode_id TEXT,
                    tags TEXT,
                    metadata TEXT
                )
            """)
            
            # Evaluation history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_history (
                    evaluation_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    global_score REAL NOT NULL,
                    success_rate REAL NOT NULL,
                    is_champion INTEGER DEFAULT 0,
                    champion_rank INTEGER,
                    artifacts TEXT,
                    metadata TEXT
                )
            """)
            
            # Champion progression table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS champion_progression (
                    champion_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    global_score REAL NOT NULL,
                    success_rate REAL NOT NULL,
                    previous_champion TEXT,
                    improvement_metrics TEXT,
                    artifacts TEXT
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(artifact_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_model ON artifacts(model_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_evaluation ON artifacts(evaluation_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_model ON evaluation_history(model_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_timestamp ON evaluation_history(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_champions_timestamp ON champion_progression(timestamp)")
            
            conn.commit()
    
    def store_artifact(self, 
                      file_path: Union[str, Path],
                      artifact_type: ArtifactType,
                      model_id: Optional[str] = None,
                      evaluation_id: Optional[str] = None,
                      suite_name: Optional[str] = None,
                      map_name: Optional[str] = None,
                      episode_id: Optional[str] = None,
                      tags: Optional[List[str]] = None,
                      metadata: Optional[Dict[str, Any]] = None,
                      compress: Optional[bool] = None) -> str:
        """
        Store an artifact with metadata and optional compression.
        
        Args:
            file_path: Path to the file to store
            artifact_type: Type of artifact
            model_id: Associated model ID
            evaluation_id: Associated evaluation ID
            suite_name: Associated test suite name
            map_name: Associated map name
            episode_id: Associated episode ID
            tags: List of tags for categorization
            metadata: Additional metadata
            compress: Whether to compress (uses config default if None)
            
        Returns:
            Artifact ID for the stored artifact
        """
        with self._lock:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Generate artifact ID
            artifact_id = self._generate_artifact_id(file_path, artifact_type, model_id, evaluation_id)
            
            # Determine compression
            should_compress = compress if compress is not None else self.config.compression_enabled
            compression_type = self.config.default_compression if should_compress else CompressionType.NONE
            
            # Calculate original file size and checksum
            original_size = file_path.stat().st_size
            checksum = self._calculate_checksum(file_path)
            
            # Determine storage path
            storage_dir = self.subdirs[artifact_type]
            if model_id:
                storage_dir = storage_dir / model_id
                storage_dir.mkdir(parents=True, exist_ok=True)
            
            # Store file with optional compression
            if should_compress:
                stored_path, compressed_size = self._store_compressed(
                    file_path, storage_dir, artifact_id, compression_type
                )
            else:
                stored_path = storage_dir / f"{artifact_id}{file_path.suffix}"
                shutil.copy2(file_path, stored_path)
                compressed_size = None
            
            # Create metadata
            artifact_metadata = ArtifactMetadata(
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                file_path=str(stored_path),
                original_size=original_size,
                compressed_size=compressed_size,
                compression_type=compression_type,
                checksum=checksum,
                model_id=model_id,
                evaluation_id=evaluation_id,
                suite_name=suite_name,
                map_name=map_name,
                episode_id=episode_id,
                tags=tags or [],
                metadata=metadata or {}
            )
            
            # Store in database
            self._store_artifact_metadata(artifact_metadata)
            
            self.logger.info(f"Stored artifact {artifact_id} at {stored_path}")
            return artifact_id
    
    def _generate_artifact_id(self, file_path: Path, artifact_type: ArtifactType, 
                             model_id: Optional[str], evaluation_id: Optional[str]) -> str:
        """Generate unique artifact ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # Include microseconds for uniqueness
        components = [
            artifact_type.value,
            timestamp,
            file_path.stem
        ]
        
        if model_id:
            components.append(model_id[:8])  # First 8 chars of model ID
        if evaluation_id:
            components.append(evaluation_id[:8])  # First 8 chars of evaluation ID
            
        return "_".join(components)
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _store_compressed(self, file_path: Path, storage_dir: Path, 
                         artifact_id: str, compression_type: CompressionType) -> Tuple[Path, int]:
        """Store file with compression."""
        if compression_type == CompressionType.GZIP:
            stored_path = storage_dir / f"{artifact_id}.gz"
            with open(file_path, 'rb') as f_in:
                with gzip.open(stored_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        elif compression_type == CompressionType.ZIP:
            stored_path = storage_dir / f"{artifact_id}.zip"
            with zipfile.ZipFile(stored_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(file_path, file_path.name)
        
        elif compression_type == CompressionType.TAR_GZ:
            stored_path = storage_dir / f"{artifact_id}.tar.gz"
            with tarfile.open(stored_path, 'w:gz') as tar:
                tar.add(file_path, arcname=file_path.name)
        
        else:
            raise ValueError(f"Unsupported compression type: {compression_type}")
        
        compressed_size = stored_path.stat().st_size
        return stored_path, compressed_size
    
    def _store_artifact_metadata(self, metadata: ArtifactMetadata):
        """Store artifact metadata in database."""
        with sqlite3.connect(self.config.database_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO artifacts 
                (artifact_id, artifact_type, file_path, original_size, compressed_size,
                 compression_type, checksum, created_at, model_id, evaluation_id,
                 suite_name, map_name, episode_id, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.artifact_id,
                metadata.artifact_type.value,
                metadata.file_path,
                metadata.original_size,
                metadata.compressed_size,
                metadata.compression_type.value,
                metadata.checksum,
                metadata.created_at,
                metadata.model_id,
                metadata.evaluation_id,
                metadata.suite_name,
                metadata.map_name,
                metadata.episode_id,
                json.dumps(metadata.tags),
                json.dumps(metadata.metadata)
            ))
            conn.commit()
    
    def export_episode_data(self, 
                           episode_results: List[Dict[str, Any]],
                           format_type: str = "both",
                           model_id: Optional[str] = None,
                           evaluation_id: Optional[str] = None) -> Dict[str, str]:
        """
        Export episode-level data in CSV/JSON formats.
        
        Args:
            episode_results: List of episode result dictionaries
            format_type: Export format ("csv", "json", or "both")
            model_id: Associated model ID
            evaluation_id: Associated evaluation ID
            
        Returns:
            Dictionary mapping format to artifact ID
        """
        if not episode_results:
            raise ValueError("No episode results provided")
        
        artifact_ids = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create temporary files for export
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            if format_type in ["json", "both"]:
                # Export as JSON
                json_file = temp_path / f"episode_data_{timestamp}.json"
                with open(json_file, 'w') as f:
                    json.dump({
                        "metadata": {
                            "export_timestamp": datetime.now().isoformat(),
                            "model_id": model_id,
                            "evaluation_id": evaluation_id,
                            "total_episodes": len(episode_results)
                        },
                        "episodes": episode_results
                    }, f, indent=2, default=str)
                
                artifact_id = self.store_artifact(
                    json_file,
                    ArtifactType.EPISODE_DATA,
                    model_id=model_id,
                    evaluation_id=evaluation_id,
                    tags=["episode_data", "json"],
                    metadata={"format": "json", "episode_count": len(episode_results)}
                )
                artifact_ids["json"] = artifact_id
            
            if format_type in ["csv", "both"]:
                # Export as CSV
                try:
                    df = pd.DataFrame(episode_results)
                    csv_file = temp_path / f"episode_data_{timestamp}.csv"
                    df.to_csv(csv_file, index=False)
                    
                    artifact_id = self.store_artifact(
                        csv_file,
                        ArtifactType.EPISODE_DATA,
                        model_id=model_id,
                        evaluation_id=evaluation_id,
                        tags=["episode_data", "csv"],
                        metadata={"format": "csv", "episode_count": len(episode_results)}
                    )
                    artifact_ids["csv"] = artifact_id
                    
                except Exception as e:
                    self.logger.warning(f"Failed to export CSV format: {e}")
                    if format_type == "csv":
                        raise
        
        self.logger.info(f"Exported episode data in {len(artifact_ids)} format(s)")
        return artifact_ids
    
    def store_video_with_compression(self, 
                                   video_path: Union[str, Path],
                                   model_id: Optional[str] = None,
                                   evaluation_id: Optional[str] = None,
                                   episode_id: Optional[str] = None,
                                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store video file with compression and metadata.
        
        Args:
            video_path: Path to video file
            model_id: Associated model ID
            evaluation_id: Associated evaluation ID
            episode_id: Associated episode ID
            metadata: Additional metadata
            
        Returns:
            Artifact ID for stored video
        """
        return self.store_artifact(
            video_path,
            ArtifactType.VIDEO,
            model_id=model_id,
            evaluation_id=evaluation_id,
            episode_id=episode_id,
            tags=["video", "failure_analysis"],
            metadata=metadata,
            compress=True
        )
    
    def store_trace_file(self, 
                        trace_path: Union[str, Path],
                        model_id: Optional[str] = None,
                        evaluation_id: Optional[str] = None,
                        episode_id: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store trace file with compression and metadata.
        
        Args:
            trace_path: Path to trace file
            model_id: Associated model ID
            evaluation_id: Associated evaluation ID
            episode_id: Associated episode ID
            metadata: Additional metadata
            
        Returns:
            Artifact ID for stored trace
        """
        return self.store_artifact(
            trace_path,
            ArtifactType.TRACE,
            model_id=model_id,
            evaluation_id=evaluation_id,
            episode_id=episode_id,
            tags=["trace", "failure_analysis"],
            metadata=metadata,
            compress=True
        )
    
    def record_evaluation_history(self, 
                                evaluation_id: str,
                                model_id: str,
                                global_score: float,
                                success_rate: float,
                                is_champion: bool = False,
                                champion_rank: Optional[int] = None,
                                artifacts: Optional[List[str]] = None,
                                metadata: Optional[Dict[str, Any]] = None):
        """
        Record evaluation history entry.
        
        Args:
            evaluation_id: Unique evaluation ID
            model_id: Model identifier
            global_score: Global composite score
            success_rate: Success rate percentage
            is_champion: Whether this is a champion model
            champion_rank: Rank if champion
            artifacts: List of associated artifact IDs
            metadata: Additional metadata
        """
        with self._lock:
            history_entry = EvaluationHistory(
                evaluation_id=evaluation_id,
                model_id=model_id,
                timestamp=datetime.now().isoformat(),
                global_score=global_score,
                success_rate=success_rate,
                is_champion=is_champion,
                champion_rank=champion_rank,
                artifacts=artifacts or [],
                metadata=metadata or {}
            )
            
            with sqlite3.connect(self.config.database_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO evaluation_history
                    (evaluation_id, model_id, timestamp, global_score, success_rate,
                     is_champion, champion_rank, artifacts, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    history_entry.evaluation_id,
                    history_entry.model_id,
                    history_entry.timestamp,
                    history_entry.global_score,
                    history_entry.success_rate,
                    int(history_entry.is_champion),
                    history_entry.champion_rank,
                    json.dumps(history_entry.artifacts),
                    json.dumps(history_entry.metadata)
                ))
                conn.commit()
            
            self.logger.info(f"Recorded evaluation history for {model_id}: {global_score:.3f}")
    
    def record_champion_progression(self, 
                                  model_id: str,
                                  global_score: float,
                                  success_rate: float,
                                  previous_champion: Optional[str] = None,
                                  improvement_metrics: Optional[Dict[str, float]] = None,
                                  artifacts: Optional[List[str]] = None) -> str:
        """
        Record champion progression entry.
        
        Args:
            model_id: New champion model ID
            global_score: Global composite score
            success_rate: Success rate percentage
            previous_champion: Previous champion ID
            improvement_metrics: Metrics showing improvement
            artifacts: Associated artifact IDs
            
        Returns:
            Champion progression ID
        """
        with self._lock:
            champion_id = f"champion_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_id[:8]}"
            
            progression = ChampionProgression(
                champion_id=champion_id,
                model_id=model_id,
                timestamp=datetime.now().isoformat(),
                global_score=global_score,
                success_rate=success_rate,
                previous_champion=previous_champion,
                improvement_metrics=improvement_metrics or {},
                artifacts=artifacts or []
            )
            
            with sqlite3.connect(self.config.database_path) as conn:
                conn.execute("""
                    INSERT INTO champion_progression
                    (champion_id, model_id, timestamp, global_score, success_rate,
                     previous_champion, improvement_metrics, artifacts)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    progression.champion_id,
                    progression.model_id,
                    progression.timestamp,
                    progression.global_score,
                    progression.success_rate,
                    progression.previous_champion,
                    json.dumps(progression.improvement_metrics),
                    json.dumps(progression.artifacts)
                ))
                conn.commit()
            
            self.logger.info(f"Recorded champion progression: {model_id} -> {champion_id}")
            return champion_id
    
    def get_evaluation_history(self, 
                             model_id: Optional[str] = None,
                             limit: Optional[int] = None,
                             champions_only: bool = False) -> List[EvaluationHistory]:
        """
        Retrieve evaluation history.
        
        Args:
            model_id: Filter by model ID
            limit: Maximum number of entries
            champions_only: Only return champion entries
            
        Returns:
            List of evaluation history entries
        """
        query = "SELECT * FROM evaluation_history"
        params = []
        conditions = []
        
        if model_id:
            conditions.append("model_id = ?")
            params.append(model_id)
        
        if champions_only:
            conditions.append("is_champion = 1")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        with sqlite3.connect(self.config.database_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        
        history = []
        for row in rows:
            history.append(EvaluationHistory(
                evaluation_id=row['evaluation_id'],
                model_id=row['model_id'],
                timestamp=row['timestamp'],
                global_score=row['global_score'],
                success_rate=row['success_rate'],
                is_champion=bool(row['is_champion']),
                champion_rank=row['champion_rank'],
                artifacts=json.loads(row['artifacts']) if row['artifacts'] else [],
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            ))
        
        return history
    
    def get_champion_progression(self, limit: Optional[int] = None) -> List[ChampionProgression]:
        """
        Retrieve champion progression history.
        
        Args:
            limit: Maximum number of entries
            
        Returns:
            List of champion progression entries
        """
        query = "SELECT * FROM champion_progression ORDER BY timestamp DESC"
        params = []
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        with sqlite3.connect(self.config.database_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        
        progression = []
        for row in rows:
            progression.append(ChampionProgression(
                champion_id=row['champion_id'],
                model_id=row['model_id'],
                timestamp=row['timestamp'],
                global_score=row['global_score'],
                success_rate=row['success_rate'],
                previous_champion=row['previous_champion'],
                improvement_metrics=json.loads(row['improvement_metrics']) if row['improvement_metrics'] else {},
                artifacts=json.loads(row['artifacts']) if row['artifacts'] else []
            ))
        
        return progression
    
    def get_artifacts(self, 
                     artifact_type: Optional[ArtifactType] = None,
                     model_id: Optional[str] = None,
                     evaluation_id: Optional[str] = None,
                     tags: Optional[List[str]] = None,
                     limit: Optional[int] = None) -> List[ArtifactMetadata]:
        """
        Retrieve artifacts matching criteria.
        
        Args:
            artifact_type: Filter by artifact type
            model_id: Filter by model ID
            evaluation_id: Filter by evaluation ID
            tags: Filter by tags (any match)
            limit: Maximum number of results
            
        Returns:
            List of artifact metadata
        """
        query = "SELECT * FROM artifacts"
        params = []
        conditions = []
        
        if artifact_type:
            conditions.append("artifact_type = ?")
            params.append(artifact_type.value)
        
        if model_id:
            conditions.append("model_id = ?")
            params.append(model_id)
        
        if evaluation_id:
            conditions.append("evaluation_id = ?")
            params.append(evaluation_id)
        
        if tags:
            tag_conditions = []
            for tag in tags:
                tag_conditions.append("tags LIKE ?")
                params.append(f'%"{tag}"%')
            conditions.append("(" + " OR ".join(tag_conditions) + ")")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY created_at DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        with sqlite3.connect(self.config.database_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        
        artifacts = []
        for row in rows:
            artifacts.append(ArtifactMetadata(
                artifact_id=row['artifact_id'],
                artifact_type=ArtifactType(row['artifact_type']),
                file_path=row['file_path'],
                original_size=row['original_size'],
                compressed_size=row['compressed_size'],
                compression_type=CompressionType(row['compression_type']),
                checksum=row['checksum'],
                created_at=row['created_at'],
                model_id=row['model_id'],
                evaluation_id=row['evaluation_id'],
                suite_name=row['suite_name'],
                map_name=row['map_name'],
                episode_id=row['episode_id'],
                tags=json.loads(row['tags']) if row['tags'] else [],
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            ))
        
        return artifacts
    
    def retrieve_artifact(self, artifact_id: str, output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Retrieve and decompress artifact.
        
        Args:
            artifact_id: Artifact identifier
            output_path: Output path (uses temp if None)
            
        Returns:
            Path to retrieved file
        """
        # Get artifact metadata
        artifacts = self.get_artifacts()
        artifact = None
        for a in artifacts:
            if a.artifact_id == artifact_id:
                artifact = a
                break
        
        if not artifact:
            raise ValueError(f"Artifact not found: {artifact_id}")
        
        stored_path = Path(artifact.file_path)
        if not stored_path.exists():
            raise FileNotFoundError(f"Artifact file not found: {stored_path}")
        
        # Determine output path
        if output_path is None:
            output_path = Path(tempfile.mkdtemp()) / f"retrieved_{artifact_id}"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Decompress if needed
        if artifact.compression_type == CompressionType.NONE:
            shutil.copy2(stored_path, output_path)
        
        elif artifact.compression_type == CompressionType.GZIP:
            with gzip.open(stored_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        elif artifact.compression_type == CompressionType.ZIP:
            with zipfile.ZipFile(stored_path, 'r') as zipf:
                # Extract first file
                names = zipf.namelist()
                if names:
                    with zipf.open(names[0]) as f_in:
                        with open(output_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
        
        elif artifact.compression_type == CompressionType.TAR_GZ:
            with tarfile.open(stored_path, 'r:gz') as tar:
                # Extract first file
                members = tar.getmembers()
                if members:
                    with tar.extractfile(members[0]) as f_in:
                        with open(output_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
        
        self.logger.info(f"Retrieved artifact {artifact_id} to {output_path}")
        return output_path
    
    def cleanup_artifacts(self, 
                         dry_run: bool = False,
                         max_age_days: Optional[int] = None,
                         keep_champions: bool = True) -> Dict[str, int]:
        """
        Clean up old artifacts based on storage policy.
        
        Args:
            dry_run: Only report what would be deleted
            max_age_days: Maximum age in days (uses config default if None)
            keep_champions: Preserve champion-related artifacts
            
        Returns:
            Dictionary with cleanup statistics
        """
        max_age = max_age_days or self.config.max_age_days
        cutoff_date = datetime.now() - timedelta(days=max_age)
        cutoff_timestamp = cutoff_date.timestamp()
        
        stats = {
            "total_artifacts": 0,
            "deleted_artifacts": 0,
            "preserved_champions": 0,
            "freed_bytes": 0
        }
        
        # Get all artifacts
        all_artifacts = self.get_artifacts()
        stats["total_artifacts"] = len(all_artifacts)
        
        # Get champion artifact IDs if preserving
        champion_artifact_ids = set()
        if keep_champions:
            champions = self.get_champion_progression()
            for champion in champions:
                champion_artifact_ids.update(champion.artifacts)
        
        # Identify artifacts to delete
        to_delete = []
        for artifact in all_artifacts:
            # Skip if champion artifact
            if keep_champions and artifact.artifact_id in champion_artifact_ids:
                stats["preserved_champions"] += 1
                continue
            
            # Check age
            artifact_timestamp = datetime.fromisoformat(artifact.created_at).timestamp()
            if artifact_timestamp < cutoff_timestamp:
                to_delete.append(artifact)
        
        # Delete artifacts
        for artifact in to_delete:
            if not dry_run:
                try:
                    # Delete file
                    file_path = Path(artifact.file_path)
                    if file_path.exists():
                        stats["freed_bytes"] += file_path.stat().st_size
                        file_path.unlink()
                    
                    # Remove from database
                    with sqlite3.connect(self.config.database_path) as conn:
                        conn.execute("DELETE FROM artifacts WHERE artifact_id = ?", 
                                   (artifact.artifact_id,))
                        conn.commit()
                    
                    stats["deleted_artifacts"] += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to delete artifact {artifact.artifact_id}: {e}")
            else:
                stats["deleted_artifacts"] += 1
                if Path(artifact.file_path).exists():
                    stats["freed_bytes"] += Path(artifact.file_path).stat().st_size
        
        action = "Would delete" if dry_run else "Deleted"
        self.logger.info(f"{action} {stats['deleted_artifacts']} artifacts, "
                        f"freed {stats['freed_bytes'] / 1024 / 1024:.1f} MB")
        
        return stats
    
    def create_archive(self, 
                      archive_path: Union[str, Path],
                      model_id: Optional[str] = None,
                      evaluation_id: Optional[str] = None,
                      artifact_types: Optional[List[ArtifactType]] = None,
                      include_metadata: bool = True) -> str:
        """
        Create archive of artifacts.
        
        Args:
            archive_path: Path for archive file
            model_id: Filter by model ID
            evaluation_id: Filter by evaluation ID
            artifact_types: Filter by artifact types
            include_metadata: Include metadata in archive
            
        Returns:
            Archive artifact ID
        """
        archive_path = Path(archive_path)
        
        # Get artifacts to archive
        artifacts_to_archive = []
        if artifact_types:
            for artifact_type in artifact_types:
                artifacts_to_archive.extend(
                    self.get_artifacts(artifact_type=artifact_type, model_id=model_id, evaluation_id=evaluation_id)
                )
        else:
            artifacts_to_archive = self.get_artifacts(model_id=model_id, evaluation_id=evaluation_id)
        
        if not artifacts_to_archive:
            raise ValueError("No artifacts found matching criteria")
        
        # Create archive
        with tarfile.open(archive_path, 'w:gz') as tar:
            # Add artifact files
            for artifact in artifacts_to_archive:
                file_path = Path(artifact.file_path)
                if file_path.exists():
                    arcname = f"artifacts/{artifact.artifact_type.value}/{artifact.artifact_id}"
                    tar.add(file_path, arcname=arcname)
            
            # Add metadata if requested
            if include_metadata:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    metadata = {
                        "archive_created": datetime.now().isoformat(),
                        "model_id": model_id,
                        "evaluation_id": evaluation_id,
                        "artifact_count": len(artifacts_to_archive),
                        "artifacts": [asdict(a) for a in artifacts_to_archive]
                    }
                    json.dump(metadata, f, indent=2, default=str)
                    temp_path = f.name
                
                tar.add(temp_path, arcname="metadata.json")
                os.unlink(temp_path)
        
        # Store archive as artifact
        archive_id = self.store_artifact(
            archive_path,
            ArtifactType.LOG,  # Use LOG type for archives
            model_id=model_id,
            evaluation_id=evaluation_id,
            tags=["archive", "backup"],
            metadata={
                "archive_type": "artifact_collection",
                "artifact_count": len(artifacts_to_archive),
                "includes_metadata": include_metadata
            },
            compress=False  # Already compressed
        )
        
        self.logger.info(f"Created archive {archive_id} with {len(artifacts_to_archive)} artifacts")
        return archive_id
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            "total_artifacts": 0,
            "total_size_bytes": 0,
            "compressed_size_bytes": 0,
            "compression_ratio": 0.0,
            "artifacts_by_type": {},
            "artifacts_by_model": {},
            "storage_efficiency": 0.0
        }
        
        artifacts = self.get_artifacts()
        stats["total_artifacts"] = len(artifacts)
        
        for artifact in artifacts:
            # Size statistics
            stats["total_size_bytes"] += artifact.original_size
            if artifact.compressed_size:
                stats["compressed_size_bytes"] += artifact.compressed_size
            else:
                stats["compressed_size_bytes"] += artifact.original_size
            
            # Type statistics
            type_name = artifact.artifact_type.value
            if type_name not in stats["artifacts_by_type"]:
                stats["artifacts_by_type"][type_name] = {"count": 0, "size_bytes": 0}
            stats["artifacts_by_type"][type_name]["count"] += 1
            stats["artifacts_by_type"][type_name]["size_bytes"] += artifact.original_size
            
            # Model statistics
            if artifact.model_id:
                if artifact.model_id not in stats["artifacts_by_model"]:
                    stats["artifacts_by_model"][artifact.model_id] = {"count": 0, "size_bytes": 0}
                stats["artifacts_by_model"][artifact.model_id]["count"] += 1
                stats["artifacts_by_model"][artifact.model_id]["size_bytes"] += artifact.original_size
        
        # Calculate ratios
        if stats["total_size_bytes"] > 0:
            stats["compression_ratio"] = stats["compressed_size_bytes"] / stats["total_size_bytes"]
            stats["storage_efficiency"] = 1.0 - stats["compression_ratio"]
        
        return stats


def create_artifact_manager(config_path: Optional[str] = None) -> ArtifactManager:
    """
    Factory function to create ArtifactManager instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured ArtifactManager instance
    """
    config = ArtifactManagerConfig()
    
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    return ArtifactManager(config)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create artifact manager
    manager = create_artifact_manager()
    
    # Example: Store some artifacts
    print("Artifact Manager initialized successfully!")
    print(f"Base path: {manager.base_path}")
    
    # Print storage stats
    stats = manager.get_storage_stats()
    print(f"Storage stats: {json.dumps(stats, indent=2)}")