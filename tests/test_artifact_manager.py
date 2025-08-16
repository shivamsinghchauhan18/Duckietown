#!/usr/bin/env python3
"""
🧪 ARTIFACT MANAGER TESTS 🧪
Comprehensive unit tests for the ArtifactManager system

Tests cover artifact storage and versioning, episode data export,
video and trace file management, evaluation history tracking,
champion progression, and cleanup utilities.
"""

import os
import sys
import json
import gzip
import tempfile
import shutil
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pytest
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from duckietown_utils.artifact_manager import (
    ArtifactManager, ArtifactManagerConfig, ArtifactType, CompressionType,
    StoragePolicy, ArtifactMetadata, EvaluationHistory, ChampionProgression,
    create_artifact_manager
)

class TestArtifactManager:
    """Test suite for ArtifactManager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration."""
        return ArtifactManagerConfig(
            base_path=str(temp_dir / "artifacts"),
            database_path=str(temp_dir / "artifacts" / "test_registry.db"),
            compression_enabled=True,
            auto_cleanup_enabled=False
        )
    
    @pytest.fixture
    def manager(self, config):
        """Create ArtifactManager instance."""
        return ArtifactManager(config)
    
    @pytest.fixture
    def sample_file(self, temp_dir):
        """Create sample file for testing."""
        file_path = temp_dir / "sample.txt"
        with open(file_path, 'w') as f:
            f.write("This is a test file for artifact management.\n" * 100)
        return file_path
    
    @pytest.fixture
    def sample_json_file(self, temp_dir):
        """Create sample JSON file for testing."""
        file_path = temp_dir / "sample.json"
        data = {"test": "data", "numbers": [1, 2, 3, 4, 5]}
        with open(file_path, 'w') as f:
            json.dump(data, f)
        return file_path
    
    @pytest.fixture
    def sample_episode_data(self):
        """Create sample episode data."""
        return [
            {
                "episode_id": "ep_001",
                "model_id": "test_model",
                "success": True,
                "reward": 0.85,
                "steps": 150,
                "map_name": "loop_empty"
            },
            {
                "episode_id": "ep_002",
                "model_id": "test_model",
                "success": False,
                "reward": 0.23,
                "steps": 75,
                "map_name": "loop_empty"
            }
        ]
    
    def test_initialization(self, config):
        """Test ArtifactManager initialization."""
        manager = ArtifactManager(config)
        
        # Check directory structure
        assert manager.base_path.exists()
        for artifact_type in ArtifactType:
            assert manager.subdirs[artifact_type].exists()
        
        # Check database initialization
        db_path = Path(config.database_path)
        assert db_path.exists()
        
        # Verify tables exist
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}
            assert "artifacts" in tables
            assert "evaluation_history" in tables
            assert "champion_progression" in tables
    
    def test_store_artifact_uncompressed(self, manager, sample_file):
        """Test storing artifact without compression."""
        artifact_id = manager.store_artifact(
            sample_file,
            ArtifactType.EVALUATION_RESULT,
            model_id="test_model",
            compress=False
        )
        
        assert artifact_id is not None
        
        # Verify artifact is stored
        artifacts = manager.get_artifacts(artifact_type=ArtifactType.EVALUATION_RESULT)
        assert len(artifacts) == 1
        
        artifact = artifacts[0]
        assert artifact.artifact_id == artifact_id
        assert artifact.model_id == "test_model"
        assert artifact.compression_type == CompressionType.NONE
        assert artifact.compressed_size is None
        
        # Verify file exists
        stored_path = Path(artifact.file_path)
        assert stored_path.exists()
        assert stored_path.stat().st_size == sample_file.stat().st_size
    
    def test_store_artifact_compressed(self, manager, sample_file):
        """Test storing artifact with compression."""
        artifact_id = manager.store_artifact(
            sample_file,
            ArtifactType.VIDEO,
            model_id="test_model",
            compress=True
        )
        
        artifacts = manager.get_artifacts(artifact_type=ArtifactType.VIDEO)
        assert len(artifacts) == 1
        
        artifact = artifacts[0]
        assert artifact.compression_type == CompressionType.GZIP
        assert artifact.compressed_size is not None
        assert artifact.compressed_size < artifact.original_size
        
        # Verify compressed file exists
        stored_path = Path(artifact.file_path)
        assert stored_path.exists()
        assert stored_path.suffix == ".gz"
    
    def test_store_artifact_with_metadata(self, manager, sample_file):
        """Test storing artifact with metadata and tags."""
        metadata = {"test_key": "test_value", "number": 42}
        tags = ["test", "evaluation", "important"]
        
        artifact_id = manager.store_artifact(
            sample_file,
            ArtifactType.REPORT,
            model_id="test_model",
            evaluation_id="eval_001",
            suite_name="base",
            map_name="loop_empty",
            episode_id="ep_001",
            tags=tags,
            metadata=metadata
        )
        
        artifacts = manager.get_artifacts(artifact_type=ArtifactType.REPORT)
        artifact = artifacts[0]
        
        assert artifact.model_id == "test_model"
        assert artifact.evaluation_id == "eval_001"
        assert artifact.suite_name == "base"
        assert artifact.map_name == "loop_empty"
        assert artifact.episode_id == "ep_001"
        assert artifact.tags == tags
        assert artifact.metadata == metadata
    
    def test_export_episode_data_json(self, manager, sample_episode_data):
        """Test exporting episode data as JSON."""
        artifact_ids = manager.export_episode_data(
            sample_episode_data,
            format_type="json",
            model_id="test_model",
            evaluation_id="eval_001"
        )
        
        assert "json" in artifact_ids
        
        # Verify artifact is stored
        artifacts = manager.get_artifacts(artifact_type=ArtifactType.EPISODE_DATA)
        assert len(artifacts) == 1
        
        artifact = artifacts[0]
        assert "json" in artifact.tags
        assert artifact.metadata["format"] == "json"
        assert artifact.metadata["episode_count"] == 2
    
    def test_export_episode_data_csv(self, manager, sample_episode_data):
        """Test exporting episode data as CSV."""
        artifact_ids = manager.export_episode_data(
            sample_episode_data,
            format_type="csv",
            model_id="test_model",
            evaluation_id="eval_001"
        )
        
        assert "csv" in artifact_ids
        
        # Verify artifact is stored
        artifacts = manager.get_artifacts(artifact_type=ArtifactType.EPISODE_DATA)
        assert len(artifacts) == 1
        
        artifact = artifacts[0]
        assert "csv" in artifact.tags
        assert artifact.metadata["format"] == "csv"
    
    def test_export_episode_data_both(self, manager, sample_episode_data):
        """Test exporting episode data in both formats."""
        artifact_ids = manager.export_episode_data(
            sample_episode_data,
            format_type="both",
            model_id="test_model",
            evaluation_id="eval_001"
        )
        
        assert "json" in artifact_ids
        assert "csv" in artifact_ids
        
        # Verify both artifacts are stored
        artifacts = manager.get_artifacts(artifact_type=ArtifactType.EPISODE_DATA)
        assert len(artifacts) == 2
    
    def test_store_video_with_compression(self, manager, sample_file):
        """Test storing video file with compression."""
        artifact_id = manager.store_video_with_compression(
            sample_file,
            model_id="test_model",
            evaluation_id="eval_001",
            episode_id="ep_001",
            metadata={"failure_type": "collision"}
        )
        
        artifacts = manager.get_artifacts(artifact_type=ArtifactType.VIDEO)
        assert len(artifacts) == 1
        
        artifact = artifacts[0]
        assert artifact.artifact_id == artifact_id
        assert "video" in artifact.tags
        assert "failure_analysis" in artifact.tags
        assert artifact.metadata["failure_type"] == "collision"
        assert artifact.compression_type != CompressionType.NONE
    
    def test_store_trace_file(self, manager, sample_json_file):
        """Test storing trace file."""
        artifact_id = manager.store_trace_file(
            sample_json_file,
            model_id="test_model",
            evaluation_id="eval_001",
            episode_id="ep_001",
            metadata={"trace_type": "state_sequence"}
        )
        
        artifacts = manager.get_artifacts(artifact_type=ArtifactType.TRACE)
        assert len(artifacts) == 1
        
        artifact = artifacts[0]
        assert artifact.artifact_id == artifact_id
        assert "trace" in artifact.tags
        assert "failure_analysis" in artifact.tags
        assert artifact.metadata["trace_type"] == "state_sequence"
    
    def test_record_evaluation_history(self, manager):
        """Test recording evaluation history."""
        manager.record_evaluation_history(
            evaluation_id="eval_001",
            model_id="test_model",
            global_score=0.85,
            success_rate=0.92,
            is_champion=True,
            champion_rank=1,
            artifacts=["artifact_001", "artifact_002"],
            metadata={"suite": "comprehensive"}
        )
        
        history = manager.get_evaluation_history()
        assert len(history) == 1
        
        entry = history[0]
        assert entry.evaluation_id == "eval_001"
        assert entry.model_id == "test_model"
        assert entry.global_score == 0.85
        assert entry.success_rate == 0.92
        assert entry.is_champion is True
        assert entry.champion_rank == 1
        assert entry.artifacts == ["artifact_001", "artifact_002"]
        assert entry.metadata["suite"] == "comprehensive"
    
    def test_record_champion_progression(self, manager):
        """Test recording champion progression."""
        champion_id = manager.record_champion_progression(
            model_id="new_champion",
            global_score=0.92,
            success_rate=0.95,
            previous_champion="old_champion",
            improvement_metrics={"score_improvement": 0.07, "success_improvement": 0.03},
            artifacts=["champion_artifact_001"]
        )
        
        assert champion_id is not None
        assert "new_cham" in champion_id  # Truncated to 8 chars
        
        progression = manager.get_champion_progression()
        assert len(progression) == 1
        
        entry = progression[0]
        assert entry.champion_id == champion_id
        assert entry.model_id == "new_champion"
        assert entry.global_score == 0.92
        assert entry.success_rate == 0.95
        assert entry.previous_champion == "old_champion"
        assert entry.improvement_metrics["score_improvement"] == 0.07
    
    def test_get_evaluation_history_filtered(self, manager):
        """Test getting filtered evaluation history."""
        # Add multiple history entries
        manager.record_evaluation_history("eval_001", "model_a", 0.8, 0.9)
        manager.record_evaluation_history("eval_002", "model_b", 0.7, 0.8)
        manager.record_evaluation_history("eval_003", "model_a", 0.85, 0.92, is_champion=True)
        
        # Test model filter
        history = manager.get_evaluation_history(model_id="model_a")
        assert len(history) == 2
        assert all(entry.model_id == "model_a" for entry in history)
        
        # Test champions only
        history = manager.get_evaluation_history(champions_only=True)
        assert len(history) == 1
        assert history[0].is_champion is True
        
        # Test limit
        history = manager.get_evaluation_history(limit=2)
        assert len(history) == 2
    
    def test_get_artifacts_filtered(self, manager, sample_file):
        """Test getting filtered artifacts."""
        # Store multiple artifacts
        manager.store_artifact(sample_file, ArtifactType.VIDEO, model_id="model_a", tags=["test"])
        manager.store_artifact(sample_file, ArtifactType.TRACE, model_id="model_b", tags=["debug"])
        manager.store_artifact(sample_file, ArtifactType.VIDEO, model_id="model_a", tags=["production"])
        
        # Test type filter
        artifacts = manager.get_artifacts(artifact_type=ArtifactType.VIDEO)
        assert len(artifacts) == 2
        assert all(a.artifact_type == ArtifactType.VIDEO for a in artifacts)
        
        # Test model filter
        artifacts = manager.get_artifacts(model_id="model_a")
        assert len(artifacts) == 2
        assert all(a.model_id == "model_a" for a in artifacts)
        
        # Test tag filter
        artifacts = manager.get_artifacts(tags=["test"])
        assert len(artifacts) == 1
        assert "test" in artifacts[0].tags
        
        # Test limit
        artifacts = manager.get_artifacts(limit=2)
        assert len(artifacts) == 2
    
    def test_retrieve_artifact_uncompressed(self, manager, sample_file, temp_dir):
        """Test retrieving uncompressed artifact."""
        artifact_id = manager.store_artifact(
            sample_file,
            ArtifactType.EVALUATION_RESULT,
            compress=False
        )
        
        output_path = temp_dir / "retrieved.txt"
        retrieved_path = manager.retrieve_artifact(artifact_id, output_path)
        
        assert retrieved_path == output_path
        assert output_path.exists()
        
        # Verify content matches
        with open(sample_file, 'r') as f1, open(output_path, 'r') as f2:
            assert f1.read() == f2.read()
    
    def test_retrieve_artifact_compressed(self, manager, sample_file, temp_dir):
        """Test retrieving compressed artifact."""
        artifact_id = manager.store_artifact(
            sample_file,
            ArtifactType.VIDEO,
            compress=True
        )
        
        output_path = temp_dir / "retrieved.txt"
        retrieved_path = manager.retrieve_artifact(artifact_id, output_path)
        
        assert retrieved_path == output_path
        assert output_path.exists()
        
        # Verify content matches original
        with open(sample_file, 'r') as f1, open(output_path, 'r') as f2:
            assert f1.read() == f2.read()
    
    def test_cleanup_artifacts(self, manager, sample_file):
        """Test artifact cleanup."""
        # Store some artifacts
        artifact_id1 = manager.store_artifact(sample_file, ArtifactType.VIDEO, model_id="model_a")
        artifact_id2 = manager.store_artifact(sample_file, ArtifactType.TRACE, model_id="model_b")
        
        # Record champion with artifact
        champion_id = manager.record_champion_progression(
            "model_a", 0.9, 0.95, artifacts=[artifact_id1]
        )
        
        # Test dry run - use negative age to force cleanup of all artifacts
        import time
        time.sleep(0.1)  # Ensure some time passes
        stats = manager.cleanup_artifacts(dry_run=True, max_age_days=-1, keep_champions=True)
        assert stats["total_artifacts"] == 2
        assert stats["deleted_artifacts"] == 1  # Only non-champion artifact
        assert stats["preserved_champions"] == 1
        
        # Verify artifacts still exist
        artifacts = manager.get_artifacts()
        assert len(artifacts) == 2
        
        # Test actual cleanup
        stats = manager.cleanup_artifacts(dry_run=False, max_age_days=-1, keep_champions=True)
        assert stats["deleted_artifacts"] == 1
        
        # Verify champion artifact preserved
        artifacts = manager.get_artifacts()
        assert len(artifacts) == 1
        assert artifacts[0].artifact_id == artifact_id1
    
    def test_create_archive(self, manager, sample_file, temp_dir):
        """Test creating artifact archive."""
        # Store some artifacts
        manager.store_artifact(sample_file, ArtifactType.VIDEO, model_id="test_model")
        manager.store_artifact(sample_file, ArtifactType.TRACE, model_id="test_model")
        
        archive_path = temp_dir / "test_archive.tar.gz"
        archive_id = manager.create_archive(
            archive_path,
            model_id="test_model",
            include_metadata=True
        )
        
        assert archive_id is not None
        assert archive_path.exists()
        
        # Verify archive artifact is stored
        archives = manager.get_artifacts(tags=["archive"])
        assert len(archives) == 1
        assert archives[0].artifact_id == archive_id
        assert archives[0].metadata["artifact_count"] == 2
        assert archives[0].metadata["includes_metadata"] is True
    
    def test_get_storage_stats(self, manager, sample_file):
        """Test getting storage statistics."""
        # Store some artifacts
        manager.store_artifact(sample_file, ArtifactType.VIDEO, model_id="model_a", compress=True)
        manager.store_artifact(sample_file, ArtifactType.TRACE, model_id="model_b", compress=False)
        
        stats = manager.get_storage_stats()
        
        assert stats["total_artifacts"] == 2
        assert stats["total_size_bytes"] > 0
        assert stats["compressed_size_bytes"] > 0
        assert stats["compression_ratio"] > 0
        assert "video" in stats["artifacts_by_type"]
        assert "trace" in stats["artifacts_by_type"]
        assert "model_a" in stats["artifacts_by_model"]
        assert "model_b" in stats["artifacts_by_model"]
    
    def test_error_handling(self, manager, temp_dir):
        """Test error handling scenarios."""
        # Test storing non-existent file
        with pytest.raises(FileNotFoundError):
            manager.store_artifact(
                temp_dir / "nonexistent.txt",
                ArtifactType.VIDEO
            )
        
        # Test retrieving non-existent artifact
        with pytest.raises(ValueError):
            manager.retrieve_artifact("nonexistent_artifact")
        
        # Test exporting empty episode data
        with pytest.raises(ValueError):
            manager.export_episode_data([])
    
    def test_create_artifact_manager_factory(self, temp_dir):
        """Test factory function for creating ArtifactManager."""
        # Test with default config
        manager = create_artifact_manager()
        assert isinstance(manager, ArtifactManager)
        
        # Test with config file
        config_path = temp_dir / "config.json"
        config_data = {
            "base_path": str(temp_dir / "custom_artifacts"),
            "compression_enabled": False
        }
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        manager = create_artifact_manager(str(config_path))
        assert manager.config.base_path == str(temp_dir / "custom_artifacts")
        assert manager.config.compression_enabled is False
    
    def test_concurrent_access(self, manager, sample_file):
        """Test concurrent access to artifact manager."""
        import threading
        
        results = []
        errors = []
        
        def store_artifact(i):
            try:
                artifact_id = manager.store_artifact(
                    sample_file,
                    ArtifactType.VIDEO,
                    model_id=f"model_{i}",
                    tags=[f"thread_{i}"]
                )
                results.append(artifact_id)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=store_artifact, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0
        assert len(results) == 5
        assert len(set(results)) == 5  # All unique artifact IDs
        
        # Verify all artifacts stored
        artifacts = manager.get_artifacts(artifact_type=ArtifactType.VIDEO)
        assert len(artifacts) == 5


if __name__ == "__main__":
    pytest.main([__file__])