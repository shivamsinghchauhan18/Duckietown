#!/usr/bin/env python3
"""
📦 ARTIFACT MANAGER EXAMPLE 📦
Comprehensive example demonstrating ArtifactManager capabilities

This example shows how to use the ArtifactManager for:
- Storing evaluation results and artifacts
- Managing episode data exports
- Tracking evaluation history and champion progression
- Cleaning up and archiving artifacts
"""

import os
import sys
import json
import tempfile
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from duckietown_utils.artifact_manager import (
    ArtifactManager, ArtifactManagerConfig, ArtifactType, CompressionType,
    create_artifact_manager
)

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def create_sample_data(temp_dir: Path):
    """Create sample data files for demonstration."""
    # Create sample evaluation report
    report_data = {
        "model_id": "enhanced_duckietown_v1",
        "evaluation_id": "eval_20250816_001",
        "timestamp": datetime.now().isoformat(),
        "global_score": 0.87,
        "success_rate": 0.92,
        "metrics": {
            "mean_reward": 0.85,
            "episode_length": 145.2,
            "lateral_deviation": 0.12,
            "heading_error": 2.3,
            "smoothness": 0.08
        },
        "suite_results": {
            "base": {"success_rate": 0.95, "mean_reward": 0.88},
            "hard": {"success_rate": 0.89, "mean_reward": 0.82},
            "ood": {"success_rate": 0.87, "mean_reward": 0.79}
        }
    }
    
    report_file = temp_dir / "evaluation_report.json"
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    # Create sample episode data
    episode_data = []
    for i in range(50):
        episode_data.append({
            "episode_id": f"ep_{i:03d}",
            "model_id": "enhanced_duckietown_v1",
            "suite": "base" if i < 20 else "hard" if i < 40 else "ood",
            "map_name": "loop_empty" if i % 2 == 0 else "zigzag_dists",
            "seed": 1000 + i,
            "success": np.random.random() > 0.1,
            "collision": np.random.random() < 0.05,
            "off_lane": np.random.random() < 0.03,
            "reward_mean": np.random.normal(0.8, 0.15),
            "episode_length": int(np.random.normal(150, 30)),
            "lateral_deviation": np.random.exponential(0.1),
            "heading_error": np.random.exponential(2.0),
            "smoothness": np.random.exponential(0.05),
            "timestamp": datetime.now().isoformat()
        })
    
    # Create sample video file (mock)
    video_file = temp_dir / "failure_episode_042.mp4"
    with open(video_file, 'wb') as f:
        f.write(b"MOCK_VIDEO_DATA" * 1000)  # Simulate video data
    
    # Create sample trace file
    trace_data = {
        "episode_id": "ep_042",
        "failure_type": "collision",
        "states": [
            {"step": i, "position": [i * 0.1, 0.0], "velocity": 0.5, "steering": np.random.normal(0, 0.1)}
            for i in range(100)
        ],
        "actions": [
            {"step": i, "throttle": 0.5, "steering": np.random.normal(0, 0.1)}
            for i in range(100)
        ]
    }
    
    trace_file = temp_dir / "failure_trace_042.json"
    with open(trace_file, 'w') as f:
        json.dump(trace_data, f, indent=2, default=str)
    
    return {
        "report_file": report_file,
        "episode_data": episode_data,
        "video_file": video_file,
        "trace_file": trace_file
    }

def demonstrate_basic_storage(manager: ArtifactManager, sample_data: dict):
    """Demonstrate basic artifact storage."""
    print("\n🔹 Demonstrating Basic Artifact Storage")
    print("=" * 50)
    
    # Store evaluation report
    report_id = manager.store_artifact(
        sample_data["report_file"],
        ArtifactType.EVALUATION_RESULT,
        model_id="enhanced_duckietown_v1",
        evaluation_id="eval_20250816_001",
        tags=["evaluation", "comprehensive", "v1"],
        metadata={"version": "1.0", "suite_count": 3}
    )
    print(f"✅ Stored evaluation report: {report_id}")
    
    # Store video with compression
    video_id = manager.store_video_with_compression(
        sample_data["video_file"],
        model_id="enhanced_duckietown_v1",
        evaluation_id="eval_20250816_001",
        episode_id="ep_042",
        metadata={"failure_type": "collision", "map_name": "loop_empty"}
    )
    print(f"✅ Stored failure video: {video_id}")
    
    # Store trace file
    trace_id = manager.store_trace_file(
        sample_data["trace_file"],
        model_id="enhanced_duckietown_v1",
        evaluation_id="eval_20250816_001",
        episode_id="ep_042",
        metadata={"failure_type": "collision", "step_count": 100}
    )
    print(f"✅ Stored failure trace: {trace_id}")
    
    return {"report_id": report_id, "video_id": video_id, "trace_id": trace_id}

def demonstrate_episode_export(manager: ArtifactManager, episode_data: list):
    """Demonstrate episode data export."""
    print("\n🔹 Demonstrating Episode Data Export")
    print("=" * 50)
    
    # Export in both formats
    artifact_ids = manager.export_episode_data(
        episode_data,
        format_type="both",
        model_id="enhanced_duckietown_v1",
        evaluation_id="eval_20250816_001"
    )
    
    print(f"✅ Exported episode data:")
    print(f"   📄 JSON format: {artifact_ids['json']}")
    print(f"   📊 CSV format: {artifact_ids['csv']}")
    print(f"   📈 Total episodes: {len(episode_data)}")
    
    return artifact_ids

def demonstrate_history_tracking(manager: ArtifactManager, artifact_ids: dict):
    """Demonstrate evaluation history and champion tracking."""
    print("\n🔹 Demonstrating History Tracking")
    print("=" * 50)
    
    # Record evaluation history
    manager.record_evaluation_history(
        evaluation_id="eval_20250816_001",
        model_id="enhanced_duckietown_v1",
        global_score=0.87,
        success_rate=0.92,
        is_champion=False,
        artifacts=list(artifact_ids.values()),
        metadata={"suite_count": 3, "total_episodes": 50}
    )
    print("✅ Recorded evaluation history")
    
    # Record previous champion for comparison
    manager.record_evaluation_history(
        evaluation_id="eval_20250815_001",
        model_id="baseline_model_v2",
        global_score=0.82,
        success_rate=0.88,
        is_champion=True,
        champion_rank=1,
        metadata={"suite_count": 3, "total_episodes": 50}
    )
    print("✅ Recorded previous champion")
    
    # Record new champion progression
    champion_id = manager.record_champion_progression(
        model_id="enhanced_duckietown_v1",
        global_score=0.87,
        success_rate=0.92,
        previous_champion="baseline_model_v2",
        improvement_metrics={
            "score_improvement": 0.05,
            "success_improvement": 0.04,
            "reward_improvement": 0.03
        },
        artifacts=list(artifact_ids.values())
    )
    print(f"✅ Recorded champion progression: {champion_id}")
    
    return champion_id

def demonstrate_querying(manager: ArtifactManager):
    """Demonstrate querying artifacts and history."""
    print("\n🔹 Demonstrating Querying Capabilities")
    print("=" * 50)
    
    # Query all artifacts
    all_artifacts = manager.get_artifacts()
    print(f"📦 Total artifacts: {len(all_artifacts)}")
    
    # Query by type
    videos = manager.get_artifacts(artifact_type=ArtifactType.VIDEO)
    traces = manager.get_artifacts(artifact_type=ArtifactType.TRACE)
    reports = manager.get_artifacts(artifact_type=ArtifactType.EVALUATION_RESULT)
    
    print(f"🎥 Video artifacts: {len(videos)}")
    print(f"📊 Trace artifacts: {len(traces)}")
    print(f"📋 Report artifacts: {len(reports)}")
    
    # Query by model
    model_artifacts = manager.get_artifacts(model_id="enhanced_duckietown_v1")
    print(f"🤖 Artifacts for enhanced_duckietown_v1: {len(model_artifacts)}")
    
    # Query evaluation history
    history = manager.get_evaluation_history()
    print(f"📈 Evaluation history entries: {len(history)}")
    
    for entry in history:
        print(f"   📅 {entry.timestamp[:19]} - {entry.model_id}: "
              f"Score={entry.global_score:.3f}, Success={entry.success_rate:.3f}")
    
    # Query champion progression
    champions = manager.get_champion_progression()
    print(f"🏆 Champion progression entries: {len(champions)}")
    
    for champion in champions:
        print(f"   👑 {champion.timestamp[:19]} - {champion.model_id}: "
              f"Score={champion.global_score:.3f}")

def demonstrate_retrieval(manager: ArtifactManager, artifact_ids: dict, temp_dir: Path):
    """Demonstrate artifact retrieval."""
    print("\n🔹 Demonstrating Artifact Retrieval")
    print("=" * 50)
    
    # Retrieve video artifact
    if "video_id" in artifact_ids:
        output_path = temp_dir / "retrieved_video.mp4"
        retrieved_path = manager.retrieve_artifact(artifact_ids["video_id"], output_path)
        print(f"✅ Retrieved video to: {retrieved_path}")
        print(f"   📏 File size: {retrieved_path.stat().st_size} bytes")
    
    # Retrieve trace artifact
    if "trace_id" in artifact_ids:
        output_path = temp_dir / "retrieved_trace.json"
        retrieved_path = manager.retrieve_artifact(artifact_ids["trace_id"], output_path)
        print(f"✅ Retrieved trace to: {retrieved_path}")
        
        # Verify content
        with open(retrieved_path, 'r') as f:
            trace_data = json.load(f)
        print(f"   📊 Trace contains {len(trace_data['states'])} states")

def demonstrate_storage_stats(manager: ArtifactManager):
    """Demonstrate storage statistics."""
    print("\n🔹 Demonstrating Storage Statistics")
    print("=" * 50)
    
    stats = manager.get_storage_stats()
    
    print(f"📊 Storage Statistics:")
    print(f"   📦 Total artifacts: {stats['total_artifacts']}")
    print(f"   💾 Total size: {stats['total_size_bytes'] / 1024 / 1024:.2f} MB")
    print(f"   🗜️  Compressed size: {stats['compressed_size_bytes'] / 1024 / 1024:.2f} MB")
    print(f"   📉 Compression ratio: {stats['compression_ratio']:.3f}")
    print(f"   💡 Storage efficiency: {stats['storage_efficiency']:.1%}")
    
    print(f"\n📋 Artifacts by type:")
    for artifact_type, type_stats in stats['artifacts_by_type'].items():
        print(f"   {artifact_type}: {type_stats['count']} artifacts, "
              f"{type_stats['size_bytes'] / 1024:.1f} KB")
    
    print(f"\n🤖 Artifacts by model:")
    for model_id, model_stats in stats['artifacts_by_model'].items():
        print(f"   {model_id}: {model_stats['count']} artifacts, "
              f"{model_stats['size_bytes'] / 1024:.1f} KB")

def demonstrate_archival(manager: ArtifactManager, temp_dir: Path):
    """Demonstrate artifact archival."""
    print("\n🔹 Demonstrating Artifact Archival")
    print("=" * 50)
    
    # Create archive for specific model
    archive_path = temp_dir / "model_archive.tar.gz"
    archive_id = manager.create_archive(
        archive_path,
        model_id="enhanced_duckietown_v1",
        include_metadata=True
    )
    
    print(f"✅ Created archive: {archive_id}")
    print(f"   📁 Archive path: {archive_path}")
    print(f"   📏 Archive size: {archive_path.stat().st_size / 1024:.1f} KB")
    
    # Get archive artifact info
    archives = manager.get_artifacts(tags=["archive"])
    if archives:
        archive = archives[0]
        print(f"   📦 Archived {archive.metadata['artifact_count']} artifacts")
        print(f"   📋 Includes metadata: {archive.metadata['includes_metadata']}")

def demonstrate_cleanup(manager: ArtifactManager):
    """Demonstrate artifact cleanup."""
    print("\n🔹 Demonstrating Artifact Cleanup")
    print("=" * 50)
    
    # Dry run cleanup
    print("🧹 Performing dry run cleanup...")
    stats = manager.cleanup_artifacts(
        dry_run=True,
        max_age_days=0,  # Clean everything for demo
        keep_champions=True
    )
    
    print(f"   📊 Cleanup statistics (dry run):")
    print(f"   📦 Total artifacts: {stats['total_artifacts']}")
    print(f"   🗑️  Would delete: {stats['deleted_artifacts']}")
    print(f"   🏆 Preserved champions: {stats['preserved_champions']}")
    print(f"   💾 Would free: {stats['freed_bytes'] / 1024:.1f} KB")
    
    # Note: In a real scenario, you might want to actually perform cleanup
    # stats = manager.cleanup_artifacts(dry_run=False, max_age_days=30, keep_champions=True)

def main():
    """Main demonstration function."""
    setup_logging()
    
    print("🚀 ARTIFACT MANAGER DEMONSTRATION")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create artifact manager with custom config
        config = ArtifactManagerConfig(
            base_path=str(temp_path / "artifacts"),
            database_path=str(temp_path / "artifacts" / "demo_registry.db"),
            compression_enabled=True,
            default_compression=CompressionType.GZIP
        )
        
        manager = ArtifactManager(config)
        print(f"✅ Initialized ArtifactManager at: {manager.base_path}")
        
        # Create sample data
        sample_data = create_sample_data(temp_path)
        print(f"✅ Created sample data files")
        
        # Demonstrate capabilities
        artifact_ids = demonstrate_basic_storage(manager, sample_data)
        episode_ids = demonstrate_episode_export(manager, sample_data["episode_data"])
        artifact_ids.update(episode_ids)
        
        champion_id = demonstrate_history_tracking(manager, artifact_ids)
        demonstrate_querying(manager)
        demonstrate_retrieval(manager, artifact_ids, temp_path)
        demonstrate_storage_stats(manager)
        demonstrate_archival(manager, temp_path)
        demonstrate_cleanup(manager)
        
        print("\n🎉 DEMONSTRATION COMPLETE!")
        print("=" * 60)
        print("The ArtifactManager provides comprehensive capabilities for:")
        print("• 📦 Artifact storage with compression and versioning")
        print("• 📊 Episode data export in multiple formats")
        print("• 🎥 Video and trace file management")
        print("• 📈 Evaluation history tracking")
        print("• 🏆 Champion progression monitoring")
        print("• 🔍 Flexible querying and retrieval")
        print("• 📋 Storage statistics and monitoring")
        print("• 📁 Archival and backup capabilities")
        print("• 🧹 Automated cleanup utilities")

if __name__ == "__main__":
    main()