#!/usr/bin/env python3
"""
🧹 PROJECT CLEANUP SCRIPT 🧹

This script removes all unnecessary files and keeps only the essential components
for the Ultimate Champion Training System.

KEEPS:
- Core training scripts (ultimate champion system)
- Essential configuration files
- Best trained models
- Core utilities
- Documentation (essential)
- README and setup files

REMOVES:
- Redundant training scripts
- Old validation files
- Excessive logs
- Duplicate models
- Development artifacts
- Test files (keeping core tests only)
- Redundant documentation
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

class ProjectCleaner:
    """Clean up the project by removing unnecessary files."""
    
    def __init__(self):
        self.removed_files = []
        self.removed_dirs = []
        self.kept_files = []
        self.cleanup_report = []
        
    def clean_project(self):
        """Execute comprehensive project cleanup."""
        print("🧹 STARTING PROJECT CLEANUP")
        print("=" * 50)
        
        # Clean different categories
        self._clean_redundant_training_scripts()
        self._clean_old_models()
        self._clean_excessive_logs()
        self._clean_redundant_checkpoints()
        self._clean_development_artifacts()
        self._clean_redundant_docs()
        self._clean_test_files()
        self._clean_cache_files()
        self._clean_redundant_configs()
        self._clean_examples()
        
        # Generate cleanup report
        self._generate_cleanup_report()
        
        print("\n✅ PROJECT CLEANUP COMPLETED!")
        print(f"📊 Files removed: {len(self.removed_files)}")
        print(f"📁 Directories removed: {len(self.removed_dirs)}")
        print(f"💾 Essential files kept: {len(self.kept_files)}")
    
    def _clean_redundant_training_scripts(self):
        """Remove redundant training scripts, keep only the ultimate champion system."""
        print("\n🚀 Cleaning training scripts...")
        
        # Keep only essential training scripts
        keep_scripts = [
            'train_ultimate_champion.py',
            'continuous_champion_training.py',
            'evaluate_trained_model.py'
        ]
        
        # Remove redundant training scripts
        remove_scripts = [
            'train_simple_reliable.py',
            'train_best_strategy.py',
            'train_cpu_optimized.py',
            'train_competitive_champion.py',
            'train_extended_champion.py',
            'training_summary.py'
        ]
        
        for script in remove_scripts:
            if Path(script).exists():
                self._remove_file(script)
        
        for script in keep_scripts:
            if Path(script).exists():
                self._keep_file(script)
    
    def _clean_old_models(self):
        """Keep only the best champion models."""
        print("\n🏆 Cleaning model files...")
        
        models_dir = Path("models")
        if models_dir.exists():
            model_files = list(models_dir.glob("*.json"))
            
            # Keep only the latest and best champion models
            champion_models = [f for f in model_files if "ultimate_champion" in f.name]
            simple_models = [f for f in model_files if "simple_model" in f.name]
            
            # Keep latest 2 champion models and remove the rest
            if len(champion_models) > 2:
                champion_models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                keep_models = champion_models[:2]  # Keep latest 2
                remove_models = champion_models[2:]  # Remove older ones
                
                for model in remove_models:
                    self._remove_file(str(model))
                
                for model in keep_models:
                    self._keep_file(str(model))
            
            # Remove simple models (outdated)
            for model in simple_models:
                self._remove_file(str(model))
    
    def _clean_excessive_logs(self):
        """Clean excessive log files, keep only essential ones."""
        print("\n📊 Cleaning log files...")
        
        logs_dir = Path("logs")
        if logs_dir.exists():
            # Keep essential logs
            keep_logs = [
                "logs/champion_training/",
                "logs/training_log.jsonl"
            ]
            
            # Remove old validation and integration logs
            remove_patterns = [
                "*validation*",
                "*integration*",
                "*production*",
                "*system_status*",
                "*enhanced_rl*",
                "training_plots.png",
                "training_metrics_*.png"
            ]
            
            for pattern in remove_patterns:
                for file in logs_dir.glob(pattern):
                    if file.is_file():
                        self._remove_file(str(file))
            
            # Remove old log directories
            old_dirs = ["enhanced_logging_demo", "training_monitor"]
            for dir_name in old_dirs:
                dir_path = logs_dir / dir_name
                if dir_path.exists():
                    self._remove_directory(str(dir_path))
    
    def _clean_redundant_checkpoints(self):
        """Clean old checkpoints, keep only recent ones."""
        print("\n💾 Cleaning checkpoint files...")
        
        checkpoints_dir = Path("checkpoints")
        if checkpoints_dir.exists():
            checkpoint_files = list(checkpoints_dir.glob("*.json"))
            
            # Keep only recent champion checkpoints
            champion_checkpoints = [f for f in checkpoint_files if "champion_stage" in f.name]
            simple_checkpoints = [f for f in checkpoint_files if "checkpoint_ep" in f.name]
            
            # Keep latest 5 champion checkpoints
            if len(champion_checkpoints) > 5:
                champion_checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                keep_checkpoints = champion_checkpoints[:5]
                remove_checkpoints = champion_checkpoints[5:]
                
                for checkpoint in remove_checkpoints:
                    self._remove_file(str(checkpoint))
            
            # Remove all simple checkpoints (outdated)
            for checkpoint in simple_checkpoints:
                self._remove_file(str(checkpoint))
    
    def _clean_development_artifacts(self):
        """Remove development and debugging artifacts."""
        print("\n🔧 Cleaning development artifacts...")
        
        # Remove development files
        dev_files = [
            'validate_core_integration.py',
            'validate_debugging_tools.py',
            'debug_enhanced_rl.py',
            'check_dependencies.py',
            'fix_pytorch_for_ultralytics.py',
            'install_real_yolo_duckietown.py',
            'setup_real_environment.sh',
            'training_analysis_20250816_000511.png'
        ]
        
        for file in dev_files:
            if Path(file).exists():
                self._remove_file(file)
        
        # Remove development directories
        dev_dirs = ['artifacts', '.vscode', '__pycache__']
        for dir_name in dev_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists():
                self._remove_directory(str(dir_path))
    
    def _clean_redundant_docs(self):
        """Keep only essential documentation."""
        print("\n📚 Cleaning documentation...")
        
        docs_dir = Path("docs")
        if docs_dir.exists():
            # Keep essential docs
            keep_docs = [
                "README.md",
                "API_Documentation.md",
                "Configuration_Guide.md",
                "Usage_Examples_and_Tutorials.md"
            ]
            
            # Remove redundant docs
            all_docs = list(docs_dir.glob("*.md"))
            for doc in all_docs:
                if doc.name not in keep_docs:
                    self._remove_file(str(doc))
                else:
                    self._keep_file(str(doc))
        
        # Remove redundant summary files
        summary_files = [
            'CLEANUP_SUMMARY.md',
            'CPU_TRAINING_SOLUTIONS.md',
            'DEPENDENCY_RESOLUTION_SUMMARY.md',
            'FINAL_INTEGRATION_SUMMARY.md',
            'INSTALL_REAL_DEPENDENCIES.md',
            'TRAINING_SUCCESS_SUMMARY.md'
        ]
        
        for file in summary_files:
            if Path(file).exists():
                self._remove_file(file)
    
    def _clean_test_files(self):
        """Remove most test files, keep only essential ones."""
        print("\n🧪 Cleaning test files...")
        
        tests_dir = Path("tests")
        if tests_dir.exists():
            # Keep only core tests
            keep_tests = [
                "test_enhanced_config.py",
                "test_enhanced_logger.py",
                "run_comprehensive_tests.py",
                "README.md",
                "pytest.ini"
            ]
            
            all_test_files = list(tests_dir.glob("*.py")) + list(tests_dir.glob("*.md")) + list(tests_dir.glob("*.ini"))
            for test_file in all_test_files:
                if test_file.name not in keep_tests:
                    self._remove_file(str(test_file))
                else:
                    self._keep_file(str(test_file))
    
    def _clean_cache_files(self):
        """Remove cache and temporary files."""
        print("\n🗑️ Cleaning cache files...")
        
        # Remove Python cache
        cache_dirs = []
        for root, dirs, files in os.walk('.'):
            for dir_name in dirs:
                if dir_name == '__pycache__':
                    cache_dirs.append(os.path.join(root, dir_name))
        
        for cache_dir in cache_dirs:
            self._remove_directory(cache_dir)
        
        # Remove other cache files
        cache_files = ['.DS_Store']
        for file in cache_files:
            if Path(file).exists():
                self._remove_file(file)
    
    def _clean_redundant_configs(self):
        """Keep only essential configuration files."""
        print("\n⚙️ Cleaning configuration files...")
        
        config_dir = Path("config")
        if config_dir.exists():
            # Keep essential configs
            keep_configs = [
                "competitive_champion_config.yml",
                "enhanced_config.py",
                "config_utils.py"
            ]
            
            # Remove redundant configs
            all_configs = list(config_dir.glob("*.yml")) + list(config_dir.glob("*.py"))
            for config in all_configs:
                if config.name not in keep_configs:
                    self._remove_file(str(config))
                else:
                    self._keep_file(str(config))
            
            # Remove config cache
            config_cache = config_dir / "__pycache__"
            if config_cache.exists():
                self._remove_directory(str(config_cache))
    
    def _clean_examples(self):
        """Keep only essential examples."""
        print("\n📝 Cleaning example files...")
        
        examples_dir = Path("examples")
        if examples_dir.exists():
            # Keep only essential examples
            keep_examples = [
                "complete_enhanced_training_example.py",
                "enhanced_config_example.py"
            ]
            
            all_examples = list(examples_dir.glob("*.py")) + list(examples_dir.glob("*.md"))
            for example in all_examples:
                if example.name not in keep_examples:
                    self._remove_file(str(example))
                else:
                    self._keep_file(str(example))
            
            # Remove examples cache
            examples_cache = examples_dir / "__pycache__"
            if examples_cache.exists():
                self._remove_directory(str(examples_cache))
    
    def _remove_file(self, file_path):
        """Remove a file and track it."""
        try:
            os.remove(file_path)
            self.removed_files.append(file_path)
            print(f"  ❌ Removed: {file_path}")
        except Exception as e:
            print(f"  ⚠️ Could not remove {file_path}: {e}")
    
    def _remove_directory(self, dir_path):
        """Remove a directory and track it."""
        try:
            shutil.rmtree(dir_path)
            self.removed_dirs.append(dir_path)
            print(f"  ❌ Removed directory: {dir_path}")
        except Exception as e:
            print(f"  ⚠️ Could not remove directory {dir_path}: {e}")
    
    def _keep_file(self, file_path):
        """Track kept files."""
        self.kept_files.append(file_path)
        print(f"  ✅ Kept: {file_path}")
    
    def _generate_cleanup_report(self):
        """Generate comprehensive cleanup report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report_data = {
            'timestamp': timestamp,
            'cleanup_type': 'comprehensive_project_cleanup',
            'files_removed': len(self.removed_files),
            'directories_removed': len(self.removed_dirs),
            'files_kept': len(self.kept_files),
            'removed_files_list': self.removed_files,
            'removed_directories_list': self.removed_dirs,
            'kept_files_list': self.kept_files
        }
        
        # Save cleanup report
        report_path = f"PROJECT_CLEANUP_REPORT_{timestamp}.md"
        
        with open(report_path, 'w') as f:
            f.write(f"# 🧹 PROJECT CLEANUP REPORT 🧹\n\n")
            f.write(f"**Cleanup Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## 📊 CLEANUP SUMMARY\n\n")
            f.write(f"- **Files Removed**: {len(self.removed_files)}\n")
            f.write(f"- **Directories Removed**: {len(self.removed_dirs)}\n")
            f.write(f"- **Essential Files Kept**: {len(self.kept_files)}\n\n")
            
            f.write(f"## ❌ REMOVED FILES\n\n")
            for file in self.removed_files:
                f.write(f"- {file}\n")
            
            f.write(f"\n## ❌ REMOVED DIRECTORIES\n\n")
            for dir in self.removed_dirs:
                f.write(f"- {dir}\n")
            
            f.write(f"\n## ✅ ESSENTIAL FILES KEPT\n\n")
            for file in self.kept_files:
                f.write(f"- {file}\n")
            
            f.write(f"\n## 🎯 PROJECT STRUCTURE AFTER CLEANUP\n\n")
            f.write(f"The project now contains only essential components for the Ultimate Champion Training System:\n\n")
            f.write(f"- **Core Training**: Ultimate Champion Training System\n")
            f.write(f"- **Best Models**: Latest champion models only\n")
            f.write(f"- **Essential Config**: Champion training configuration\n")
            f.write(f"- **Core Utils**: Essential utilities and wrappers\n")
            f.write(f"- **Key Documentation**: Essential docs and guides\n")
            f.write(f"- **Recent Logs**: Champion training logs only\n\n")
            f.write(f"**🏆 Ready for champion-level autonomous driving training!**\n")
        
        print(f"\n📋 Cleanup report saved: {report_path}")

def main():
    """Main cleanup function."""
    print("🧹 ENHANCED DUCKIETOWN RL - PROJECT CLEANUP")
    print("=" * 60)
    print("🎯 OBJECTIVE: Clean unnecessary files and keep only essentials")
    print("🏆 FOCUS: Ultimate Champion Training System")
    print("=" * 60)
    
    # Confirm cleanup
    response = input("\n⚠️ This will permanently delete files. Continue? (y/N): ")
    if response.lower() != 'y':
        print("❌ Cleanup cancelled.")
        return
    
    # Execute cleanup
    cleaner = ProjectCleaner()
    cleaner.clean_project()
    
    print("\n🎉 PROJECT CLEANUP COMPLETED!")
    print("🏆 Project optimized for Ultimate Champion Training System")
    print("📁 Check PROJECT_CLEANUP_REPORT_*.md for details")

if __name__ == "__main__":
    main()