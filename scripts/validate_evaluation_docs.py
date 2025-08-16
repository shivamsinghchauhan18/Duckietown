#!/usr/bin/env python3
"""
Documentation validation script for the Enhanced Duckietown RL evaluation system.

This script validates that all evaluation documentation files exist, are properly
formatted, and contain the expected content sections.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re


def check_file_exists(file_path: str) -> bool:
    """Check if a file exists."""
    return os.path.exists(file_path)


def check_markdown_structure(file_path: str) -> Tuple[bool, List[str]]:
    """Check if markdown file has proper structure."""
    issues = []
    
    if not os.path.exists(file_path):
        return False, [f"File does not exist: {file_path}"]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return False, [f"Error reading file: {e}"]
    
    # Check for basic markdown structure
    if not content.strip():
        issues.append("File is empty")
    
    # Check for title (# at start of line)
    if not re.search(r'^# .+', content, re.MULTILINE):
        issues.append("Missing main title (# heading)")
    
    # Check for overview section
    if 'overview' not in content.lower():
        issues.append("Missing Overview section")
    
    # Check for code blocks (should have proper formatting)
    code_blocks = re.findall(r'```[\s\S]*?```', content)
    for i, block in enumerate(code_blocks):
        if not block.strip().endswith('```'):
            issues.append(f"Malformed code block {i+1}")
    
    return len(issues) == 0, issues


def check_internal_links(file_path: str, docs_dir: str) -> Tuple[bool, List[str]]:
    """Check if internal markdown links are valid."""
    issues = []
    
    if not os.path.exists(file_path):
        return False, [f"File does not exist: {file_path}"]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return False, [f"Error reading file: {e}"]
    
    # Find markdown links [text](link)
    links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
    
    for link_text, link_url in links:
        # Skip external links (http/https)
        if link_url.startswith(('http://', 'https://')):
            continue
        
        # Skip anchors within same document
        if link_url.startswith('#'):
            continue
        
        # Check if relative link exists
        if link_url.startswith('../'):
            # Handle relative paths
            full_path = os.path.join(os.path.dirname(file_path), link_url)
            full_path = os.path.normpath(full_path)
        else:
            # Assume it's in docs directory
            full_path = os.path.join(docs_dir, link_url)
        
        if not os.path.exists(full_path):
            issues.append(f"Broken link: [{link_text}]({link_url}) -> {full_path}")
    
    return len(issues) == 0, issues


def validate_api_documentation(file_path: str) -> Tuple[bool, List[str]]:
    """Validate API documentation specific requirements."""
    issues = []
    
    if not os.path.exists(file_path):
        return False, [f"File does not exist: {file_path}"]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return False, [f"Error reading file: {e}"]
    
    # Check for required sections
    required_sections = [
        'Core Components',
        'EvaluationOrchestrator',
        'SuiteManager',
        'MetricsCalculator',
        'StatisticalAnalyzer',
        'FailureAnalyzer',
        'RobustnessAnalyzer',
        'ChampionSelector',
        'ReportGenerator',
        'Data Models',
        'Usage Examples',
        'Error Handling'
    ]
    
    for section in required_sections:
        if section.lower() not in content.lower():
            issues.append(f"Missing required section: {section}")
    
    # Check for code examples
    python_blocks = re.findall(r'```python[\s\S]*?```', content)
    if len(python_blocks) < 10:
        issues.append(f"Insufficient Python code examples (found {len(python_blocks)}, expected ≥10)")
    
    return len(issues) == 0, issues


def validate_configuration_guide(file_path: str) -> Tuple[bool, List[str]]:
    """Validate configuration guide specific requirements."""
    issues = []
    
    if not os.path.exists(file_path):
        return False, [f"File does not exist: {file_path}"]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return False, [f"Error reading file: {e}"]
    
    # Check for required sections
    required_sections = [
        'Configuration Structure',
        'Suite Configuration',
        'Metrics Configuration',
        'Scoring Configuration',
        'Artifact Configuration',
        'Reproducibility Configuration',
        'Configuration Templates',
        'Configuration Validation',
        'Best Practices'
    ]
    
    for section in required_sections:
        if section.lower() not in content.lower():
            issues.append(f"Missing required section: {section}")
    
    # Check for YAML examples
    yaml_blocks = re.findall(r'```yaml[\s\S]*?```', content)
    if len(yaml_blocks) < 5:
        issues.append(f"Insufficient YAML examples (found {len(yaml_blocks)}, expected ≥5)")
    
    return len(issues) == 0, issues


def validate_examples_file(file_path: str) -> Tuple[bool, List[str]]:
    """Validate examples file specific requirements."""
    issues = []
    
    if not os.path.exists(file_path):
        return False, [f"File does not exist: {file_path}"]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return False, [f"Error reading file: {e}"]
    
    # Check for required example functions
    required_examples = [
        'example_1_basic_model_evaluation',
        'example_2_multi_model_comparison',
        'example_3_champion_selection',
        'example_4_robustness_analysis',
        'example_5_deployment_readiness',
        'example_6_continuous_evaluation',
        'example_7_custom_metrics_evaluation'
    ]
    
    for example in required_examples:
        if example not in content:
            issues.append(f"Missing required example function: {example}")
    
    # Check for proper docstrings
    function_pattern = r'def (example_\d+_[^(]+)\([^)]*\):\s*"""([^"]+)"""'
    functions = re.findall(function_pattern, content, re.DOTALL)
    
    if len(functions) < 7:
        issues.append(f"Insufficient documented example functions (found {len(functions)}, expected ≥7)")
    
    return len(issues) == 0, issues


def validate_troubleshooting_guide(file_path: str) -> Tuple[bool, List[str]]:
    """Validate troubleshooting guide specific requirements."""
    issues = []
    
    if not os.path.exists(file_path):
        return False, [f"File does not exist: {file_path}"]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return False, [f"Error reading file: {e}"]
    
    # Check for required sections
    required_sections = [
        'Quick Diagnostic Checklist',
        'Installation Issues',
        'Configuration Issues',
        'Runtime Issues',
        'Performance Issues',
        'Output and Reporting Issues',
        'Environment-Specific Issues',
        'Debugging Tools',
        'Getting Help'
    ]
    
    for section in required_sections:
        if section.lower() not in content.lower():
            issues.append(f"Missing required section: {section}")
    
    # Check for symptom/solution patterns
    symptom_count = len(re.findall(r'#### Symptom:', content))
    solution_count = len(re.findall(r'#### Solutions?:', content))
    
    if symptom_count < 10:
        issues.append(f"Insufficient symptom descriptions (found {symptom_count}, expected ≥10)")
    
    if solution_count < 10:
        issues.append(f"Insufficient solution sections (found {solution_count}, expected ≥10)")
    
    return len(issues) == 0, issues


def main():
    """Main validation function."""
    print("Validating Enhanced Duckietown RL Evaluation Documentation")
    print("=" * 60)
    
    # Define documentation files to validate
    docs_dir = 'docs'
    examples_dir = 'examples'
    
    documentation_files = {
        'API Documentation': {
            'path': os.path.join(docs_dir, 'Evaluation_Orchestrator_API_Documentation.md'),
            'validator': validate_api_documentation
        },
        'Configuration Guide': {
            'path': os.path.join(docs_dir, 'Evaluation_Configuration_Guide.md'),
            'validator': validate_configuration_guide
        },
        'Result Interpretation Guide': {
            'path': os.path.join(docs_dir, 'Evaluation_Result_Interpretation_Guide.md'),
            'validator': check_markdown_structure
        },
        'Troubleshooting Guide': {
            'path': os.path.join(docs_dir, 'Evaluation_Troubleshooting_Guide.md'),
            'validator': validate_troubleshooting_guide
        },
        'Documentation Index': {
            'path': os.path.join(docs_dir, 'Evaluation_Documentation_Index.md'),
            'validator': check_markdown_structure
        },
        'Evaluation Examples': {
            'path': os.path.join(examples_dir, 'evaluation_examples.py'),
            'validator': validate_examples_file
        }
    }
    
    all_valid = True
    total_issues = 0
    
    for doc_name, doc_info in documentation_files.items():
        print(f"\nValidating {doc_name}...")
        print("-" * 40)
        
        file_path = doc_info['path']
        validator = doc_info['validator']
        
        # Check if file exists
        if not check_file_exists(file_path):
            print(f"❌ FAIL: File does not exist: {file_path}")
            all_valid = False
            total_issues += 1
            continue
        
        # Run specific validator
        is_valid, issues = validator(file_path)
        
        if is_valid:
            print(f"✅ PASS: {doc_name}")
        else:
            print(f"❌ FAIL: {doc_name}")
            for issue in issues:
                print(f"  - {issue}")
            all_valid = False
            total_issues += len(issues)
        
        # Check internal links for markdown files
        if file_path.endswith('.md'):
            links_valid, link_issues = check_internal_links(file_path, docs_dir)
            if not links_valid:
                print(f"⚠️  WARN: Link issues in {doc_name}")
                for issue in link_issues:
                    print(f"  - {issue}")
                total_issues += len(link_issues)
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    if all_valid and total_issues == 0:
        print("✅ ALL DOCUMENTATION VALID")
        print("All documentation files exist and meet requirements.")
        return 0
    else:
        print(f"❌ VALIDATION FAILED")
        print(f"Total issues found: {total_issues}")
        print("\nPlease fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())