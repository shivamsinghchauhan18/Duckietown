#!/usr/bin/env python3
"""
📊 EVALUATION MONITOR 📊
Real-time monitoring and progress reporting for evaluation orchestrator

This module provides:
- Real-time progress monitoring with rich terminal UI
- Web-based monitoring dashboard
- Progress notifications and alerts
- Performance metrics tracking
- Resource utilization monitoring

Usage Examples:
    # Terminal-based monitoring
    python evaluation_monitor.py terminal --follow
    
    # Web dashboard
    python evaluation_monitor.py web --port 8080
    
    # Export progress report
    python evaluation_monitor.py report --output progress_report.html
    
    # Monitor with alerts
    python evaluation_monitor.py terminal --alerts --email user@example.com
"""

import os
import sys
import json
import time
import argparse
import logging
import threading
import psutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
import signal

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from duckietown_utils.evaluation_orchestrator import EvaluationOrchestrator, EvaluationProgress
from evaluation_cli import EvaluationCLI

# Rich terminal UI (optional dependency)
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.live import Live
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None

# Flask for web dashboard (optional dependency)
try:
    from flask import Flask, render_template_string, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """Configuration for evaluation monitoring."""
    update_interval_seconds: float = 2.0
    history_length: int = 100
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'failure_rate': 0.1,  # 10% failure rate
        'memory_usage_gb': 8.0,  # 8GB memory usage
        'cpu_usage_percent': 80.0,  # 80% CPU usage
        'evaluation_timeout_hours': 24.0  # 24 hour timeout
    })
    
    # Notification settings
    enable_alerts: bool = False
    email_notifications: List[str] = field(default_factory=list)
    webhook_url: Optional[str] = None
    
    # Web dashboard settings
    web_port: int = 8080
    web_host: str = "localhost"
    auto_refresh_seconds: int = 5


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_usage_percent: float
    network_io_mb: float
    gpu_usage_percent: Optional[float] = None
    gpu_memory_percent: Optional[float] = None


@dataclass
class EvaluationMetrics:
    """Evaluation-specific metrics."""
    timestamp: str
    total_tasks: int
    completed_tasks: int
    running_tasks: int
    failed_tasks: int
    pending_tasks: int
    overall_progress: float
    estimated_time_remaining: Optional[float]
    average_task_duration: Optional[float]
    failure_rate: float
    throughput_tasks_per_hour: float


class SystemMonitor:
    """Monitor system resources during evaluation."""
    
    def __init__(self):
        self.metrics_history: List[SystemMetrics] = []
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
    
    def start_monitoring(self, interval_seconds: float = 2.0):
        """Start system monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval_seconds,)
        )
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
    
    def _monitor_loop(self, interval_seconds: float):
        """System monitoring loop."""
        while self._monitoring:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only recent history
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                time.sleep(interval_seconds)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        # Network I/O (simplified)
        network = psutil.net_io_counters()
        network_io_mb = (network.bytes_sent + network.bytes_recv) / (1024 * 1024)
        
        # GPU metrics (if available)
        gpu_usage = None
        gpu_memory = None
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                gpu_usage = gpu.load * 100
                gpu_memory = gpu.memoryUtil * 100
        except ImportError:
            pass
        
        return SystemMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3),
            disk_usage_percent=disk.percent,
            network_io_mb=network_io_mb,
            gpu_usage_percent=gpu_usage,
            gpu_memory_percent=gpu_memory
        )
    
    def get_latest_metrics(self) -> Optional[SystemMetrics]:
        """Get the latest system metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, minutes: int = 60) -> List[SystemMetrics]:
        """Get system metrics history for the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        return [
            metrics for metrics in self.metrics_history
            if datetime.fromisoformat(metrics.timestamp) >= cutoff_time
        ]


class EvaluationMonitor:
    """Monitor evaluation progress and performance."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.cli = EvaluationCLI()
        self.system_monitor = SystemMonitor()
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Metrics history
        self.evaluation_history: List[EvaluationMetrics] = []
        self.alert_history: List[Dict[str, Any]] = []
        
        # Callbacks
        self._progress_callbacks: List[Callable[[EvaluationProgress], None]] = []
        self._alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
    
    def start_monitoring(self):
        """Start evaluation monitoring."""
        if self._monitoring:
            return
        
        logger.info("📊 Starting evaluation monitoring")
        
        self._monitoring = True
        self.system_monitor.start_monitoring(self.config.update_interval_seconds)
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop evaluation monitoring."""
        logger.info("📊 Stopping evaluation monitoring")
        
        self._monitoring = False
        self.system_monitor.stop_monitoring()
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
    
    def add_progress_callback(self, callback: Callable[[EvaluationProgress], None]):
        """Add progress update callback."""
        self._progress_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add alert callback."""
        self._alert_callbacks.append(callback)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        orchestrator = self.cli._get_orchestrator()
        start_time = time.time()
        task_completion_times = []
        
        while self._monitoring:
            try:
                # Get evaluation progress
                progress = orchestrator.get_progress()
                
                # Calculate additional metrics
                current_time = time.time()
                elapsed_hours = (current_time - start_time) / 3600
                
                # Calculate throughput
                throughput = progress.completed_tasks / max(elapsed_hours, 0.001)
                
                # Calculate failure rate
                total_finished = progress.completed_tasks + progress.failed_tasks
                failure_rate = progress.failed_tasks / max(total_finished, 1)
                
                # Create evaluation metrics
                eval_metrics = EvaluationMetrics(
                    timestamp=datetime.now().isoformat(),
                    total_tasks=progress.total_tasks,
                    completed_tasks=progress.completed_tasks,
                    running_tasks=progress.running_tasks,
                    failed_tasks=progress.failed_tasks,
                    pending_tasks=progress.pending_tasks,
                    overall_progress=progress.overall_progress,
                    estimated_time_remaining=progress.estimated_time_remaining,
                    average_task_duration=None,  # Could be calculated from history
                    failure_rate=failure_rate,
                    throughput_tasks_per_hour=throughput
                )
                
                # Store metrics
                self.evaluation_history.append(eval_metrics)
                
                # Keep only recent history
                if len(self.evaluation_history) > self.config.history_length:
                    self.evaluation_history = self.evaluation_history[-self.config.history_length:]
                
                # Trigger progress callbacks
                for callback in self._progress_callbacks:
                    try:
                        callback(progress)
                    except Exception as e:
                        logger.error(f"Progress callback error: {e}")
                
                # Check for alerts
                if self.config.enable_alerts:
                    self._check_alerts(eval_metrics)
                
                time.sleep(self.config.update_interval_seconds)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.config.update_interval_seconds)
    
    def _check_alerts(self, eval_metrics: EvaluationMetrics):
        """Check for alert conditions."""
        alerts = []
        
        # Check failure rate
        if eval_metrics.failure_rate > self.config.alert_thresholds['failure_rate']:
            alerts.append({
                'type': 'high_failure_rate',
                'message': f"High failure rate: {eval_metrics.failure_rate:.1%}",
                'severity': 'warning',
                'value': eval_metrics.failure_rate,
                'threshold': self.config.alert_thresholds['failure_rate']
            })
        
        # Check system resources
        system_metrics = self.system_monitor.get_latest_metrics()
        if system_metrics:
            if system_metrics.memory_used_gb > self.config.alert_thresholds['memory_usage_gb']:
                alerts.append({
                    'type': 'high_memory_usage',
                    'message': f"High memory usage: {system_metrics.memory_used_gb:.1f}GB",
                    'severity': 'warning',
                    'value': system_metrics.memory_used_gb,
                    'threshold': self.config.alert_thresholds['memory_usage_gb']
                })
            
            if system_metrics.cpu_percent > self.config.alert_thresholds['cpu_usage_percent']:
                alerts.append({
                    'type': 'high_cpu_usage',
                    'message': f"High CPU usage: {system_metrics.cpu_percent:.1f}%",
                    'severity': 'warning',
                    'value': system_metrics.cpu_percent,
                    'threshold': self.config.alert_thresholds['cpu_usage_percent']
                })
        
        # Trigger alerts
        for alert in alerts:
            self._trigger_alert(alert)
    
    def _trigger_alert(self, alert: Dict[str, Any]):
        """Trigger an alert."""
        alert['timestamp'] = datetime.now().isoformat()
        self.alert_history.append(alert)
        
        logger.warning(f"ALERT: {alert['message']}")
        
        # Trigger alert callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert['type'], alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        latest_eval = self.evaluation_history[-1] if self.evaluation_history else None
        latest_system = self.system_monitor.get_latest_metrics()
        
        return {
            'monitoring_active': self._monitoring,
            'timestamp': datetime.now().isoformat(),
            'evaluation_metrics': latest_eval.__dict__ if latest_eval else None,
            'system_metrics': latest_system.__dict__ if latest_system else None,
            'recent_alerts': self.alert_history[-10:],  # Last 10 alerts
            'history_length': len(self.evaluation_history)
        }


class TerminalMonitor:
    """Rich terminal-based monitoring interface."""
    
    def __init__(self, monitor: EvaluationMonitor):
        self.monitor = monitor
        self.console = Console() if RICH_AVAILABLE else None
        
        if not RICH_AVAILABLE:
            logger.warning("Rich library not available. Using basic terminal output.")
    
    def start_terminal_monitoring(self, follow: bool = True):
        """Start terminal-based monitoring."""
        if not RICH_AVAILABLE:
            self._basic_terminal_monitoring(follow)
            return
        
        self.monitor.start_monitoring()
        
        if follow:
            self._rich_live_monitoring()
        else:
            self._rich_snapshot()
    
    def _rich_live_monitoring(self):
        """Rich live monitoring interface."""
        def generate_layout():
            layout = Layout()
            
            # Split into sections
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="main"),
                Layout(name="footer", size=5)
            )
            
            layout["main"].split_row(
                Layout(name="progress"),
                Layout(name="system")
            )
            
            # Header
            layout["header"].update(
                Panel(
                    Text("🏆 Enhanced Duckietown RL - Evaluation Monitor", style="bold blue"),
                    title="Evaluation Orchestrator"
                )
            )
            
            # Progress section
            status = self.monitor.get_current_status()
            eval_metrics = status.get('evaluation_metrics')
            
            if eval_metrics:
                progress_table = Table(title="Evaluation Progress")
                progress_table.add_column("Metric", style="cyan")
                progress_table.add_column("Value", style="green")
                
                progress_table.add_row("Total Tasks", str(eval_metrics['total_tasks']))
                progress_table.add_row("Completed", str(eval_metrics['completed_tasks']))
                progress_table.add_row("Running", str(eval_metrics['running_tasks']))
                progress_table.add_row("Failed", str(eval_metrics['failed_tasks']))
                progress_table.add_row("Progress", f"{eval_metrics['overall_progress']:.1f}%")
                progress_table.add_row("Failure Rate", f"{eval_metrics['failure_rate']:.1%}")
                progress_table.add_row("Throughput", f"{eval_metrics['throughput_tasks_per_hour']:.1f} tasks/hour")
                
                if eval_metrics['estimated_time_remaining']:
                    eta_hours = eval_metrics['estimated_time_remaining'] / 3600
                    progress_table.add_row("ETA", f"{eta_hours:.1f} hours")
                
                layout["progress"].update(Panel(progress_table))
            else:
                layout["progress"].update(Panel("No evaluation data available"))
            
            # System section
            system_metrics = status.get('system_metrics')
            
            if system_metrics:
                system_table = Table(title="System Resources")
                system_table.add_column("Resource", style="cyan")
                system_table.add_column("Usage", style="yellow")
                
                system_table.add_row("CPU", f"{system_metrics['cpu_percent']:.1f}%")
                system_table.add_row("Memory", f"{system_metrics['memory_percent']:.1f}% ({system_metrics['memory_used_gb']:.1f}GB)")
                system_table.add_row("Disk", f"{system_metrics['disk_usage_percent']:.1f}%")
                
                if system_metrics['gpu_usage_percent'] is not None:
                    system_table.add_row("GPU", f"{system_metrics['gpu_usage_percent']:.1f}%")
                    system_table.add_row("GPU Memory", f"{system_metrics['gpu_memory_percent']:.1f}%")
                
                layout["system"].update(Panel(system_table))
            else:
                layout["system"].update(Panel("No system data available"))
            
            # Footer with alerts
            recent_alerts = status.get('recent_alerts', [])
            if recent_alerts:
                alert_text = "\n".join([
                    f"⚠️  {alert['message']} ({alert['timestamp']})"
                    for alert in recent_alerts[-3:]  # Show last 3 alerts
                ])
                layout["footer"].update(Panel(alert_text, title="Recent Alerts", border_style="red"))
            else:
                layout["footer"].update(Panel("No recent alerts", title="Status", border_style="green"))
            
            return layout
        
        # Live monitoring
        try:
            with Live(generate_layout(), refresh_per_second=0.5, screen=True) as live:
                while True:
                    live.update(generate_layout())
                    time.sleep(2)
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Monitoring stopped by user[/yellow]")
        finally:
            self.monitor.stop_monitoring()
    
    def _rich_snapshot(self):
        """Rich snapshot of current status."""
        status = self.monitor.get_current_status()
        
        # Create and display tables
        eval_metrics = status.get('evaluation_metrics')
        if eval_metrics:
            table = Table(title="Evaluation Status")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            for key, value in eval_metrics.items():
                if key != 'timestamp':
                    table.add_row(key.replace('_', ' ').title(), str(value))
            
            self.console.print(table)
        
        system_metrics = status.get('system_metrics')
        if system_metrics:
            table = Table(title="System Resources")
            table.add_column("Resource", style="cyan")
            table.add_column("Value", style="yellow")
            
            table.add_row("CPU Usage", f"{system_metrics['cpu_percent']:.1f}%")
            table.add_row("Memory Usage", f"{system_metrics['memory_percent']:.1f}%")
            table.add_row("Memory Used", f"{system_metrics['memory_used_gb']:.1f}GB")
            
            self.console.print(table)
    
    def _basic_terminal_monitoring(self, follow: bool):
        """Basic terminal monitoring without Rich."""
        self.monitor.start_monitoring()
        
        try:
            if follow:
                while True:
                    status = self.monitor.get_current_status()
                    eval_metrics = status.get('evaluation_metrics')
                    
                    if eval_metrics:
                        print(f"\r📊 Progress: {eval_metrics['overall_progress']:.1f}% | "
                              f"✅ {eval_metrics['completed_tasks']} | "
                              f"🏃 {eval_metrics['running_tasks']} | "
                              f"❌ {eval_metrics['failed_tasks']} | "
                              f"⏳ {eval_metrics['pending_tasks']}", end="", flush=True)
                    
                    time.sleep(2)
            else:
                status = self.monitor.get_current_status()
                print(json.dumps(status, indent=2))
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        finally:
            self.monitor.stop_monitoring()


class WebDashboard:
    """Web-based monitoring dashboard."""
    
    def __init__(self, monitor: EvaluationMonitor, config: MonitoringConfig):
        self.monitor = monitor
        self.config = config
        
        if not FLASK_AVAILABLE:
            raise ImportError("Flask is required for web dashboard")
        
        self.app = Flask(__name__)
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def dashboard():
            return render_template_string(self._get_dashboard_template())
        
        @self.app.route('/api/status')
        def api_status():
            return jsonify(self.monitor.get_current_status())
        
        @self.app.route('/api/history/<int:minutes>')
        def api_history(minutes):
            # Get evaluation history
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            
            eval_history = [
                metrics.__dict__ for metrics in self.monitor.evaluation_history
                if datetime.fromisoformat(metrics.timestamp) >= cutoff_time
            ]
            
            system_history = [
                metrics.__dict__ for metrics in self.monitor.system_monitor.get_metrics_history(minutes)
            ]
            
            return jsonify({
                'evaluation_history': eval_history,
                'system_history': system_history
            })
    
    def _get_dashboard_template(self) -> str:
        """Get HTML template for dashboard."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Evaluation Monitor Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
                .card { border: 1px solid #ddd; padding: 20px; border-radius: 8px; }
                .metric { font-size: 24px; font-weight: bold; color: #2196F3; }
                .chart-container { height: 300px; }
                .alert { background: #ffebee; border: 1px solid #f44336; padding: 10px; margin: 10px 0; border-radius: 4px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏆 Evaluation Monitor Dashboard</h1>
                
                <div class="grid">
                    <div class="card">
                        <h3>Evaluation Progress</h3>
                        <div id="progress-metrics">Loading...</div>
                    </div>
                    
                    <div class="card">
                        <h3>System Resources</h3>
                        <div id="system-metrics">Loading...</div>
                    </div>
                </div>
                
                <div class="card">
                    <h3>Progress Chart</h3>
                    <div class="chart-container">
                        <canvas id="progressChart"></canvas>
                    </div>
                </div>
                
                <div class="card">
                    <h3>Recent Alerts</h3>
                    <div id="alerts">Loading...</div>
                </div>
            </div>
            
            <script>
                // Auto-refresh data
                function updateDashboard() {
                    fetch('/api/status')
                        .then(response => response.json())
                        .then(data => {
                            updateProgressMetrics(data.evaluation_metrics);
                            updateSystemMetrics(data.system_metrics);
                            updateAlerts(data.recent_alerts);
                        });
                }
                
                function updateProgressMetrics(metrics) {
                    if (!metrics) return;
                    
                    document.getElementById('progress-metrics').innerHTML = `
                        <div class="metric">${metrics.overall_progress.toFixed(1)}%</div>
                        <p>Completed: ${metrics.completed_tasks} / ${metrics.total_tasks}</p>
                        <p>Running: ${metrics.running_tasks}</p>
                        <p>Failed: ${metrics.failed_tasks}</p>
                        <p>Failure Rate: ${(metrics.failure_rate * 100).toFixed(1)}%</p>
                        <p>Throughput: ${metrics.throughput_tasks_per_hour.toFixed(1)} tasks/hour</p>
                    `;
                }
                
                function updateSystemMetrics(metrics) {
                    if (!metrics) return;
                    
                    document.getElementById('system-metrics').innerHTML = `
                        <p>CPU: ${metrics.cpu_percent.toFixed(1)}%</p>
                        <p>Memory: ${metrics.memory_percent.toFixed(1)}% (${metrics.memory_used_gb.toFixed(1)}GB)</p>
                        <p>Disk: ${metrics.disk_usage_percent.toFixed(1)}%</p>
                        ${metrics.gpu_usage_percent ? `<p>GPU: ${metrics.gpu_usage_percent.toFixed(1)}%</p>` : ''}
                    `;
                }
                
                function updateAlerts(alerts) {
                    if (!alerts || alerts.length === 0) {
                        document.getElementById('alerts').innerHTML = '<p>No recent alerts</p>';
                        return;
                    }
                    
                    const alertsHtml = alerts.map(alert => 
                        `<div class="alert">${alert.message} (${alert.timestamp})</div>`
                    ).join('');
                    
                    document.getElementById('alerts').innerHTML = alertsHtml;
                }
                
                // Initialize and start auto-refresh
                updateDashboard();
                setInterval(updateDashboard, """ + str(self.config.auto_refresh_seconds * 1000) + """);
            </script>
        </body>
        </html>
        """
    
    def start_server(self):
        """Start the web dashboard server."""
        self.monitor.start_monitoring()
        
        logger.info(f"🌐 Starting web dashboard at http://{self.config.web_host}:{self.config.web_port}")
        
        self.app.run(
            host=self.config.web_host,
            port=self.config.web_port,
            debug=False,
            threaded=True
        )


def create_monitor_parser() -> argparse.ArgumentParser:
    """Create argument parser for evaluation monitor."""
    parser = argparse.ArgumentParser(
        description="Evaluation Progress Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Terminal monitoring
    terminal_parser = subparsers.add_parser('terminal', help='Terminal-based monitoring')
    terminal_parser.add_argument('--follow', action='store_true', help='Continuously monitor')
    terminal_parser.add_argument('--alerts', action='store_true', help='Enable alerts')
    terminal_parser.add_argument('--email', nargs='*', help='Email addresses for alerts')
    
    # Web dashboard
    web_parser = subparsers.add_parser('web', help='Web-based dashboard')
    web_parser.add_argument('--port', type=int, default=8080, help='Web server port')
    web_parser.add_argument('--host', default='localhost', help='Web server host')
    
    # Report generation
    report_parser = subparsers.add_parser('report', help='Generate progress report')
    report_parser.add_argument('--output', required=True, help='Output file path')
    report_parser.add_argument('--format', choices=['html', 'json'], default='html', help='Report format')
    
    return parser


def main():
    """Main entry point for evaluation monitor."""
    parser = create_monitor_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Create monitoring configuration
    config = MonitoringConfig()
    
    if hasattr(args, 'alerts') and args.alerts:
        config.enable_alerts = True
        if hasattr(args, 'email') and args.email:
            config.email_notifications = args.email
    
    if hasattr(args, 'port'):
        config.web_port = args.port
    if hasattr(args, 'host'):
        config.web_host = args.host
    
    # Create monitor
    monitor = EvaluationMonitor(config)
    
    try:
        if args.command == 'terminal':
            terminal_monitor = TerminalMonitor(monitor)
            terminal_monitor.start_terminal_monitoring(follow=args.follow)
            
        elif args.command == 'web':
            if not FLASK_AVAILABLE:
                logger.error("Flask is required for web dashboard. Install with: pip install flask")
                return 1
            
            dashboard = WebDashboard(monitor, config)
            dashboard.start_server()
            
        elif args.command == 'report':
            # Generate progress report
            monitor.start_monitoring()
            time.sleep(5)  # Collect some data
            
            status = monitor.get_current_status()
            
            if args.format == 'json':
                with open(args.output, 'w') as f:
                    json.dump(status, f, indent=2)
            else:  # HTML
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head><title>Evaluation Progress Report</title></head>
                <body>
                    <h1>Evaluation Progress Report</h1>
                    <p>Generated: {datetime.now().isoformat()}</p>
                    <pre>{json.dumps(status, indent=2)}</pre>
                </body>
                </html>
                """
                with open(args.output, 'w') as f:
                    f.write(html_content)
            
            monitor.stop_monitoring()
            print(f"Report saved to: {args.output}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())