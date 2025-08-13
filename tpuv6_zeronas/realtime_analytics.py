"""Real-time Analytics Engine: Advanced monitoring, predictive analytics, and intelligent optimization."""

import logging
import time
import json
import math
from typing import Dict, List, Tuple, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path
from enum import Enum
import queue
import statistics
from collections import defaultdict, deque

from .architecture import Architecture, ArchitectureSpace
from .metrics import PerformanceMetrics
from .predictor import TPUv6Predictor
from .core import SearchConfig


class MetricType(Enum):
    """Types of metrics tracked by analytics engine."""
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    QUALITY = "quality"
    BUSINESS = "business"
    SECURITY = "security"
    COMPLIANCE = "compliance"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AnomalyType(Enum):
    """Types of anomalies detected."""
    STATISTICAL = "statistical"
    PATTERN = "pattern"
    THRESHOLD = "threshold"
    TREND = "trend"
    SEASONAL = "seasonal"


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: float
    timestamp: float
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Time series of metric values."""
    name: str
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    unit: str = ""
    metric_type: MetricType = MetricType.PERFORMANCE
    retention_seconds: int = 3600  # 1 hour default
    
    def add_value(self, value: float, timestamp: Optional[float] = None):
        """Add value to time series."""
        if timestamp is None:
            timestamp = time.time()
        
        metric = Metric(name=self.name, value=value, timestamp=timestamp, unit=self.unit)
        self.values.append(metric)
        
        # Clean old values beyond retention
        cutoff_time = time.time() - self.retention_seconds
        while self.values and self.values[0].timestamp < cutoff_time:
            self.values.popleft()
    
    def get_latest(self) -> Optional[Metric]:
        """Get latest metric value."""
        return self.values[-1] if self.values else None
    
    def get_values_in_range(self, start_time: float, end_time: float) -> List[Metric]:
        """Get metrics within time range."""
        return [m for m in self.values if start_time <= m.timestamp <= end_time]


@dataclass
class Alert:
    """Alert/notification object."""
    alert_id: str
    name: str
    severity: AlertSeverity
    message: str
    timestamp: float
    metric_name: str
    threshold_value: Optional[float] = None
    actual_value: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False
    escalated: bool = False


@dataclass
class AnomalyDetection:
    """Anomaly detection configuration and state."""
    metric_name: str
    detection_type: AnomalyType
    sensitivity: float = 0.95  # 95% confidence
    window_size: int = 50
    threshold_multiplier: float = 3.0
    baseline_values: deque = field(default_factory=lambda: deque(maxlen=100))
    anomalies_detected: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Dashboard:
    """Analytics dashboard configuration."""
    dashboard_id: str
    name: str
    metrics: List[str] = field(default_factory=list)
    charts: List[Dict[str, Any]] = field(default_factory=list)
    refresh_interval_seconds: int = 30
    alert_widgets: List[str] = field(default_factory=list)
    custom_queries: Dict[str, str] = field(default_factory=dict)


class PredictiveAnalytics:
    """Predictive analytics for forecasting and optimization."""
    
    def __init__(self):
        self.models = {}
        self.forecasts = {}
        self.prediction_history = defaultdict(list)
        
    def train_forecasting_model(self, metric_series: MetricSeries, 
                               forecast_horizon_minutes: int = 60) -> Dict[str, Any]:
        """Train time series forecasting model."""
        if len(metric_series.values) < 10:
            return {'error': 'Insufficient data for training'}
        
        # Simple linear trend model (in practice, use ARIMA, LSTM, etc.)
        values = [m.value for m in metric_series.values]
        timestamps = [m.timestamp for m in metric_series.values]
        
        # Calculate trend
        n = len(values)
        sum_x = sum(range(n))
        sum_y = sum(values)
        sum_xy = sum(i * values[i] for i in range(n))
        sum_x2 = sum(i * i for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Calculate R-squared
        mean_y = sum_y / n
        ss_tot = sum((y - mean_y) ** 2 for y in values)
        ss_res = sum((values[i] - (slope * i + intercept)) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        model = {
            'metric_name': metric_series.name,
            'model_type': 'linear_trend',
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'training_samples': n,
            'forecast_horizon_minutes': forecast_horizon_minutes,
            'trained_at': time.time()
        }
        
        self.models[metric_series.name] = model
        return model
    
    def generate_forecast(self, metric_name: str, 
                         forecast_points: int = 12) -> List[Dict[str, Any]]:
        """Generate forecast for metric."""
        if metric_name not in self.models:
            return []
        
        model = self.models[metric_name]
        current_time = time.time()
        forecast_interval = 300  # 5-minute intervals
        
        forecast = []
        for i in range(forecast_points):
            future_time = current_time + (i + 1) * forecast_interval
            # Simple linear extrapolation
            future_index = model['training_samples'] + i + 1
            predicted_value = model['slope'] * future_index + model['intercept']
            
            # Add uncertainty bounds (simplified)
            uncertainty = abs(predicted_value) * 0.1  # 10% uncertainty
            
            forecast_point = {
                'timestamp': future_time,
                'predicted_value': predicted_value,
                'lower_bound': predicted_value - uncertainty,
                'upper_bound': predicted_value + uncertainty,
                'confidence': model['r_squared']
            }
            
            forecast.append(forecast_point)
        
        self.forecasts[metric_name] = {
            'forecast': forecast,
            'generated_at': current_time,
            'model_used': model['model_type']
        }
        
        return forecast
    
    def detect_capacity_needs(self, resource_metrics: Dict[str, MetricSeries]) -> Dict[str, Any]:
        """Predict future capacity requirements."""
        capacity_forecast = {}
        
        for metric_name, series in resource_metrics.items():
            if 'utilization' in metric_name.lower():
                # Train model and generate forecast
                self.train_forecasting_model(series, forecast_horizon_minutes=120)
                forecast = self.generate_forecast(metric_name, forecast_points=24)  # 2 hours
                
                # Find peak utilization in forecast
                peak_forecast = max(f['predicted_value'] for f in forecast) if forecast else 0
                current_utilization = series.get_latest().value if series.get_latest() else 0
                
                capacity_forecast[metric_name] = {
                    'current_utilization': current_utilization,
                    'peak_forecast': peak_forecast,
                    'scaling_recommendation': self._get_scaling_recommendation(
                        current_utilization, peak_forecast
                    ),
                    'forecast_confidence': forecast[0]['confidence'] if forecast else 0
                }
        
        return capacity_forecast
    
    def _get_scaling_recommendation(self, current: float, forecast: float) -> Dict[str, Any]:
        """Generate scaling recommendations based on forecasts."""
        if forecast > 80:
            return {
                'action': 'scale_up',
                'urgency': 'high' if forecast > 90 else 'medium',
                'recommended_increase': max(20, int((forecast - 70) / 10) * 10)
            }
        elif current > 70 and forecast < 50:
            return {
                'action': 'scale_down',
                'urgency': 'low',
                'recommended_decrease': min(30, int((70 - forecast) / 10) * 10)
            }
        else:
            return {
                'action': 'maintain',
                'urgency': 'none',
                'recommended_change': 0
            }


class RealTimeAnalyticsEngine:
    """Comprehensive real-time analytics and monitoring engine."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Metric storage and management
        self.metric_series = {}
        self.metric_buffer = queue.Queue(maxsize=10000)
        
        # Alerting system
        self.alert_rules = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        
        # Anomaly detection
        self.anomaly_detectors = {}
        self.anomaly_history = deque(maxlen=500)
        
        # Dashboards and visualization
        self.dashboards = {}
        self.custom_widgets = {}
        
        # Predictive analytics
        self.predictive_engine = PredictiveAnalytics()
        
        # Real-time processing
        self.processing_threads = []
        self.running = False
        
        # Performance optimization
        self.optimization_recommendations = deque(maxlen=100)
        self.optimization_actions = []
        
        # Initialize the engine
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize analytics engine components."""
        logging.info("Initializing real-time analytics engine")
        
        # Setup default metric series
        self._setup_default_metrics()
        
        # Setup default alert rules
        self._setup_default_alerts()
        
        # Setup anomaly detection
        self._setup_anomaly_detection()
        
        # Create default dashboards
        self._create_default_dashboards()
        
        # Start processing threads
        self._start_processing_threads()
        
        logging.info("Real-time analytics engine initialized successfully")
    
    def _setup_default_metrics(self):
        """Setup default metric series for tracking."""
        default_metrics = [
            ('search_latency_ms', MetricType.PERFORMANCE, 'milliseconds'),
            ('search_accuracy', MetricType.QUALITY, 'percentage'),
            ('search_throughput', MetricType.PERFORMANCE, 'searches/second'),
            ('cpu_utilization', MetricType.RESOURCE, 'percentage'),
            ('memory_utilization', MetricType.RESOURCE, 'percentage'),
            ('gpu_utilization', MetricType.RESOURCE, 'percentage'),
            ('energy_efficiency', MetricType.PERFORMANCE, 'TOPS/W'),
            ('error_rate', MetricType.QUALITY, 'percentage'),
            ('user_satisfaction', MetricType.BUSINESS, 'score'),
            ('cost_per_search', MetricType.BUSINESS, 'USD'),
            ('security_events', MetricType.SECURITY, 'count'),
            ('compliance_score', MetricType.COMPLIANCE, 'percentage')
        ]
        
        for name, metric_type, unit in default_metrics:
            self.metric_series[name] = MetricSeries(
                name=name,
                unit=unit,
                metric_type=metric_type,
                retention_seconds=7200  # 2 hours
            )
    
    def _setup_default_alerts(self):
        """Setup default alerting rules."""
        alert_rules = [
            {
                'name': 'high_search_latency',
                'metric': 'search_latency_ms',
                'condition': 'greater_than',
                'threshold': 1000.0,
                'severity': AlertSeverity.WARNING,
                'message': 'Search latency exceeds 1 second'
            },
            {
                'name': 'low_search_accuracy',
                'metric': 'search_accuracy',
                'condition': 'less_than',
                'threshold': 85.0,
                'severity': AlertSeverity.ERROR,
                'message': 'Search accuracy below acceptable threshold'
            },
            {
                'name': 'high_cpu_utilization',
                'metric': 'cpu_utilization',
                'condition': 'greater_than',
                'threshold': 90.0,
                'severity': AlertSeverity.WARNING,
                'message': 'CPU utilization is critically high'
            },
            {
                'name': 'high_error_rate',
                'metric': 'error_rate',
                'condition': 'greater_than',
                'threshold': 5.0,
                'severity': AlertSeverity.CRITICAL,
                'message': 'Error rate exceeds acceptable limits'
            },
            {
                'name': 'security_anomaly',
                'metric': 'security_events',
                'condition': 'spike',
                'threshold': 3.0,  # 3x normal rate
                'severity': AlertSeverity.CRITICAL,
                'message': 'Unusual spike in security events detected'
            }
        ]
        
        for rule in alert_rules:
            self.alert_rules[rule['name']] = rule
    
    def _setup_anomaly_detection(self):
        """Setup anomaly detection for key metrics."""
        key_metrics = [
            'search_latency_ms',
            'search_accuracy', 
            'cpu_utilization',
            'memory_utilization',
            'error_rate',
            'search_throughput'
        ]
        
        for metric_name in key_metrics:
            self.anomaly_detectors[metric_name] = AnomalyDetection(
                metric_name=metric_name,
                detection_type=AnomalyType.STATISTICAL,
                sensitivity=0.95,
                window_size=50
            )
    
    def _create_default_dashboards(self):
        """Create default monitoring dashboards."""
        # Performance dashboard
        performance_dashboard = Dashboard(
            dashboard_id='performance',
            name='Performance Monitoring',
            metrics=[
                'search_latency_ms',
                'search_accuracy',
                'search_throughput',
                'energy_efficiency'
            ],
            charts=[
                {
                    'type': 'time_series',
                    'title': 'Search Latency',
                    'metrics': ['search_latency_ms'],
                    'y_axis': 'Latency (ms)'
                },
                {
                    'type': 'gauge',
                    'title': 'Search Accuracy',
                    'metrics': ['search_accuracy'],
                    'min_value': 0,
                    'max_value': 100
                }
            ]
        )
        
        # Resource dashboard
        resource_dashboard = Dashboard(
            dashboard_id='resources',
            name='Resource Utilization',
            metrics=[
                'cpu_utilization',
                'memory_utilization',
                'gpu_utilization'
            ],
            charts=[
                {
                    'type': 'stacked_area',
                    'title': 'Resource Utilization',
                    'metrics': ['cpu_utilization', 'memory_utilization', 'gpu_utilization'],
                    'y_axis': 'Utilization (%)'
                }
            ]
        )
        
        # Business dashboard
        business_dashboard = Dashboard(
            dashboard_id='business',
            name='Business Metrics',
            metrics=[
                'user_satisfaction',
                'cost_per_search',
                'search_throughput'
            ],
            charts=[
                {
                    'type': 'line_chart',
                    'title': 'Cost Efficiency',
                    'metrics': ['cost_per_search'],
                    'y_axis': 'Cost (USD)'
                }
            ]
        )
        
        self.dashboards['performance'] = performance_dashboard
        self.dashboards['resources'] = resource_dashboard
        self.dashboards['business'] = business_dashboard
    
    def _start_processing_threads(self):
        """Start background processing threads."""
        self.running = True
        
        # Metric processing thread
        metric_processor = threading.Thread(
            target=self._process_metrics, daemon=True
        )
        metric_processor.start()
        self.processing_threads.append(metric_processor)
        
        # Alert evaluation thread
        alert_processor = threading.Thread(
            target=self._evaluate_alerts, daemon=True
        )
        alert_processor.start()
        self.processing_threads.append(alert_processor)
        
        # Anomaly detection thread
        anomaly_processor = threading.Thread(
            target=self._detect_anomalies, daemon=True
        )
        anomaly_processor.start()
        self.processing_threads.append(anomaly_processor)
        
        # Predictive analytics thread
        prediction_processor = threading.Thread(
            target=self._run_predictive_analytics, daemon=True
        )
        prediction_processor.start()
        self.processing_threads.append(prediction_processor)
    
    def record_metric(self, name: str, value: float, 
                     timestamp: Optional[float] = None, 
                     tags: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        if timestamp is None:
            timestamp = time.time()
        
        metric = Metric(
            name=name,
            value=value,
            timestamp=timestamp,
            tags=tags or {}
        )
        
        try:
            self.metric_buffer.put_nowait(metric)
        except queue.Full:
            logging.warning("Metric buffer full, dropping metric")
    
    def _process_metrics(self):
        """Background thread to process incoming metrics."""
        while self.running:
            try:
                metric = self.metric_buffer.get(timeout=1)
                
                # Add to appropriate time series
                if metric.name in self.metric_series:
                    self.metric_series[metric.name].add_value(
                        metric.value, metric.timestamp
                    )
                else:
                    # Create new series for unknown metrics
                    self.metric_series[metric.name] = MetricSeries(
                        name=metric.name,
                        unit=metric.unit
                    )
                    self.metric_series[metric.name].add_value(
                        metric.value, metric.timestamp
                    )
                
                self.metric_buffer.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error processing metric: {e}")
    
    def _evaluate_alerts(self):
        """Background thread to evaluate alert conditions."""
        while self.running:
            try:
                for rule_name, rule in self.alert_rules.items():
                    metric_name = rule['metric']
                    
                    if metric_name not in self.metric_series:
                        continue
                    
                    latest_metric = self.metric_series[metric_name].get_latest()
                    if not latest_metric:
                        continue
                    
                    # Check if alert condition is met
                    alert_triggered = self._check_alert_condition(rule, latest_metric)
                    
                    if alert_triggered and rule_name not in self.active_alerts:
                        # Create new alert
                        alert = Alert(
                            alert_id=f"alert_{int(time.time())}_{rule_name}",
                            name=rule_name,
                            severity=rule['severity'],
                            message=rule['message'],
                            timestamp=time.time(),
                            metric_name=metric_name,
                            threshold_value=rule.get('threshold'),
                            actual_value=latest_metric.value
                        )
                        
                        self.active_alerts[rule_name] = alert
                        self.alert_history.append(alert)
                        
                        logging.warning(f"Alert triggered: {rule_name} - {rule['message']}")
                    
                    elif not alert_triggered and rule_name in self.active_alerts:
                        # Resolve alert
                        self.active_alerts[rule_name].resolved = True
                        del self.active_alerts[rule_name]
                        
                        logging.info(f"Alert resolved: {rule_name}")
                
                time.sleep(10)  # Check alerts every 10 seconds
                
            except Exception as e:
                logging.error(f"Error evaluating alerts: {e}")
                time.sleep(5)
    
    def _check_alert_condition(self, rule: Dict[str, Any], metric: Metric) -> bool:
        """Check if alert condition is satisfied."""
        condition = rule['condition']
        threshold = rule['threshold']
        value = metric.value
        
        if condition == 'greater_than':
            return value > threshold
        elif condition == 'less_than':
            return value < threshold
        elif condition == 'equals':
            return abs(value - threshold) < 0.001
        elif condition == 'spike':
            # Check for spike (simplified implementation)
            series = self.metric_series[rule['metric']]
            if len(series.values) < 10:
                return False
            
            recent_values = [m.value for m in list(series.values)[-10:-1]]
            avg_recent = sum(recent_values) / len(recent_values)
            
            return value > avg_recent * threshold
        
        return False
    
    def _detect_anomalies(self):
        """Background thread for anomaly detection."""
        while self.running:
            try:
                for detector in self.anomaly_detectors.values():
                    metric_name = detector.metric_name
                    
                    if metric_name not in self.metric_series:
                        continue
                    
                    series = self.metric_series[metric_name]
                    latest_metric = series.get_latest()
                    
                    if not latest_metric:
                        continue
                    
                    # Add to baseline
                    detector.baseline_values.append(latest_metric.value)
                    
                    # Detect anomaly if enough baseline data
                    if len(detector.baseline_values) >= detector.window_size:
                        anomaly = self._detect_statistical_anomaly(detector, latest_metric.value)
                        
                        if anomaly:
                            anomaly_event = {
                                'metric_name': metric_name,
                                'anomaly_type': detector.detection_type.value,
                                'timestamp': latest_metric.timestamp,
                                'value': latest_metric.value,
                                'expected_range': anomaly['expected_range'],
                                'severity': anomaly['severity'],
                                'confidence': anomaly['confidence']
                            }
                            
                            detector.anomalies_detected.append(anomaly_event)
                            self.anomaly_history.append(anomaly_event)
                            
                            logging.warning(f"Anomaly detected in {metric_name}: "
                                          f"value={latest_metric.value}, "
                                          f"expected={anomaly['expected_range']}")
                
                time.sleep(30)  # Check for anomalies every 30 seconds
                
            except Exception as e:
                logging.error(f"Error in anomaly detection: {e}")
                time.sleep(10)
    
    def _detect_statistical_anomaly(self, detector: AnomalyDetection, 
                                  current_value: float) -> Optional[Dict[str, Any]]:
        """Detect statistical anomalies using Z-score method."""
        values = list(detector.baseline_values)[:-1]  # Exclude current value
        
        if len(values) < 10:
            return None
        
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        
        if std_val == 0:
            return None
        
        z_score = abs((current_value - mean_val) / std_val)
        threshold = detector.threshold_multiplier
        
        if z_score > threshold:
            confidence = min(0.99, (z_score - threshold) / threshold)
            severity = 'high' if z_score > threshold * 1.5 else 'medium'
            
            return {
                'expected_range': (mean_val - threshold * std_val, 
                                 mean_val + threshold * std_val),
                'z_score': z_score,
                'confidence': confidence,
                'severity': severity
            }
        
        return None
    
    def _run_predictive_analytics(self):
        """Background thread for predictive analytics."""
        while self.running:
            try:
                # Run capacity planning every hour
                resource_metrics = {
                    name: series for name, series in self.metric_series.items()
                    if 'utilization' in name
                }
                
                if resource_metrics:
                    capacity_forecast = self.predictive_engine.detect_capacity_needs(
                        resource_metrics
                    )
                    
                    # Generate optimization recommendations
                    recommendations = self._generate_optimization_recommendations(
                        capacity_forecast
                    )
                    
                    if recommendations:
                        self.optimization_recommendations.extend(recommendations)
                        logging.info(f"Generated {len(recommendations)} optimization recommendations")
                
                time.sleep(3600)  # Run every hour
                
            except Exception as e:
                logging.error(f"Error in predictive analytics: {e}")
                time.sleep(600)  # Retry in 10 minutes
    
    def _generate_optimization_recommendations(self, 
                                             capacity_forecast: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate system optimization recommendations."""
        recommendations = []
        
        for metric_name, forecast in capacity_forecast.items():
            scaling_rec = forecast['scaling_recommendation']
            
            if scaling_rec['action'] != 'maintain':
                recommendation = {
                    'type': 'resource_scaling',
                    'metric': metric_name,
                    'action': scaling_rec['action'],
                    'urgency': scaling_rec['urgency'],
                    'recommendation': f"{scaling_rec['action'].replace('_', ' ').title()} "
                                    f"by {abs(scaling_rec.get('recommended_change', 0))}% "
                                    f"for {metric_name}",
                    'confidence': forecast['forecast_confidence'],
                    'timestamp': time.time()
                }
                recommendations.append(recommendation)
        
        # Add performance optimization recommendations
        if 'search_latency_ms' in self.metric_series:
            latency_series = self.metric_series['search_latency_ms']
            latest_latency = latency_series.get_latest()
            
            if latest_latency and latest_latency.value > 500:
                recommendation = {
                    'type': 'performance_optimization',
                    'metric': 'search_latency_ms',
                    'action': 'optimize_search_algorithm',
                    'urgency': 'medium',
                    'recommendation': 'Consider enabling caching or optimizing search algorithms '
                                    'to reduce latency',
                    'confidence': 0.8,
                    'timestamp': time.time()
                }
                recommendations.append(recommendation)
        
        return recommendations
    
    def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """Get dashboard data for visualization."""
        if dashboard_id not in self.dashboards:
            return {'error': f'Dashboard {dashboard_id} not found'}
        
        dashboard = self.dashboards[dashboard_id]
        dashboard_data = {
            'dashboard_id': dashboard_id,
            'name': dashboard.name,
            'refresh_interval': dashboard.refresh_interval_seconds,
            'last_updated': time.time(),
            'metrics': {},
            'charts': []
        }
        
        # Get metric data
        for metric_name in dashboard.metrics:
            if metric_name in self.metric_series:
                series = self.metric_series[metric_name]
                latest = series.get_latest()
                
                # Get recent values for charts
                recent_values = list(series.values)[-100:]  # Last 100 points
                
                dashboard_data['metrics'][metric_name] = {
                    'current_value': latest.value if latest else None,
                    'unit': series.unit,
                    'trend': self._calculate_trend(recent_values),
                    'data_points': [
                        {'timestamp': m.timestamp, 'value': m.value}
                        for m in recent_values
                    ]
                }
        
        # Prepare chart data
        for chart_config in dashboard.charts:
            chart_data = {
                'type': chart_config['type'],
                'title': chart_config['title'],
                'data': {}
            }
            
            for metric_name in chart_config['metrics']:
                if metric_name in dashboard_data['metrics']:
                    chart_data['data'][metric_name] = dashboard_data['metrics'][metric_name]
            
            dashboard_data['charts'].append(chart_data)
        
        return dashboard_data
    
    def _calculate_trend(self, values: List[Metric]) -> str:
        """Calculate trend direction for metric values."""
        if len(values) < 2:
            return 'stable'
        
        recent_values = [m.value for m in values[-10:]]
        if len(recent_values) < 2:
            return 'stable'
        
        # Simple trend calculation
        first_half = sum(recent_values[:len(recent_values)//2]) / (len(recent_values)//2)
        second_half = sum(recent_values[len(recent_values)//2:]) / (len(recent_values) - len(recent_values)//2)
        
        change_percent = ((second_half - first_half) / first_half) * 100 if first_half != 0 else 0
        
        if change_percent > 5:
            return 'increasing'
        elif change_percent < -5:
            return 'decreasing'
        else:
            return 'stable'
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary."""
        current_time = time.time()
        
        # Calculate overall system health
        health_score = self._calculate_health_score()
        
        # Get latest metrics
        latest_metrics = {}
        for name, series in self.metric_series.items():
            latest = series.get_latest()
            if latest:
                latest_metrics[name] = {
                    'value': latest.value,
                    'unit': series.unit,
                    'timestamp': latest.timestamp
                }
        
        # Anomaly summary
        recent_anomalies = [
            anomaly for anomaly in self.anomaly_history
            if current_time - anomaly['timestamp'] < 3600  # Last hour
        ]
        
        # Alert summary
        critical_alerts = [
            alert for alert in self.active_alerts.values()
            if alert.severity == AlertSeverity.CRITICAL
        ]
        
        # Optimization summary
        recent_recommendations = [
            rec for rec in self.optimization_recommendations
            if current_time - rec['timestamp'] < 3600  # Last hour
        ]
        
        return {
            'system_health': {
                'overall_score': health_score,
                'status': 'healthy' if health_score > 80 else 'degraded' if health_score > 60 else 'unhealthy'
            },
            'metrics_summary': {
                'total_metrics': len(self.metric_series),
                'active_series': len([s for s in self.metric_series.values() if s.get_latest()]),
                'latest_values': latest_metrics
            },
            'alerts_summary': {
                'active_alerts': len(self.active_alerts),
                'critical_alerts': len(critical_alerts),
                'total_alert_history': len(self.alert_history)
            },
            'anomalies_summary': {
                'recent_anomalies': len(recent_anomalies),
                'total_anomaly_history': len(self.anomaly_history)
            },
            'optimization_summary': {
                'recent_recommendations': len(recent_recommendations),
                'total_recommendations': len(self.optimization_recommendations)
            },
            'predictive_analytics': {
                'models_trained': len(self.predictive_engine.models),
                'active_forecasts': len(self.predictive_engine.forecasts)
            },
            'timestamp': current_time
        }
    
    def _calculate_health_score(self) -> float:
        """Calculate overall system health score (0-100)."""
        scores = []
        
        # Performance score
        if 'search_latency_ms' in self.metric_series:
            latency = self.metric_series['search_latency_ms'].get_latest()
            if latency:
                performance_score = max(0, 100 - (latency.value - 100) / 10)  # 100ms baseline
                scores.append(performance_score)
        
        # Quality score
        if 'search_accuracy' in self.metric_series:
            accuracy = self.metric_series['search_accuracy'].get_latest()
            if accuracy:
                scores.append(accuracy.value)
        
        # Resource utilization score
        resource_metrics = ['cpu_utilization', 'memory_utilization']
        resource_scores = []
        for metric in resource_metrics:
            if metric in self.metric_series:
                util = self.metric_series[metric].get_latest()
                if util:
                    # Optimal around 70%, penalty for too high or too low
                    if util.value < 30:
                        resource_scores.append(50 + util.value)  # Underutilization penalty
                    elif util.value > 90:
                        resource_scores.append(max(0, 190 - util.value))  # Overutilization penalty
                    else:
                        resource_scores.append(100)  # Optimal range
        
        if resource_scores:
            scores.append(sum(resource_scores) / len(resource_scores))
        
        # Alert penalty
        alert_penalty = len(self.active_alerts) * 10
        
        # Anomaly penalty
        recent_anomalies = sum(1 for a in self.anomaly_history 
                             if time.time() - a['timestamp'] < 3600)
        anomaly_penalty = recent_anomalies * 5
        
        if scores:
            base_score = sum(scores) / len(scores)
            final_score = max(0, base_score - alert_penalty - anomaly_penalty)
            return min(100, final_score)
        
        return 50  # Default neutral score
    
    def shutdown(self):
        """Shutdown analytics engine."""
        self.running = False
        logging.info("Real-time analytics engine shutting down")


# Factory functions
def create_realtime_analytics_engine(config: Optional[Dict[str, Any]] = None) -> RealTimeAnalyticsEngine:
    """Create real-time analytics engine with optional configuration."""
    return RealTimeAnalyticsEngine(config)


def create_custom_dashboard(dashboard_id: str, name: str, 
                          metrics: List[str], charts: List[Dict[str, Any]]) -> Dashboard:
    """Create custom analytics dashboard."""
    return Dashboard(
        dashboard_id=dashboard_id,
        name=name,
        metrics=metrics,
        charts=charts
    )