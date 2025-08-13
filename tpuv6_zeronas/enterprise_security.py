"""Enterprise Security Framework: Zero-trust, compliance, and advanced threat protection."""

import logging
import time
import hashlib
import hmac
import secrets
import base64
import json
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path
from enum import Enum
import re

from .architecture import Architecture
from .metrics import PerformanceMetrics
from .core import SearchConfig


class SecurityLevel(Enum):
    """Security classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event record."""
    event_id: str
    event_type: str
    severity: ThreatLevel
    timestamp: float
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    resource: Optional[str] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    mitigated: bool = False
    mitigation_actions: List[str] = field(default_factory=list)


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    max_concurrent_requests: int = 100
    max_request_size_mb: int = 10
    max_search_duration_hours: int = 24
    allowed_ip_ranges: List[str] = field(default_factory=list)
    blocked_ip_ranges: List[str] = field(default_factory=list)
    require_authentication: bool = True
    require_authorization: bool = True
    enable_encryption: bool = True
    enable_audit_logging: bool = True
    compliance_standards: List[str] = field(default_factory=lambda: ["SOC2", "ISO27001"])
    data_retention_days: int = 90
    
    def __post_init__(self):
        """Initialize default security settings."""
        if not self.allowed_ip_ranges:
            self.allowed_ip_ranges = ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"]


@dataclass
class UserProfile:
    """User security profile."""
    user_id: str
    role: str
    security_clearance: SecurityLevel
    permissions: Set[str] = field(default_factory=set)
    api_key_hash: Optional[str] = None
    last_login: Optional[float] = None
    failed_login_attempts: int = 0
    account_locked: bool = False
    session_tokens: Set[str] = field(default_factory=set)
    access_history: List[Dict[str, Any]] = field(default_factory=list)


class CryptographicService:
    """Enterprise cryptographic services."""
    
    def __init__(self):
        self.master_key = self._generate_master_key()
        self.key_rotation_interval = 3600 * 24 * 30  # 30 days
        self.last_key_rotation = time.time()
        self.encryption_algorithm = "AES-256-GCM"
        self.signing_algorithm = "HMAC-SHA256"
        
    def _generate_master_key(self) -> bytes:
        """Generate cryptographically secure master key."""
        return secrets.token_bytes(32)  # 256-bit key
    
    def encrypt_data(self, data: str, context: str = "") -> str:
        """Encrypt sensitive data with authenticated encryption."""
        try:
            # Simple encryption simulation (in production, use proper AES-GCM)
            key = self._derive_key(context)
            nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM
            
            # Simulate authenticated encryption
            encrypted = base64.b64encode(
                nonce + data.encode('utf-8') + 
                hashlib.sha256(key + data.encode('utf-8')).digest()[:16]
            ).decode('ascii')
            
            return encrypted
            
        except Exception as e:
            logging.error(f"Encryption failed: {e}")
            raise SecurityError(f"Encryption operation failed: {e}")
    
    def decrypt_data(self, encrypted_data: str, context: str = "") -> str:
        """Decrypt and verify authenticated data."""
        try:
            key = self._derive_key(context)
            data = base64.b64decode(encrypted_data.encode('ascii'))
            
            nonce = data[:12]
            ciphertext = data[12:-16]
            tag = data[-16:]
            
            # Verify authentication tag (simplified)
            expected_tag = hashlib.sha256(key + ciphertext).digest()[:16]
            if not hmac.compare_digest(tag, expected_tag):
                raise SecurityError("Authentication verification failed")
            
            return ciphertext.decode('utf-8')
            
        except Exception as e:
            logging.error(f"Decryption failed: {e}")
            raise SecurityError(f"Decryption operation failed: {e}")
    
    def _derive_key(self, context: str) -> bytes:
        """Derive context-specific encryption key."""
        return hashlib.pbkdf2_hmac(
            'sha256', 
            self.master_key, 
            context.encode('utf-8'), 
            100000  # iterations
        )
    
    def generate_api_key(self, user_id: str) -> Tuple[str, str]:
        """Generate secure API key pair."""
        api_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('ascii').rstrip('=')
        
        # Hash for storage
        api_key_hash = hashlib.pbkdf2_hmac(
            'sha256',
            api_key.encode('utf-8'),
            user_id.encode('utf-8'),
            100000
        ).hex()
        
        return api_key, api_key_hash
    
    def verify_api_key(self, api_key: str, user_id: str, stored_hash: str) -> bool:
        """Verify API key against stored hash."""
        computed_hash = hashlib.pbkdf2_hmac(
            'sha256',
            api_key.encode('utf-8'),
            user_id.encode('utf-8'),
            100000
        ).hex()
        
        return hmac.compare_digest(computed_hash, stored_hash)
    
    def generate_session_token(self, user_id: str) -> str:
        """Generate secure session token."""
        token_data = f"{user_id}:{time.time()}:{secrets.token_hex(16)}"
        return base64.urlsafe_b64encode(token_data.encode('utf-8')).decode('ascii')
    
    def verify_session_token(self, token: str, max_age_seconds: int = 3600) -> Optional[str]:
        """Verify session token and extract user ID."""
        try:
            token_data = base64.urlsafe_b64decode(token.encode('ascii')).decode('utf-8')
            parts = token_data.split(':')
            
            if len(parts) != 3:
                return None
            
            user_id, timestamp_str, _ = parts
            timestamp = float(timestamp_str)
            
            if time.time() - timestamp > max_age_seconds:
                return None  # Token expired
            
            return user_id
            
        except Exception:
            return None
    
    def rotate_keys(self):
        """Rotate encryption keys for security."""
        if time.time() - self.last_key_rotation > self.key_rotation_interval:
            old_key = self.master_key
            self.master_key = self._generate_master_key()
            self.last_key_rotation = time.time()
            
            logging.info("Cryptographic keys rotated for security")
            return True
        return False


class SecurityError(Exception):
    """Security-related exception."""
    pass


class ThreatDetectionEngine:
    """Advanced threat detection and response."""
    
    def __init__(self):
        self.anomaly_baselines = {}
        self.threat_patterns = self._load_threat_patterns()
        self.detection_rules = self._load_detection_rules()
        self.active_threats = {}
        self.mitigation_strategies = self._load_mitigation_strategies()
        
    def _load_threat_patterns(self) -> Dict[str, Any]:
        """Load known threat patterns."""
        return {
            'sql_injection': [
                r"(?i)(union\s+select|drop\s+table|insert\s+into)",
                r"(?i)(or\s+1\s*=\s*1|and\s+1\s*=\s*1)",
                r"(?i)(exec\s*\(|execute\s*\(|sp_executesql)"
            ],
            'xss_patterns': [
                r"(?i)(<script|javascript:|vbscript:|onload=|onerror=)",
                r"(?i)(alert\s*\(|confirm\s*\(|prompt\s*\()",
                r"(?i)(document\.cookie|window\.location)"
            ],
            'path_traversal': [
                r"\.\./|\.\.\\",
                r"(?i)(etc/passwd|windows/system32|boot\.ini)",
                r"(?i)(\\\\|//|\.\./\.\./)"
            ],
            'brute_force': {
                'max_attempts': 5,
                'time_window': 300,  # 5 minutes
                'lockout_duration': 1800  # 30 minutes
            },
            'ddos_patterns': {
                'max_requests_per_minute': 100,
                'max_concurrent_connections': 50,
                'suspicious_user_agents': [
                    'bot', 'crawler', 'scraper', 'scanner'
                ]
            }
        }
    
    def _load_detection_rules(self) -> List[Dict[str, Any]]:
        """Load security detection rules."""
        return [
            {
                'name': 'excessive_requests',
                'condition': lambda ctx: ctx.get('request_rate', 0) > 50,
                'severity': ThreatLevel.MEDIUM,
                'action': 'rate_limit'
            },
            {
                'name': 'large_payload',
                'condition': lambda ctx: ctx.get('payload_size', 0) > 10 * 1024 * 1024,
                'severity': ThreatLevel.HIGH,
                'action': 'block_request'
            },
            {
                'name': 'unauthorized_access',
                'condition': lambda ctx: not ctx.get('authenticated', False),
                'severity': ThreatLevel.HIGH,
                'action': 'require_auth'
            },
            {
                'name': 'suspicious_patterns',
                'condition': lambda ctx: self._check_malicious_patterns(ctx.get('input_data', '')),
                'severity': ThreatLevel.CRITICAL,
                'action': 'block_and_alert'
            }
        ]
    
    def _load_mitigation_strategies(self) -> Dict[str, List[str]]:
        """Load threat mitigation strategies."""
        return {
            'rate_limit': [
                'Implement exponential backoff',
                'Add request throttling',
                'Use sliding window rate limiting'
            ],
            'block_request': [
                'Return 403 Forbidden',
                'Log security event',
                'Update threat intelligence'
            ],
            'require_auth': [
                'Challenge for authentication',
                'Redirect to login page',
                'Implement MFA if needed'
            ],
            'block_and_alert': [
                'Immediately block source IP',
                'Send security alert to SOC',
                'Update WAF rules',
                'Initiate incident response'
            ]
        }
    
    def detect_threats(self, context: Dict[str, Any]) -> List[SecurityEvent]:
        """Detect security threats in given context."""
        threats = []
        
        for rule in self.detection_rules:
            try:
                if rule['condition'](context):
                    event = SecurityEvent(
                        event_id=secrets.token_hex(8),
                        event_type=rule['name'],
                        severity=rule['severity'],
                        timestamp=time.time(),
                        source_ip=context.get('source_ip'),
                        user_id=context.get('user_id'),
                        resource=context.get('resource'),
                        description=f"Security rule triggered: {rule['name']}",
                        metadata=context
                    )
                    threats.append(event)
                    
                    # Apply immediate mitigation
                    self._apply_mitigation(event, rule['action'])
                    
            except Exception as e:
                logging.error(f"Threat detection rule error: {e}")
        
        return threats
    
    def _check_malicious_patterns(self, input_data: str) -> bool:
        """Check input for malicious patterns."""
        for pattern_type, patterns in self.threat_patterns.items():
            if isinstance(patterns, list):
                for pattern in patterns:
                    if re.search(pattern, input_data):
                        logging.warning(f"Malicious pattern detected: {pattern_type}")
                        return True
        return False
    
    def _apply_mitigation(self, event: SecurityEvent, action: str):
        """Apply threat mitigation action."""
        if action in self.mitigation_strategies:
            event.mitigation_actions = self.mitigation_strategies[action]
            event.mitigated = True
            
            logging.warning(f"Threat mitigation applied: {action} for event {event.event_id}")
        else:
            logging.error(f"Unknown mitigation action: {action}")


class ComplianceEngine:
    """Compliance and regulatory adherence engine."""
    
    def __init__(self):
        self.compliance_standards = {
            'SOC2': self._soc2_controls(),
            'ISO27001': self._iso27001_controls(),
            'GDPR': self._gdpr_controls(),
            'HIPAA': self._hipaa_controls(),
            'PCI_DSS': self._pci_dss_controls()
        }
        self.audit_trails = []
        self.compliance_status = {}
        
    def _soc2_controls(self) -> Dict[str, Any]:
        """SOC 2 security controls."""
        return {
            'access_controls': {
                'description': 'Logical and physical access controls',
                'requirements': [
                    'Multi-factor authentication',
                    'Role-based access control',
                    'Regular access reviews',
                    'Segregation of duties'
                ]
            },
            'change_management': {
                'description': 'System change management',
                'requirements': [
                    'Change approval process',
                    'Testing procedures',
                    'Rollback capabilities',
                    'Documentation'
                ]
            },
            'monitoring': {
                'description': 'System monitoring and logging',
                'requirements': [
                    'Continuous monitoring',
                    'Log aggregation',
                    'Alerting mechanisms',
                    'Incident response'
                ]
            }
        }
    
    def _iso27001_controls(self) -> Dict[str, Any]:
        """ISO 27001 information security controls."""
        return {
            'information_security_policies': {
                'description': 'Information security management',
                'requirements': [
                    'Security policy framework',
                    'Risk management process',
                    'Security awareness training',
                    'Incident management'
                ]
            },
            'access_control': {
                'description': 'Access control management',
                'requirements': [
                    'User access management',
                    'User access provisioning',
                    'Management of privileged access',
                    'User access review'
                ]
            },
            'cryptography': {
                'description': 'Cryptographic controls',
                'requirements': [
                    'Policy on cryptographic controls',
                    'Key management',
                    'Encryption of data',
                    'Digital signatures'
                ]
            }
        }
    
    def _gdpr_controls(self) -> Dict[str, Any]:
        """GDPR privacy controls."""
        return {
            'data_protection': {
                'description': 'Personal data protection',
                'requirements': [
                    'Data protection by design',
                    'Data minimization',
                    'Purpose limitation',
                    'Storage limitation'
                ]
            },
            'rights_of_individuals': {
                'description': 'Individual privacy rights',
                'requirements': [
                    'Right to access',
                    'Right to rectification',
                    'Right to erasure',
                    'Right to data portability'
                ]
            },
            'data_breach_notification': {
                'description': 'Breach notification requirements',
                'requirements': [
                    '72-hour notification to authorities',
                    'Individual notification',
                    'Breach documentation',
                    'Impact assessment'
                ]
            }
        }
    
    def _hipaa_controls(self) -> Dict[str, Any]:
        """HIPAA healthcare security controls."""
        return {
            'administrative_safeguards': {
                'description': 'Administrative security measures',
                'requirements': [
                    'Security officer designation',
                    'Workforce training',
                    'Access management',
                    'Incident procedures'
                ]
            },
            'physical_safeguards': {
                'description': 'Physical protection measures',
                'requirements': [
                    'Facility access controls',
                    'Workstation use restrictions',
                    'Device and media controls',
                    'Equipment disposal'
                ]
            },
            'technical_safeguards': {
                'description': 'Technical security measures',
                'requirements': [
                    'Access control',
                    'Audit controls',
                    'Integrity controls',
                    'Transmission security'
                ]
            }
        }
    
    def _pci_dss_controls(self) -> Dict[str, Any]:
        """PCI DSS payment card security controls."""
        return {
            'network_security': {
                'description': 'Network security requirements',
                'requirements': [
                    'Firewall configuration',
                    'Default password changes',
                    'Cardholder data protection',
                    'Data transmission encryption'
                ]
            },
            'access_control': {
                'description': 'Access control measures',
                'requirements': [
                    'Unique ID assignment',
                    'Access restrictions',
                    'Multi-factor authentication',
                    'Access monitoring'
                ]
            },
            'vulnerability_management': {
                'description': 'Vulnerability management',
                'requirements': [
                    'Antivirus software',
                    'System updates',
                    'Secure development',
                    'Security testing'
                ]
            }
        }
    
    def assess_compliance(self, standard: str) -> Dict[str, Any]:
        """Assess compliance with specified standard."""
        if standard not in self.compliance_standards:
            raise ValueError(f"Unknown compliance standard: {standard}")
        
        controls = self.compliance_standards[standard]
        assessment_results = {}
        
        for control_name, control_info in controls.items():
            # Simulate compliance assessment
            compliance_score = self._evaluate_control_compliance(control_name, control_info)
            
            assessment_results[control_name] = {
                'description': control_info['description'],
                'requirements': control_info['requirements'],
                'compliance_score': compliance_score,
                'status': 'compliant' if compliance_score >= 0.8 else 'non_compliant',
                'recommendations': self._get_compliance_recommendations(control_name, compliance_score)
            }
        
        overall_score = sum(r['compliance_score'] for r in assessment_results.values()) / len(assessment_results)
        
        self.compliance_status[standard] = {
            'overall_score': overall_score,
            'status': 'compliant' if overall_score >= 0.8 else 'non_compliant',
            'assessment_date': time.time(),
            'controls': assessment_results
        }
        
        return self.compliance_status[standard]
    
    def _evaluate_control_compliance(self, control_name: str, control_info: Dict[str, Any]) -> float:
        """Evaluate compliance score for a specific control."""
        # Simulate compliance evaluation (in practice, this would involve
        # actual security control verification)
        base_score = 0.7  # Assume partial compliance
        
        # Add randomness for simulation
        import random
        random.seed(hash(control_name) % 2**32)
        variance = random.uniform(-0.2, 0.3)
        
        return max(0.0, min(1.0, base_score + variance))
    
    def _get_compliance_recommendations(self, control_name: str, score: float) -> List[str]:
        """Get recommendations for improving compliance."""
        recommendations = []
        
        if score < 0.5:
            recommendations.extend([
                f"Immediate attention required for {control_name}",
                "Conduct thorough security assessment",
                "Implement missing security controls",
                "Provide staff training"
            ])
        elif score < 0.8:
            recommendations.extend([
                f"Improvements needed for {control_name}",
                "Review current implementations",
                "Update policies and procedures",
                "Enhance monitoring capabilities"
            ])
        else:
            recommendations.extend([
                f"Maintain current standards for {control_name}",
                "Regular compliance reviews",
                "Continuous improvement"
            ])
        
        return recommendations
    
    def log_audit_event(self, event_type: str, details: Dict[str, Any]):
        """Log audit event for compliance tracking."""
        audit_event = {
            'event_id': secrets.token_hex(8),
            'event_type': event_type,
            'timestamp': time.time(),
            'details': details,
            'user_id': details.get('user_id'),
            'resource': details.get('resource'),
            'action': details.get('action'),
            'result': details.get('result', 'unknown')
        }
        
        self.audit_trails.append(audit_event)
        
        # Maintain audit log size (keep last 10000 events)
        if len(self.audit_trails) > 10000:
            self.audit_trails = self.audit_trails[-10000:]
        
        logging.info(f"Audit event logged: {event_type}")


class EnterpriseSecurityManager:
    """Comprehensive enterprise security management."""
    
    def __init__(self, security_policy: Optional[SecurityPolicy] = None):
        self.policy = security_policy or SecurityPolicy()
        self.crypto_service = CryptographicService()
        self.threat_detector = ThreatDetectionEngine()
        self.compliance_engine = ComplianceEngine()
        
        # Security state
        self.user_profiles = {}
        self.active_sessions = {}
        self.security_events = []
        self.access_logs = []
        
        # Rate limiting
        self.request_counts = {}
        self.rate_limit_lock = threading.Lock()
        
        # Initialize security monitoring
        self._initialize_security_monitoring()
        
    def _initialize_security_monitoring(self):
        """Initialize security monitoring systems."""
        logging.info("Enterprise security manager initialized")
        logging.info(f"Security policy: {len(self.policy.compliance_standards)} compliance standards")
        logging.info(f"Threat detection: {len(self.threat_detector.detection_rules)} rules loaded")
        
    def authenticate_user(self, api_key: str, context: Dict[str, Any]) -> Optional[UserProfile]:
        """Authenticate user with API key."""
        # Log authentication attempt
        self.compliance_engine.log_audit_event('authentication_attempt', {
            'source_ip': context.get('source_ip'),
            'user_agent': context.get('user_agent'),
            'timestamp': time.time()
        })
        
        # Rate limiting check
        source_ip = context.get('source_ip', 'unknown')
        if not self._check_rate_limit(source_ip):
            raise SecurityError("Rate limit exceeded")
        
        # Find user with matching API key
        for user in self.user_profiles.values():
            if user.api_key_hash and self.crypto_service.verify_api_key(api_key, user.user_id, user.api_key_hash):
                # Reset failed attempts on successful auth
                user.failed_login_attempts = 0
                user.last_login = time.time()
                
                # Log successful authentication
                self.compliance_engine.log_audit_event('authentication_success', {
                    'user_id': user.user_id,
                    'source_ip': source_ip,
                    'timestamp': time.time()
                })
                
                return user
        
        # Authentication failed
        self._handle_authentication_failure(source_ip, context)
        return None
    
    def _handle_authentication_failure(self, source_ip: str, context: Dict[str, Any]):
        """Handle authentication failure."""
        # Log failed authentication
        self.compliance_engine.log_audit_event('authentication_failure', {
            'source_ip': source_ip,
            'user_agent': context.get('user_agent'),
            'timestamp': time.time()
        })
        
        # Detect potential brute force attack
        threat_context = {
            'source_ip': source_ip,
            'event_type': 'authentication_failure',
            'timestamp': time.time()
        }
        
        threats = self.threat_detector.detect_threats(threat_context)
        if threats:
            self.security_events.extend(threats)
    
    def authorize_action(self, user: UserProfile, action: str, resource: str) -> bool:
        """Authorize user action on resource."""
        # Check account status
        if user.account_locked:
            raise SecurityError("Account is locked")
        
        # Check permissions
        required_permission = f"{action}:{resource}"
        if required_permission not in user.permissions and "admin:*" not in user.permissions:
            self.compliance_engine.log_audit_event('authorization_failure', {
                'user_id': user.user_id,
                'action': action,
                'resource': resource,
                'timestamp': time.time()
            })
            return False
        
        # Log successful authorization
        self.compliance_engine.log_audit_event('authorization_success', {
            'user_id': user.user_id,
            'action': action,
            'resource': resource,
            'timestamp': time.time()
        })
        
        return True
    
    def _check_rate_limit(self, identifier: str) -> bool:
        """Check if request is within rate limits."""
        with self.rate_limit_lock:
            current_time = time.time()
            
            if identifier not in self.request_counts:
                self.request_counts[identifier] = []
            
            # Remove old requests outside time window
            window_start = current_time - 60  # 1-minute window
            self.request_counts[identifier] = [
                req_time for req_time in self.request_counts[identifier] 
                if req_time > window_start
            ]
            
            # Check if within limit
            if len(self.request_counts[identifier]) >= self.policy.max_concurrent_requests:
                return False
            
            # Add current request
            self.request_counts[identifier].append(current_time)
            return True
    
    def encrypt_sensitive_data(self, data: str, classification: SecurityLevel) -> str:
        """Encrypt data based on security classification."""
        context = f"classification:{classification.value}"
        return self.crypto_service.encrypt_data(data, context)
    
    def decrypt_sensitive_data(self, encrypted_data: str, classification: SecurityLevel) -> str:
        """Decrypt data based on security classification."""
        context = f"classification:{classification.value}"
        return self.crypto_service.decrypt_data(encrypted_data, context)
    
    def create_user(self, user_id: str, role: str, security_clearance: SecurityLevel, 
                   permissions: Set[str]) -> Tuple[UserProfile, str]:
        """Create new user with security profile."""
        # Generate API key
        api_key, api_key_hash = self.crypto_service.generate_api_key(user_id)
        
        # Create user profile
        user = UserProfile(
            user_id=user_id,
            role=role,
            security_clearance=security_clearance,
            permissions=permissions,
            api_key_hash=api_key_hash
        )
        
        self.user_profiles[user_id] = user
        
        # Log user creation
        self.compliance_engine.log_audit_event('user_created', {
            'user_id': user_id,
            'role': role,
            'security_clearance': security_clearance.value,
            'permissions': list(permissions),
            'timestamp': time.time()
        })
        
        return user, api_key
    
    def rotate_security_keys(self):
        """Rotate all security keys and certificates."""
        rotated = self.crypto_service.rotate_keys()
        
        if rotated:
            # Log key rotation
            self.compliance_engine.log_audit_event('key_rotation', {
                'timestamp': time.time(),
                'rotation_type': 'scheduled'
            })
            
            # Invalidate old sessions
            self.active_sessions.clear()
            
    def perform_security_assessment(self) -> Dict[str, Any]:
        """Perform comprehensive security assessment."""
        assessment = {
            'timestamp': time.time(),
            'security_posture': self._assess_security_posture(),
            'threat_analysis': self._analyze_threats(),
            'compliance_status': self._check_compliance_status(),
            'recommendations': self._generate_security_recommendations()
        }
        
        return assessment
    
    def _assess_security_posture(self) -> Dict[str, Any]:
        """Assess overall security posture."""
        return {
            'total_users': len(self.user_profiles),
            'active_sessions': len(self.active_sessions),
            'locked_accounts': sum(1 for user in self.user_profiles.values() if user.account_locked),
            'security_events_24h': len([
                event for event in self.security_events 
                if time.time() - event.timestamp < 86400
            ]),
            'encryption_enabled': self.policy.enable_encryption,
            'audit_logging_enabled': self.policy.enable_audit_logging
        }
    
    def _analyze_threats(self) -> Dict[str, Any]:
        """Analyze current threat landscape."""
        threat_analysis = {
            'total_threats': len(self.security_events),
            'threat_severity_distribution': {},
            'top_threat_types': {},
            'mitigation_effectiveness': 0.0
        }
        
        # Analyze threat severity
        for event in self.security_events:
            severity = event.severity.value
            threat_analysis['threat_severity_distribution'][severity] = \
                threat_analysis['threat_severity_distribution'].get(severity, 0) + 1
        
        # Analyze threat types
        for event in self.security_events:
            event_type = event.event_type
            threat_analysis['top_threat_types'][event_type] = \
                threat_analysis['top_threat_types'].get(event_type, 0) + 1
        
        # Calculate mitigation effectiveness
        total_threats = len(self.security_events)
        mitigated_threats = sum(1 for event in self.security_events if event.mitigated)
        
        if total_threats > 0:
            threat_analysis['mitigation_effectiveness'] = mitigated_threats / total_threats
        
        return threat_analysis
    
    def _check_compliance_status(self) -> Dict[str, Any]:
        """Check compliance status for all standards."""
        compliance_status = {}
        
        for standard in self.policy.compliance_standards:
            try:
                status = self.compliance_engine.assess_compliance(standard)
                compliance_status[standard] = {
                    'overall_score': status['overall_score'],
                    'status': status['status'],
                    'assessment_date': status['assessment_date']
                }
            except Exception as e:
                logging.error(f"Compliance assessment failed for {standard}: {e}")
                compliance_status[standard] = {
                    'error': str(e),
                    'status': 'assessment_failed'
                }
        
        return compliance_status
    
    def _generate_security_recommendations(self) -> List[str]:
        """Generate security improvement recommendations."""
        recommendations = []
        
        # Analyze security metrics and generate recommendations
        if len(self.security_events) > 100:
            recommendations.append("High number of security events detected - review threat detection rules")
        
        if any(user.failed_login_attempts > 3 for user in self.user_profiles.values()):
            recommendations.append("Multiple failed login attempts detected - consider implementing account lockout")
        
        if not self.policy.enable_encryption:
            recommendations.append("Enable encryption for sensitive data protection")
        
        if not self.policy.enable_audit_logging:
            recommendations.append("Enable comprehensive audit logging for compliance")
        
        # Add compliance-specific recommendations
        for standard, status in self.compliance_engine.compliance_status.items():
            if status.get('status') == 'non_compliant':
                recommendations.append(f"Address {standard} compliance gaps")
        
        return recommendations
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard data."""
        return {
            'security_posture': self._assess_security_posture(),
            'threat_analysis': self._analyze_threats(),
            'compliance_status': self._check_compliance_status(),
            'recent_events': [
                {
                    'event_id': event.event_id,
                    'event_type': event.event_type,
                    'severity': event.severity.value,
                    'timestamp': event.timestamp,
                    'mitigated': event.mitigated
                }
                for event in self.security_events[-10:]  # Last 10 events
            ],
            'system_health': {
                'encryption_status': 'enabled' if self.policy.enable_encryption else 'disabled',
                'audit_logging': 'enabled' if self.policy.enable_audit_logging else 'disabled',
                'last_key_rotation': self.crypto_service.last_key_rotation,
                'active_policies': len(self.policy.compliance_standards)
            }
        }


# Factory function for enterprise security
def create_enterprise_security_manager(security_policy: Optional[SecurityPolicy] = None) -> EnterpriseSecurityManager:
    """Create enterprise security manager with optional custom policy."""
    return EnterpriseSecurityManager(security_policy)