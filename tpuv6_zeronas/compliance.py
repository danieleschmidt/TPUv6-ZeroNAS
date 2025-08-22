"""Compliance and regulatory framework for global deployment."""

import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum

class ComplianceRegion(Enum):
    """Supported compliance regions."""
    EU = "eu"       # GDPR
    US = "us"       # CCPA, PIPEDA
    ASIA = "asia"   # PDPA
    GLOBAL = "global"

class DataCategory(Enum):
    """Categories of data for compliance classification."""
    PERSONAL = "personal"
    SENSITIVE = "sensitive"
    ANONYMOUS = "anonymous"
    TELEMETRY = "telemetry"
    RESEARCH = "research"

@dataclass
class CompliancePolicy:
    """Compliance policy configuration."""
    region: ComplianceRegion
    data_retention_days: int
    encryption_required: bool
    audit_logging: bool
    user_consent_required: bool
    data_minimization: bool
    right_to_deletion: bool

class ComplianceManager:
    """Global compliance management system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.policies = self._initialize_policies()
        self.audit_log: List[Dict[str, Any]] = []
    
    def _initialize_policies(self) -> Dict[ComplianceRegion, CompliancePolicy]:
        """Initialize compliance policies for different regions."""
        return {
            ComplianceRegion.EU: CompliancePolicy(
                region=ComplianceRegion.EU,
                data_retention_days=90,
                encryption_required=True,
                audit_logging=True,
                user_consent_required=True,
                data_minimization=True,
                right_to_deletion=True
            ),
            ComplianceRegion.US: CompliancePolicy(
                region=ComplianceRegion.US,
                data_retention_days=365,
                encryption_required=True,
                audit_logging=True,
                user_consent_required=True,
                data_minimization=False,
                right_to_deletion=True
            ),
            ComplianceRegion.ASIA: CompliancePolicy(
                region=ComplianceRegion.ASIA,
                data_retention_days=180,
                encryption_required=True,
                audit_logging=True,
                user_consent_required=True,
                data_minimization=True,
                right_to_deletion=True
            ),
            ComplianceRegion.GLOBAL: CompliancePolicy(
                region=ComplianceRegion.GLOBAL,
                data_retention_days=90,  # Most restrictive
                encryption_required=True,
                audit_logging=True,
                user_consent_required=True,
                data_minimization=True,
                right_to_deletion=True
            )
        }
    
    def validate_data_processing(self, data_category: DataCategory, 
                                region: ComplianceRegion) -> Dict[str, Any]:
        """Validate data processing compliance."""
        policy = self.policies.get(region, self.policies[ComplianceRegion.GLOBAL])
        
        validation_result = {
            "compliant": True,
            "region": region.value,
            "policy_applied": policy,
            "requirements": [],
            "violations": []
        }
        
        # Check encryption requirements
        if policy.encryption_required:
            validation_result["requirements"].append("Data must be encrypted at rest and in transit")
        
        # Check data minimization
        if policy.data_minimization:
            validation_result["requirements"].append("Only necessary data should be collected")
        
        # Check consent requirements
        if policy.user_consent_required and data_category == DataCategory.PERSONAL:
            validation_result["requirements"].append("User consent required for personal data")
        
        # Audit log entry
        self._log_compliance_check(data_category, region, validation_result)
        
        return validation_result
    
    def _log_compliance_check(self, data_category: DataCategory, 
                            region: ComplianceRegion, result: Dict[str, Any]) -> None:
        """Log compliance check for audit trail."""
        import time
        
        audit_entry = {
            "timestamp": time.time(),
            "action": "compliance_check",
            "data_category": data_category.value,
            "region": region.value,
            "compliant": result["compliant"],
            "requirements_count": len(result["requirements"])
        }
        
        self.audit_log.append(audit_entry)
    
    def get_audit_report(self, region: Optional[ComplianceRegion] = None) -> Dict[str, Any]:
        """Generate compliance audit report."""
        filtered_logs = self.audit_log
        if region:
            filtered_logs = [log for log in self.audit_log if log["region"] == region.value]
        
        return {
            "total_checks": len(filtered_logs),
            "compliant_checks": sum(1 for log in filtered_logs if log["compliant"]),
            "non_compliant_checks": sum(1 for log in filtered_logs if not log["compliant"]),
            "regions_covered": list(set(log["region"] for log in filtered_logs)),
            "audit_entries": filtered_logs[-100:]  # Last 100 entries
        }

# Global compliance manager
compliance_manager = ComplianceManager()