"""
Service & Resource Management.

Provides service metadata handling, validation, and capacity request workflow.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from models import (
    ServiceInput,
    GPUConfig,
    LLMWorkloadInput,
    TrainingInput,
    ModelConfig,
)


class RequestStatus(str, Enum):
    """Capacity request status."""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"


@dataclass
class GPUResourceRequest:
    """GPU resource request for capacity planning."""
    gpu_type: str
    current_replicas: int
    requested_replicas: int
    model_size_billions: float
    target_rps: Optional[float] = None
    target_tps: Optional[float] = None
    justification: str = ""


@dataclass
class CapacityRequest:
    """Capacity request for approval workflow."""
    request_id: str
    service_id: str
    requester_email: str
    workload_type: str  # "inference" or "training"
    requested_resources: GPUResourceRequest
    justification: str
    status: RequestStatus = RequestStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    approved_by: Optional[str] = None
    rejection_reason: Optional[str] = None
    
    def submit(self):
        """Submit the request for approval."""
        if self.status != RequestStatus.DRAFT:
            raise ValueError("Can only submit draft requests")
        self.status = RequestStatus.SUBMITTED
        self.updated_at = datetime.now()
    
    def approve(self, approver: str):
        """Approve the request."""
        if self.status != RequestStatus.SUBMITTED:
            raise ValueError("Can only approve submitted requests")
        self.status = RequestStatus.APPROVED
        self.approved_by = approver
        self.updated_at = datetime.now()
    
    def reject(self, rejector: str, reason: str):
        """Reject the request."""
        if self.status != RequestStatus.SUBMITTED:
            raise ValueError("Can only reject submitted requests")
        self.status = RequestStatus.REJECTED
        self.approved_by = rejector
        self.rejection_reason = reason
        self.updated_at = datetime.now()
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "service_id": self.service_id,
            "requester_email": self.requester_email,
            "workload_type": self.workload_type,
            "requested_resources": {
                "gpu_type": self.requested_resources.gpu_type,
                "current_replicas": self.requested_resources.current_replicas,
                "requested_replicas": self.requested_resources.requested_replicas,
                "model_size_billions": self.requested_resources.model_size_billions,
                "target_rps": self.requested_resources.target_rps,
            },
            "justification": self.justification,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "approved_by": self.approved_by,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class AuditLogEntry:
    """Audit log entry for tracking changes."""
    timestamp: datetime
    action: str
    user: str
    resource_type: str
    resource_id: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "user": self.user,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "details": self.details,
        }


class ServiceValidator:
    """Validates service metadata and configurations."""
    
    VALID_ENVIRONMENTS = ["prod", "staging", "dev"]
    VALID_CRITICALITY = ["high", "medium", "low"]
    VALID_CLOUD_PROVIDERS = ["aws", "gcp", "azure"]
    VALID_GPU_TYPES = ["A100-40GB", "A100-80GB", "H100-80GB", "H100-SXM", "L4", "T4", "V100"]
    VALID_PRECISION = ["FP32", "FP16", "BF16", "INT8", "INT4", "FP8"]
    VALID_OPTIMIZERS = ["Adam", "AdamW", "Adafactor", "SGD"]
    
    def validate_service_input(self, service: ServiceInput) -> tuple[bool, List[str]]:
        """
        Validate service input.
        
        Returns: (is_valid, list of error messages)
        """
        errors = []
        
        if service.environment and service.environment not in self.VALID_ENVIRONMENTS:
            errors.append(
                f"Invalid environment: {service.environment}. "
                f"Valid: {', '.join(self.VALID_ENVIRONMENTS)}"
            )
        
        if service.criticality and service.criticality not in self.VALID_CRITICALITY:
            errors.append(
                f"Invalid criticality: {service.criticality}. "
                f"Valid: {', '.join(self.VALID_CRITICALITY)}"
            )
        
        if service.cloud_provider and service.cloud_provider not in self.VALID_CLOUD_PROVIDERS:
            errors.append(
                f"Invalid cloud provider: {service.cloud_provider}. "
                f"Valid: {', '.join(self.VALID_CLOUD_PROVIDERS)}"
            )
        
        return len(errors) == 0, errors
    
    def validate_gpu_config(self, gpu_config: GPUConfig) -> tuple[bool, List[str]]:
        """Validate GPU configuration."""
        errors = []
        
        if gpu_config.gpu_type and gpu_config.gpu_type not in self.VALID_GPU_TYPES:
            errors.append(
                f"Invalid GPU type: {gpu_config.gpu_type}. "
                f"Valid: {', '.join(self.VALID_GPU_TYPES)}"
            )
        
        if gpu_config.target_gpu_utilization < 0 or gpu_config.target_gpu_utilization > 1:
            errors.append("Target GPU utilization must be between 0 and 1")
        
        return len(errors) == 0, errors
    
    def validate_model_config(self, model_config: ModelConfig) -> tuple[bool, List[str]]:
        """Validate model configuration."""
        errors = []
        
        if model_config.model_size_billions <= 0:
            errors.append("Model size must be greater than 0")
        
        if model_config.precision not in self.VALID_PRECISION:
            errors.append(
                f"Invalid precision: {model_config.precision}. "
                f"Valid: {', '.join(self.VALID_PRECISION)}"
            )
        
        if model_config.context_window <= 0:
            errors.append("Context window must be greater than 0")
        
        if model_config.batch_size <= 0:
            errors.append("Batch size must be greater than 0")
        
        return len(errors) == 0, errors
    
    def validate_inference_workload(self, workload: LLMWorkloadInput) -> tuple[bool, List[str]]:
        """Validate inference workload input."""
        errors = []
        
        if workload.requests_per_second <= 0:
            errors.append("Requests per second must be greater than 0")
        
        if workload.avg_input_tokens <= 0:
            errors.append("Average input tokens must be greater than 0")
        
        if workload.avg_output_tokens <= 0:
            errors.append("Average output tokens must be greater than 0")
        
        if workload.peak_load_multiplier < 1.0:
            errors.append("Peak load multiplier must be at least 1.0")
        
        if workload.growth_rate is not None and (workload.growth_rate < 0 or workload.growth_rate > 1000):
            errors.append("Growth rate must be between 0 and 1000")
        
        return len(errors) == 0, errors
    
    def validate_training_input(self, training: TrainingInput) -> tuple[bool, List[str]]:
        """Validate training input."""
        errors = []
        
        if training.dataset_size_tokens <= 0:
            errors.append("Dataset size must be greater than 0")
        
        if training.sequence_length <= 0:
            errors.append("Sequence length must be greater than 0")
        
        if training.num_epochs <= 0:
            errors.append("Number of epochs must be greater than 0")
        
        if training.global_batch_size <= 0:
            errors.append("Global batch size must be greater than 0")
        
        if training.micro_batch_size <= 0:
            errors.append("Micro batch size must be greater than 0")
        
        if training.optimizer_type not in self.VALID_OPTIMIZERS:
            errors.append(
                f"Invalid optimizer: {training.optimizer_type}. "
                f"Valid: {', '.join(self.VALID_OPTIMIZERS)}"
            )
        
        return len(errors) == 0, errors


class ServiceManager:
    """
    Manages services and capacity requests.
    """
    
    def __init__(self):
        self.services: Dict[str, ServiceInput] = {}
        self.requests: Dict[str, CapacityRequest] = {}
        self.audit_log: List[AuditLogEntry] = []
        self.validator = ServiceValidator()
    
    def register_service(self, service: ServiceInput) -> tuple[bool, List[str]]:
        """
        Register a new service.
        
        Returns: (success, errors)
        """
        is_valid, errors = self.validator.validate_service_input(service)
        
        if not is_valid:
            return False, errors
        
        if service.service_id:
            self.services[service.service_id] = service
            self._log_action(
                action="service_registered",
                user="system",
                resource_type="service",
                resource_id=service.service_id,
                details={"service_name": service.service_name}
            )
        
        return True, []
    
    def get_service(self, service_id: str) -> Optional[ServiceInput]:
        """Get service by ID."""
        return self.services.get(service_id)
    
    def create_capacity_request(
        self,
        request_id: str,
        service_id: str,
        requester_email: str,
        workload_type: str,
        resources: GPUResourceRequest,
        justification: str
    ) -> CapacityRequest:
        """Create a new capacity request."""
        request = CapacityRequest(
            request_id=request_id,
            service_id=service_id,
            requester_email=requester_email,
            workload_type=workload_type,
            requested_resources=resources,
            justification=justification,
        )
        
        self.requests[request_id] = request
        self._log_action(
            action="request_created",
            user=requester_email,
            resource_type="capacity_request",
            resource_id=request_id,
            details={
                "service_id": service_id,
                "workload_type": workload_type,
                "requested_replicas": resources.requested_replicas,
            }
        )
        
        return request
    
    def submit_request(self, request_id: str) -> bool:
        """Submit a capacity request for approval."""
        request = self.requests.get(request_id)
        if not request:
            return False
        
        request.submit()
        self._log_action(
            action="request_submitted",
            user=request.requester_email,
            resource_type="capacity_request",
            resource_id=request_id,
        )
        return True
    
    def approve_request(self, request_id: str, approver: str) -> bool:
        """Approve a capacity request."""
        request = self.requests.get(request_id)
        if not request:
            return False
        
        request.approve(approver)
        self._log_action(
            action="request_approved",
            user=approver,
            resource_type="capacity_request",
            resource_id=request_id,
        )
        return True
    
    def reject_request(self, request_id: str, rejector: str, reason: str) -> bool:
        """Reject a capacity request."""
        request = self.requests.get(request_id)
        if not request:
            return False
        
        request.reject(rejector, reason)
        self._log_action(
            action="request_rejected",
            user=rejector,
            resource_type="capacity_request",
            resource_id=request_id,
            details={"reason": reason}
        )
        return True
    
    def get_request(self, request_id: str) -> Optional[CapacityRequest]:
        """Get a capacity request by ID."""
        return self.requests.get(request_id)
    
    def list_requests(
        self,
        service_id: Optional[str] = None,
        status: Optional[RequestStatus] = None
    ) -> List[CapacityRequest]:
        """List capacity requests with optional filters."""
        results = list(self.requests.values())
        
        if service_id:
            results = [r for r in results if r.service_id == service_id]
        
        if status:
            results = [r for r in results if r.status == status]
        
        return results
    
    def _log_action(
        self,
        action: str,
        user: str,
        resource_type: str,
        resource_id: str,
        details: Dict[str, Any] = None
    ):
        """Add an entry to the audit log."""
        entry = AuditLogEntry(
            timestamp=datetime.now(),
            action=action,
            user=user,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {}
        )
        self.audit_log.append(entry)
    
    def get_audit_log(
        self,
        resource_id: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditLogEntry]:
        """Get audit log entries."""
        results = self.audit_log
        
        if resource_id:
            results = [e for e in results if e.resource_id == resource_id]
        
        # Return most recent first
        return sorted(results, key=lambda x: x.timestamp, reverse=True)[:limit]


def calculate_gpu_cost_delta(
    current_gpus: int,
    requested_gpus: int,
    gpu_type: str,
    cloud_provider: str = "aws"
) -> dict:
    """
    Calculate the cost difference for a capacity change.
    
    Args:
        current_gpus: Current number of GPUs
        requested_gpus: Requested number of GPUs
        gpu_type: GPU type
        cloud_provider: Cloud provider
        
    Returns:
        Dictionary with cost delta information
    """
    import config
    
    # Get GPU cost
    if cloud_provider in config.CLOUD_GPU_PRICING:
        cost_per_hour = config.CLOUD_GPU_PRICING[cloud_provider].get(
            gpu_type,
            config.GPU_SPECS.get(gpu_type, {}).get("cost_per_hour_usd", 1.0)
        )
    else:
        cost_per_hour = config.GPU_SPECS.get(gpu_type, {}).get("cost_per_hour_usd", 1.0)
    
    current_monthly = current_gpus * cost_per_hour * config.HOURS_PER_MONTH
    requested_monthly = requested_gpus * cost_per_hour * config.HOURS_PER_MONTH
    delta_monthly = requested_monthly - current_monthly
    
    return {
        "current_gpus": current_gpus,
        "requested_gpus": requested_gpus,
        "gpu_type": gpu_type,
        "cost_per_gpu_hour": cost_per_hour,
        "current_monthly_cost": current_monthly,
        "requested_monthly_cost": requested_monthly,
        "delta_monthly_cost": delta_monthly,
        "currency": "USD",
    }
