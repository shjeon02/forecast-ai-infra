"""
Data models for infrastructure capacity forecasting.

Supports two main workload types:
1. LLM Inference (real-time serving)
2. LLM Fine-Tuning (training)
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum


# =============================================================================
# Enums for LLM Configuration
# =============================================================================

class Precision(str, Enum):
    """Model precision/quantization options."""
    FP32 = "FP32"
    FP16 = "FP16"
    BF16 = "BF16"
    INT8 = "INT8"
    INT4 = "INT4"
    FP8 = "FP8"


class GPUType(str, Enum):
    """Supported GPU types."""
    A100_40GB = "A100-40GB"
    A100_80GB = "A100-80GB"
    H100_80GB = "H100-80GB"
    H100_SXM = "H100-SXM"
    L4 = "L4"
    T4 = "T4"
    V100 = "V100"


class OptimizerType(str, Enum):
    """Optimizer types for training."""
    ADAM = "Adam"
    ADAMW = "AdamW"
    ADAFACTOR = "Adafactor"
    SGD = "SGD"


class Environment(str, Enum):
    """Deployment environment."""
    PROD = "prod"
    STAGING = "staging"
    DEV = "dev"


class Criticality(str, Enum):
    """Service criticality level."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CloudProvider(str, Enum):
    """Cloud provider options."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"


class WorkloadType(str, Enum):
    """Workload type selection."""
    INFERENCE = "inference"
    TRAINING = "training"


# =============================================================================
# Legacy Models (backward compatibility)
# =============================================================================

@dataclass
class UserInput:
    """Input parameters from user (legacy web application model)."""
    concurrent_users: int
    registered_users: int
    growth_rate: Optional[float] = None
    peak_load_multiplier: Optional[float] = 1.5
    data_retention_days: Optional[int] = 365
    application_type: Optional[str] = "web"


# =============================================================================
# LLM Input Models
# =============================================================================

@dataclass
class ModelConfig:
    """LLM model configuration."""
    model_size_billions: float  # e.g., 7, 70, 405
    precision: str = "FP16"  # FP32, FP16, BF16, INT8, INT4
    context_window: int = 4096
    batch_size: int = 1
    target_latency_p50_ms: Optional[float] = None
    target_latency_p99_ms: Optional[float] = None
    
    @property
    def bytes_per_param(self) -> float:
        """Get bytes per parameter based on precision."""
        precision_bytes = {
            "FP32": 4.0,
            "FP16": 2.0,
            "BF16": 2.0,
            "INT8": 1.0,
            "INT4": 0.5,
            "FP8": 1.0,
        }
        return precision_bytes.get(self.precision, 2.0)
    
    @property
    def model_params(self) -> int:
        """Get model parameters count."""
        return int(self.model_size_billions * 1e9)


@dataclass
class GPUConfig:
    """GPU configuration."""
    gpu_type: Optional[str] = None  # A100-40GB, A100-80GB, H100-80GB, etc.
    target_gpu_utilization: float = 0.7
    current_replicas: Optional[int] = None
    
    @property
    def gpu_memory_gb(self) -> float:
        """Get GPU memory based on type."""
        gpu_memory_map = {
            "A100-40GB": 40.0,
            "A100-80GB": 80.0,
            "H100-80GB": 80.0,
            "H100-SXM": 80.0,
            "L4": 24.0,
            "T4": 16.0,
            "V100": 32.0,
        }
        if self.gpu_type:
            return gpu_memory_map.get(self.gpu_type, 80.0)
        return 80.0  # Default to A100-80GB


@dataclass
class LLMWorkloadInput:
    """LLM inference workload input parameters."""
    requests_per_second: float  # RPS
    avg_input_tokens: int
    avg_output_tokens: int
    peak_load_multiplier: float = 1.5
    growth_rate: Optional[float] = None  # percentage (0-1000)


@dataclass
class TrainingInput:
    """LLM fine-tuning/training input parameters."""
    dataset_size_tokens: int  # Total tokens in dataset
    sequence_length: int  # e.g., 2048, 4096, 8192
    num_epochs: int  # e.g., 1, 3, 5
    global_batch_size: int  # e.g., 32, 64, 128
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    optimizer_type: str = "Adam"  # Adam, AdamW, Adafactor, SGD
    gradient_checkpointing: bool = False
    training_precision: str = "FP16"  # FP32, FP16, BF16, FP8


@dataclass
class ServiceInput:
    """Service metadata for advanced mode."""
    service_id: Optional[str] = None
    service_name: Optional[str] = None
    team_id: Optional[str] = None
    environment: str = "prod"  # prod, staging, dev
    criticality: str = "medium"  # high, medium, low
    cloud_provider: Optional[str] = None  # aws, gcp, azure
    region: Optional[str] = None


@dataclass
class ComputeResources:
    """Compute resource requirements."""
    cpu_cores: float
    memory_gb: float
    recommended_instance_types: list[str]
    min_instances: int
    max_instances: int


@dataclass
class StorageResources:
    """Storage resource requirements."""
    database_storage_gb: float
    file_storage_gb: float
    backup_storage_gb: float
    total_storage_gb: float


@dataclass
class NetworkResources:
    """Network resource requirements."""
    bandwidth_mbps: float
    connection_limit: int
    recommended_load_balancers: int


@dataclass
class ScalingRecommendations:
    """Scaling recommendations."""
    horizontal_scaling: str
    vertical_scaling: str
    auto_scaling_min: int
    auto_scaling_max: int
    auto_scaling_threshold_cpu: float
    auto_scaling_threshold_memory: float


@dataclass
class CostEstimate:
    """Cost estimates (optional)."""
    monthly_compute_cost: Optional[float] = None
    monthly_storage_cost: Optional[float] = None
    monthly_network_cost: Optional[float] = None
    total_monthly_cost: Optional[float] = None
    currency: str = "USD"


@dataclass
class CapacityPlan:
    """Complete infrastructure capacity plan."""
    user_input: UserInput
    compute: ComputeResources
    storage: StorageResources
    network: NetworkResources
    scaling: ScalingRecommendations
    cost: Optional[CostEstimate] = None
    
    def to_dict(self) -> dict:
        """Convert capacity plan to dictionary."""
        return {
            "user_input": {
                "concurrent_users": self.user_input.concurrent_users,
                "registered_users": self.user_input.registered_users,
                "growth_rate": self.user_input.growth_rate,
                "peak_load_multiplier": self.user_input.peak_load_multiplier,
                "data_retention_days": self.user_input.data_retention_days,
                "application_type": self.user_input.application_type,
            },
            "compute": {
                "cpu_cores": round(self.compute.cpu_cores, 2),
                "memory_gb": round(self.compute.memory_gb, 2),
                "recommended_instance_types": self.compute.recommended_instance_types,
                "min_instances": self.compute.min_instances,
                "max_instances": self.compute.max_instances,
            },
            "storage": {
                "database_storage_gb": round(self.storage.database_storage_gb, 2),
                "file_storage_gb": round(self.storage.file_storage_gb, 2),
                "backup_storage_gb": round(self.storage.backup_storage_gb, 2),
                "total_storage_gb": round(self.storage.total_storage_gb, 2),
            },
            "network": {
                "bandwidth_mbps": round(self.network.bandwidth_mbps, 2),
                "connection_limit": self.network.connection_limit,
                "recommended_load_balancers": self.network.recommended_load_balancers,
            },
            "scaling": {
                "horizontal_scaling": self.scaling.horizontal_scaling,
                "vertical_scaling": self.scaling.vertical_scaling,
                "auto_scaling_min": self.scaling.auto_scaling_min,
                "auto_scaling_max": self.scaling.auto_scaling_max,
                "auto_scaling_threshold_cpu": self.scaling.auto_scaling_threshold_cpu,
                "auto_scaling_threshold_memory": self.scaling.auto_scaling_threshold_memory,
            },
            "cost": {
                "monthly_compute_cost": round(self.cost.monthly_compute_cost, 2) if self.cost and self.cost.monthly_compute_cost else None,
                "monthly_storage_cost": round(self.cost.monthly_storage_cost, 2) if self.cost and self.cost.monthly_storage_cost else None,
                "monthly_network_cost": round(self.cost.monthly_network_cost, 2) if self.cost and self.cost.monthly_network_cost else None,
                "total_monthly_cost": round(self.cost.total_monthly_cost, 2) if self.cost and self.cost.total_monthly_cost else None,
                "currency": self.cost.currency if self.cost else "USD",
            } if self.cost else None,
        }


# =============================================================================
# LLM Output Models
# =============================================================================

@dataclass
class GPUResources:
    """GPU resource requirements for inference."""
    gpu_memory_per_replica_gb: float
    min_gpus_per_replica: int
    recommended_gpu_type: str
    min_replicas: int
    max_replicas: int
    total_gpus: int


@dataclass
class ThroughputMetrics:
    """Throughput and latency metrics for inference."""
    tps_per_replica: float  # Tokens per second per replica
    total_tps_capacity: float
    max_rps_capacity: float
    estimated_ttft_ms: float  # Time to first token
    estimated_itl_ms: float  # Inter-token latency
    meets_latency_sla: bool = True


@dataclass 
class LLMStorageResources:
    """Storage requirements for LLM workloads."""
    model_storage_gb: float
    cache_storage_gb: float
    log_storage_gb: float = 10.0
    total_storage_gb: float = 0.0
    
    def __post_init__(self):
        if self.total_storage_gb == 0.0:
            self.total_storage_gb = self.model_storage_gb + self.cache_storage_gb + self.log_storage_gb


@dataclass
class LLMNetworkResources:
    """Network requirements for LLM workloads."""
    bandwidth_mbps: float
    connection_limit: int
    recommended_load_balancers: int


@dataclass
class LLMScalingRecommendations:
    """Scaling recommendations for LLM inference."""
    recommended_min_replicas: int
    recommended_max_replicas: int
    auto_scale_rps_threshold: float
    auto_scale_gpu_util_threshold: float
    auto_scale_latency_threshold_ms: float


@dataclass
class LLMCostEstimate:
    """Cost estimates for LLM workloads."""
    monthly_gpu_cost: float
    cost_per_million_tokens: float
    monthly_storage_cost: float = 0.0
    monthly_network_cost: float = 0.0
    monthly_total_cost: float = 0.0
    currency: str = "USD"
    
    def __post_init__(self):
        if self.monthly_total_cost == 0.0:
            self.monthly_total_cost = (
                self.monthly_gpu_cost + 
                self.monthly_storage_cost + 
                self.monthly_network_cost
            )


@dataclass
class LLMCapacityPlan:
    """Complete LLM inference capacity plan."""
    workload_input: LLMWorkloadInput
    model_config: ModelConfig
    gpu_resources: GPUResources
    throughput: ThroughputMetrics
    storage: LLMStorageResources
    network: LLMNetworkResources
    scaling: LLMScalingRecommendations
    cost: Optional[LLMCostEstimate] = None
    gpu_config: Optional[GPUConfig] = None
    service: Optional[ServiceInput] = None
    
    def to_dict(self) -> dict:
        """Convert capacity plan to dictionary."""
        result = {
            "workload_input": {
                "requests_per_second": self.workload_input.requests_per_second,
                "avg_input_tokens": self.workload_input.avg_input_tokens,
                "avg_output_tokens": self.workload_input.avg_output_tokens,
                "peak_load_multiplier": self.workload_input.peak_load_multiplier,
                "growth_rate": self.workload_input.growth_rate,
            },
            "model_config": {
                "model_size_billions": self.model_config.model_size_billions,
                "precision": self.model_config.precision,
                "context_window": self.model_config.context_window,
                "batch_size": self.model_config.batch_size,
                "target_latency_p50_ms": self.model_config.target_latency_p50_ms,
                "target_latency_p99_ms": self.model_config.target_latency_p99_ms,
            },
            "gpu_resources": {
                "gpu_memory_per_replica_gb": round(self.gpu_resources.gpu_memory_per_replica_gb, 2),
                "min_gpus_per_replica": self.gpu_resources.min_gpus_per_replica,
                "recommended_gpu_type": self.gpu_resources.recommended_gpu_type,
                "min_replicas": self.gpu_resources.min_replicas,
                "max_replicas": self.gpu_resources.max_replicas,
                "total_gpus": self.gpu_resources.total_gpus,
            },
            "throughput": {
                "tps_per_replica": round(self.throughput.tps_per_replica, 2),
                "total_tps_capacity": round(self.throughput.total_tps_capacity, 2),
                "max_rps_capacity": round(self.throughput.max_rps_capacity, 2),
                "estimated_ttft_ms": round(self.throughput.estimated_ttft_ms, 2),
                "estimated_itl_ms": round(self.throughput.estimated_itl_ms, 2),
                "meets_latency_sla": self.throughput.meets_latency_sla,
            },
            "storage": {
                "model_storage_gb": round(self.storage.model_storage_gb, 2),
                "cache_storage_gb": round(self.storage.cache_storage_gb, 2),
                "log_storage_gb": round(self.storage.log_storage_gb, 2),
                "total_storage_gb": round(self.storage.total_storage_gb, 2),
            },
            "network": {
                "bandwidth_mbps": round(self.network.bandwidth_mbps, 2),
                "connection_limit": self.network.connection_limit,
                "recommended_load_balancers": self.network.recommended_load_balancers,
            },
            "scaling": {
                "recommended_min_replicas": self.scaling.recommended_min_replicas,
                "recommended_max_replicas": self.scaling.recommended_max_replicas,
                "auto_scale_rps_threshold": self.scaling.auto_scale_rps_threshold,
                "auto_scale_gpu_util_threshold": self.scaling.auto_scale_gpu_util_threshold,
                "auto_scale_latency_threshold_ms": self.scaling.auto_scale_latency_threshold_ms,
            },
        }
        
        if self.cost:
            result["cost"] = {
                "monthly_gpu_cost": round(self.cost.monthly_gpu_cost, 2),
                "cost_per_million_tokens": round(self.cost.cost_per_million_tokens, 4),
                "monthly_storage_cost": round(self.cost.monthly_storage_cost, 2),
                "monthly_network_cost": round(self.cost.monthly_network_cost, 2),
                "monthly_total_cost": round(self.cost.monthly_total_cost, 2),
                "currency": self.cost.currency,
            }
        
        if self.service:
            result["service"] = {
                "service_id": self.service.service_id,
                "service_name": self.service.service_name,
                "environment": self.service.environment,
                "criticality": self.service.criticality,
                "cloud_provider": self.service.cloud_provider,
                "region": self.service.region,
            }
        
        return result


# =============================================================================
# Training Output Models
# =============================================================================

@dataclass
class TrainingGPUResources:
    """GPU resource requirements for training."""
    gpu_memory_per_gpu_gb: float
    model_weights_gb: float
    gradients_gb: float
    optimizer_states_gb: float
    activations_gb: float
    required_gpus: int
    recommended_gpu_type: str
    gpus_per_node: int = 8
    num_nodes: int = 1


@dataclass
class TrainingMetrics:
    """Training throughput and duration metrics."""
    tokens_per_second_per_gpu: float
    total_tokens_per_second: float
    steps_per_epoch: int
    total_steps: int
    estimated_duration_hours: float
    estimated_duration_days: float


@dataclass
class DataParallelConfig:
    """Data parallelism configuration."""
    data_parallel_size: int
    effective_batch_size: int
    gradient_accumulation_steps: int


@dataclass
class TrainingStorageResources:
    """Storage requirements for training."""
    model_checkpoint_gb: float
    dataset_storage_gb: float
    total_storage_gb: float


@dataclass
class TrainingCostEstimate:
    """Cost estimates for training."""
    gpu_cost_per_hour: float
    total_training_cost: float
    cost_per_epoch: float
    currency: str = "USD"


@dataclass
class TrainingCapacityPlan:
    """Complete LLM training capacity plan."""
    training_input: TrainingInput
    model_config: ModelConfig
    gpu_resources: TrainingGPUResources
    training_metrics: TrainingMetrics
    data_parallel_config: DataParallelConfig
    storage: TrainingStorageResources
    cost: Optional[TrainingCostEstimate] = None
    gpu_config: Optional[GPUConfig] = None
    service: Optional[ServiceInput] = None
    
    def to_dict(self) -> dict:
        """Convert training plan to dictionary."""
        result = {
            "training_input": {
                "dataset_size_tokens": self.training_input.dataset_size_tokens,
                "sequence_length": self.training_input.sequence_length,
                "num_epochs": self.training_input.num_epochs,
                "global_batch_size": self.training_input.global_batch_size,
                "micro_batch_size": self.training_input.micro_batch_size,
                "gradient_accumulation_steps": self.training_input.gradient_accumulation_steps,
                "optimizer_type": self.training_input.optimizer_type,
                "gradient_checkpointing": self.training_input.gradient_checkpointing,
                "training_precision": self.training_input.training_precision,
            },
            "model_config": {
                "model_size_billions": self.model_config.model_size_billions,
                "precision": self.model_config.precision,
            },
            "gpu_resources": {
                "gpu_memory_per_gpu_gb": round(self.gpu_resources.gpu_memory_per_gpu_gb, 2),
                "model_weights_gb": round(self.gpu_resources.model_weights_gb, 2),
                "gradients_gb": round(self.gpu_resources.gradients_gb, 2),
                "optimizer_states_gb": round(self.gpu_resources.optimizer_states_gb, 2),
                "activations_gb": round(self.gpu_resources.activations_gb, 2),
                "required_gpus": self.gpu_resources.required_gpus,
                "recommended_gpu_type": self.gpu_resources.recommended_gpu_type,
                "gpus_per_node": self.gpu_resources.gpus_per_node,
                "num_nodes": self.gpu_resources.num_nodes,
            },
            "training_metrics": {
                "tokens_per_second_per_gpu": round(self.training_metrics.tokens_per_second_per_gpu, 2),
                "total_tokens_per_second": round(self.training_metrics.total_tokens_per_second, 2),
                "steps_per_epoch": self.training_metrics.steps_per_epoch,
                "total_steps": self.training_metrics.total_steps,
                "estimated_duration_hours": round(self.training_metrics.estimated_duration_hours, 2),
                "estimated_duration_days": round(self.training_metrics.estimated_duration_days, 2),
            },
            "data_parallel_config": {
                "data_parallel_size": self.data_parallel_config.data_parallel_size,
                "effective_batch_size": self.data_parallel_config.effective_batch_size,
                "gradient_accumulation_steps": self.data_parallel_config.gradient_accumulation_steps,
            },
            "storage": {
                "model_checkpoint_gb": round(self.storage.model_checkpoint_gb, 2),
                "dataset_storage_gb": round(self.storage.dataset_storage_gb, 2),
                "total_storage_gb": round(self.storage.total_storage_gb, 2),
            },
        }
        
        if self.cost:
            result["cost"] = {
                "gpu_cost_per_hour": round(self.cost.gpu_cost_per_hour, 2),
                "total_training_cost": round(self.cost.total_training_cost, 2),
                "cost_per_epoch": round(self.cost.cost_per_epoch, 2),
                "currency": self.cost.currency,
            }
        
        if self.service:
            result["service"] = {
                "service_id": self.service.service_id,
                "environment": self.service.environment,
                "cloud_provider": self.service.cloud_provider,
            }
        
        return result
