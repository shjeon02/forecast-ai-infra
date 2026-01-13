"""
Data models for infrastructure capacity forecasting.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class UserInput:
    """Input parameters from user."""
    concurrent_users: int
    registered_users: int
    growth_rate: Optional[float] = None
    peak_load_multiplier: Optional[float] = 1.5
    data_retention_days: Optional[int] = 365
    application_type: Optional[str] = "web"


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
