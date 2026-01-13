"""
Core forecasting engine for infrastructure capacity planning.
"""

from typing import Optional
from models import (
    UserInput,
    ComputeResources,
    StorageResources,
    NetworkResources,
    ScalingRecommendations,
    CostEstimate,
    CapacityPlan,
)
import config


class ForecastEngine:
    """Engine for calculating infrastructure capacity requirements."""
    
    def __init__(self):
        self.config = config
    
    def calculate_compute_resources(
        self, 
        concurrent_users: int, 
        peak_multiplier: float
    ) -> ComputeResources:
        """Calculate compute resource requirements."""
        # Base resources
        base_cpu = self.config.BASE_CPU_CORES
        base_memory = self.config.BASE_MEMORY_GB
        
        # User-based resources
        user_cpu = concurrent_users * self.config.CPU_PER_CONCURRENT_USER
        user_memory_mb = concurrent_users * self.config.MEMORY_PER_CONCURRENT_USER
        user_memory_gb = user_memory_mb / 1024
        
        # Apply peak multiplier
        total_cpu = (base_cpu + user_cpu) * peak_multiplier
        total_memory = (base_memory + user_memory_gb) * peak_multiplier
        
        # Apply safety margin
        total_cpu *= (1 + self.config.SAFETY_MARGIN)
        total_memory *= (1 + self.config.SAFETY_MARGIN)
        
        # Recommend instance types
        recommended_types = self._recommend_instance_types(total_cpu, total_memory)
        
        # Calculate instance counts
        min_instances = max(2, int(total_cpu / 8) + 1)  # At least 2 for HA
        max_instances = int(total_cpu / 2) + 1
        
        return ComputeResources(
            cpu_cores=total_cpu,
            memory_gb=total_memory,
            recommended_instance_types=recommended_types,
            min_instances=min_instances,
            max_instances=max_instances,
        )
    
    def calculate_storage_resources(
        self,
        registered_users: int,
        retention_days: int,
        growth_rate: Optional[float] = None
    ) -> StorageResources:
        """Calculate storage resource requirements."""
        # Growth factor
        growth_factor = (1 + (growth_rate / 100)) if growth_rate is not None else self.config.GROWTH_FACTOR
        
        # Database storage
        base_db_storage = registered_users * self.config.AVERAGE_DATA_PER_USER / 1024  # GB
        retention_growth = retention_days * self.config.DATA_RETENTION_DAILY_GROWTH
        database_storage = base_db_storage * (1 + retention_growth) * growth_factor
        
        # File storage
        file_storage = registered_users * self.config.AVERAGE_FILE_SIZE_PER_USER / 1024 * growth_factor  # GB
        
        # Backup storage (30% of primary storage)
        primary_storage = database_storage + file_storage
        backup_storage = primary_storage * self.config.BACKUP_MULTIPLIER
        
        # Apply safety margin
        database_storage *= (1 + self.config.SAFETY_MARGIN)
        file_storage *= (1 + self.config.SAFETY_MARGIN)
        backup_storage *= (1 + self.config.SAFETY_MARGIN)
        
        total_storage = database_storage + file_storage + backup_storage
        
        return StorageResources(
            database_storage_gb=database_storage,
            file_storage_gb=file_storage,
            backup_storage_gb=backup_storage,
            total_storage_gb=total_storage,
        )
    
    def calculate_network_resources(
        self,
        concurrent_users: int,
        peak_multiplier: float
    ) -> NetworkResources:
        """Calculate network resource requirements."""
        # Bandwidth calculation
        base_bandwidth = concurrent_users * self.config.BANDWIDTH_PER_CONCURRENT_USER
        peak_bandwidth = base_bandwidth * peak_multiplier
        bandwidth_mbps = peak_bandwidth * (1 + self.config.SAFETY_MARGIN)
        
        # Connection limits
        connection_limit = concurrent_users * self.config.CONNECTION_LIMIT_PER_USER
        
        # Load balancer requirements
        recommended_load_balancers = max(
            1, 
            int(connection_limit / self.config.LOAD_BALANCER_CAPACITY) + 1
        )
        
        return NetworkResources(
            bandwidth_mbps=bandwidth_mbps,
            connection_limit=connection_limit,
            recommended_load_balancers=recommended_load_balancers,
        )
    
    def calculate_scaling_recommendations(
        self,
        compute: ComputeResources
    ) -> ScalingRecommendations:
        """Calculate scaling recommendations."""
        # Horizontal scaling
        if compute.min_instances >= 4:
            horizontal = f"Recommended: Start with {compute.min_instances} instances, scale up to {compute.max_instances} based on load"
        else:
            horizontal = f"Recommended: Start with {compute.min_instances} instances for high availability, scale up to {compute.max_instances} during peak"
        
        # Vertical scaling
        if compute.cpu_cores <= 8:
            vertical = "Current instance size is appropriate. Consider upgrading if CPU usage consistently exceeds 70%"
        elif compute.cpu_cores <= 32:
            vertical = "Consider using larger instance types for better cost efficiency"
        else:
            vertical = "Distribute load across multiple instances for better scalability and fault tolerance"
        
        return ScalingRecommendations(
            horizontal_scaling=horizontal,
            vertical_scaling=vertical,
            auto_scaling_min=compute.min_instances,
            auto_scaling_max=compute.max_instances,
            auto_scaling_threshold_cpu=self.config.AUTO_SCALING_CPU_THRESHOLD,
            auto_scaling_threshold_memory=self.config.AUTO_SCALING_MEMORY_THRESHOLD,
        )
    
    def calculate_cost_estimate(
        self,
        compute: ComputeResources,
        storage: StorageResources,
        network: NetworkResources,
        include_cost: bool = False
    ) -> Optional[CostEstimate]:
        """Calculate cost estimates."""
        if not include_cost:
            return None
        
        # Compute costs
        monthly_compute_cost = (
            compute.cpu_cores * self.config.COST_PER_CPU_CORE_MONTHLY +
            compute.memory_gb * self.config.COST_PER_GB_MEMORY_MONTHLY
        )
        
        # Storage costs
        monthly_storage_cost = (
            storage.total_storage_gb * self.config.COST_PER_GB_STORAGE_MONTHLY
        )
        
        # Network costs
        monthly_network_cost = (
            network.bandwidth_mbps * self.config.COST_PER_MBPS_BANDWIDTH_MONTHLY +
            network.recommended_load_balancers * self.config.COST_PER_LOAD_BALANCER_MONTHLY
        )
        
        total_monthly_cost = (
            monthly_compute_cost + 
            monthly_storage_cost + 
            monthly_network_cost
        )
        
        return CostEstimate(
            monthly_compute_cost=monthly_compute_cost,
            monthly_storage_cost=monthly_storage_cost,
            monthly_network_cost=monthly_network_cost,
            total_monthly_cost=total_monthly_cost,
            currency="USD",
        )
    
    def _recommend_instance_types(
        self, 
        cpu_cores: float, 
        memory_gb: float
    ) -> list[str]:
        """Recommend appropriate instance types based on requirements."""
        recommendations = []
        
        for name, (cpu, mem) in self.config.INSTANCE_TYPES.items():
            if cpu >= cpu_cores * 0.8 and mem >= memory_gb * 0.8:  # 80% of requirement
                recommendations.append(name)
        
        # If no exact match, recommend the smallest that fits
        if not recommendations:
            for name, (cpu, mem) in sorted(
                self.config.INSTANCE_TYPES.items(),
                key=lambda x: (x[1][0], x[1][1])
            ):
                if cpu >= cpu_cores and mem >= memory_gb:
                    recommendations.append(name)
                    break
        
        # Always include at least one recommendation
        if not recommendations:
            recommendations.append("4xlarge")
        
        return recommendations[:3]  # Return top 3 recommendations
    
    def generate_capacity_plan(
        self,
        user_input: UserInput,
        include_cost: bool = False
    ) -> CapacityPlan:
        """Generate complete infrastructure capacity plan."""
        # Calculate all resource requirements
        compute = self.calculate_compute_resources(
            user_input.concurrent_users,
            user_input.peak_load_multiplier or self.config.DEFAULT_PEAK_LOAD_MULTIPLIER
        )
        
        storage = self.calculate_storage_resources(
            user_input.registered_users,
            user_input.data_retention_days or 365,
            user_input.growth_rate
        )
        
        network = self.calculate_network_resources(
            user_input.concurrent_users,
            user_input.peak_load_multiplier or self.config.DEFAULT_PEAK_LOAD_MULTIPLIER
        )
        
        scaling = self.calculate_scaling_recommendations(compute)
        
        cost = self.calculate_cost_estimate(compute, storage, network, include_cost)
        
        return CapacityPlan(
            user_input=user_input,
            compute=compute,
            storage=storage,
            network=network,
            scaling=scaling,
            cost=cost,
        )
