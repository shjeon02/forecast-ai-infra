"""
LLM Inference Forecasting Engine.

Calculates GPU memory, throughput, latency, and cost for LLM inference workloads.
"""

import math
from typing import Optional

from models import (
    LLMWorkloadInput,
    ModelConfig,
    GPUConfig,
    ServiceInput,
    GPUResources,
    ThroughputMetrics,
    LLMStorageResources,
    LLMNetworkResources,
    LLMScalingRecommendations,
    LLMCostEstimate,
    LLMCapacityPlan,
)
import config


class InferenceEngine:
    """Engine for calculating LLM inference capacity requirements."""
    
    def __init__(self):
        self.config = config
    
    def _get_model_architecture(self, model_size_b: float) -> dict:
        """Get model architecture parameters based on size."""
        # Find closest model size in config
        sizes = sorted(self.config.MODEL_ARCHITECTURE.keys())
        closest_size = min(sizes, key=lambda x: abs(x - model_size_b))
        
        arch = self.config.MODEL_ARCHITECTURE[closest_size].copy()
        
        # Scale for exact size if different
        if closest_size != model_size_b:
            scale = math.sqrt(model_size_b / closest_size)
            arch["hidden_dim"] = int(arch["hidden_dim"] * scale)
            arch["num_layers"] = int(arch["num_layers"] * scale)
        
        return arch
    
    def _get_gpu_spec(self, gpu_type: Optional[str]) -> dict:
        """Get GPU specifications."""
        if gpu_type and gpu_type in self.config.GPU_SPECS:
            return self.config.GPU_SPECS[gpu_type]
        # Default to A100-80GB
        return self.config.GPU_SPECS["A100-80GB"]
    
    def calculate_gpu_memory(
        self,
        model_config: ModelConfig,
        gpu_config: Optional[GPUConfig] = None
    ) -> tuple[float, float, float]:
        """
        Calculate GPU memory requirements for inference.
        
        Returns: (model_weights_gb, kv_cache_gb, total_memory_gb)
        """
        model_params = model_config.model_params
        bytes_per_param = model_config.bytes_per_param
        
        # Model weights memory
        model_weights_gb = (model_params * bytes_per_param) / (1024 ** 3)
        
        # Get model architecture for KV cache calculation
        arch = self._get_model_architecture(model_config.model_size_billions)
        num_layers = arch["num_layers"]
        hidden_dim = arch["hidden_dim"]
        
        # KV cache memory
        # KV cache = 2 × num_layers × hidden_dim × context_length × batch_size × bytes_per_param
        kv_cache_gb = (
            2 * num_layers * hidden_dim * 
            model_config.context_window * model_config.batch_size * 
            bytes_per_param
        ) / (1024 ** 3)
        
        # Activation memory (during inference, relatively small)
        activation_gb = model_weights_gb * self.config.ACTIVATION_MEMORY_RATIO
        
        # Total with overhead
        total_memory_gb = (model_weights_gb + kv_cache_gb + activation_gb) * (1 + self.config.MEMORY_OVERHEAD)
        
        return model_weights_gb, kv_cache_gb, total_memory_gb
    
    def calculate_throughput(
        self,
        model_config: ModelConfig,
        gpu_config: Optional[GPUConfig] = None
    ) -> tuple[float, float, float]:
        """
        Calculate throughput metrics.
        
        Returns: (tps_per_replica, ttft_ms, itl_ms)
        """
        gpu_type = gpu_config.gpu_type if gpu_config else None
        gpu_spec = self._get_gpu_spec(gpu_type)
        
        model_params = model_config.model_params
        bytes_per_param = model_config.bytes_per_param
        
        # Memory bandwidth limited throughput
        # TPS ≈ GPU_memory_bandwidth / (model_size_bytes / efficiency)
        model_size_bytes = model_params * bytes_per_param
        memory_bandwidth_bytes = gpu_spec["memory_bandwidth_gbps"] * (1024 ** 3)  # Convert to bytes/s
        
        # Tokens per second per GPU (memory bandwidth limited)
        tps_per_gpu = (
            memory_bandwidth_bytes * self.config.INFERENCE_EFFICIENCY
        ) / model_size_bytes
        
        # Adjust for batch size
        tps_per_replica = tps_per_gpu * model_config.batch_size
        
        # Time to first token (TTFT) - prefill phase
        # TTFT ≈ input_tokens × time_per_token_prefill
        # Prefill is compute-bound, so use FLOPs
        flops_per_token = 2 * model_params  # Approximate FLOPs per token
        gpu_flops = gpu_spec["fp16_tflops"] * 1e12  # Convert to FLOPS
        
        time_per_token_prefill_s = flops_per_token / (gpu_flops * self.config.INFERENCE_EFFICIENCY)
        ttft_ms = 50 + (time_per_token_prefill_s * 1000)  # Base latency + prefill time
        
        # Inter-token latency (ITL) - decode phase
        # ITL ≈ model_size / memory_bandwidth
        itl_ms = (model_size_bytes / memory_bandwidth_bytes) * 1000
        
        return tps_per_replica, ttft_ms, itl_ms
    
    def calculate_replica_count(
        self,
        workload: LLMWorkloadInput,
        model_config: ModelConfig,
        tps_per_replica: float
    ) -> tuple[int, int]:
        """
        Calculate required replica count.
        
        Returns: (min_replicas, max_replicas)
        """
        # Required TPS for target RPS
        required_tps = workload.requests_per_second * workload.avg_output_tokens
        
        # Apply peak load multiplier
        peak_tps = required_tps * workload.peak_load_multiplier
        
        # Calculate replicas needed
        min_replicas = max(1, math.ceil(required_tps / tps_per_replica))
        max_replicas = max(min_replicas, math.ceil(peak_tps / tps_per_replica))
        
        # Apply growth rate if specified
        if workload.growth_rate is not None and workload.growth_rate > 0:
            growth_multiplier = 1 + (workload.growth_rate / 100)
            max_replicas = max(max_replicas, math.ceil(max_replicas * growth_multiplier))
        
        # Ensure at least 2 replicas for HA
        if min_replicas == 1:
            min_replicas = 2
            max_replicas = max(max_replicas, 2)
        
        return min_replicas, max_replicas
    
    def recommend_gpu_type(
        self,
        total_memory_gb: float,
        gpu_config: Optional[GPUConfig] = None
    ) -> tuple[str, int]:
        """
        Recommend GPU type and count per replica.
        
        Returns: (gpu_type, gpus_per_replica)
        """
        if gpu_config and gpu_config.gpu_type:
            gpu_spec = self._get_gpu_spec(gpu_config.gpu_type)
            gpu_memory = gpu_spec["memory_gb"]
            gpus_needed = math.ceil(total_memory_gb / (gpu_memory * gpu_config.target_gpu_utilization))
            return gpu_config.gpu_type, max(1, gpus_needed)
        
        # Auto-select GPU based on memory requirements
        target_utilization = gpu_config.target_gpu_utilization if gpu_config else 0.7
        
        # Prefer fitting in single GPU if possible
        for gpu_type in ["A100-40GB", "A100-80GB", "H100-80GB"]:
            gpu_spec = self.config.GPU_SPECS[gpu_type]
            usable_memory = gpu_spec["memory_gb"] * target_utilization
            if total_memory_gb <= usable_memory:
                return gpu_type, 1
        
        # Need multiple GPUs - use H100 for best performance
        gpu_type = "H100-80GB"
        gpu_spec = self.config.GPU_SPECS[gpu_type]
        usable_memory = gpu_spec["memory_gb"] * target_utilization
        gpus_needed = math.ceil(total_memory_gb / usable_memory)
        
        return gpu_type, gpus_needed
    
    def calculate_gpu_resources(
        self,
        workload: LLMWorkloadInput,
        model_config: ModelConfig,
        gpu_config: Optional[GPUConfig] = None
    ) -> GPUResources:
        """Calculate complete GPU resource requirements."""
        # Calculate memory requirements
        model_weights_gb, kv_cache_gb, total_memory_gb = self.calculate_gpu_memory(
            model_config, gpu_config
        )
        
        # Recommend GPU type and count
        recommended_gpu, gpus_per_replica = self.recommend_gpu_type(
            total_memory_gb, gpu_config
        )
        
        # Calculate throughput
        tps_per_replica, _, _ = self.calculate_throughput(model_config, gpu_config)
        
        # Adjust TPS for multi-GPU setup
        if gpus_per_replica > 1:
            # With tensor parallelism, throughput doesn't scale linearly
            tps_per_replica *= (1 + 0.7 * (gpus_per_replica - 1))
        
        # Calculate replica count
        min_replicas, max_replicas = self.calculate_replica_count(
            workload, model_config, tps_per_replica
        )
        
        # Total GPUs needed
        total_gpus = min_replicas * gpus_per_replica
        
        return GPUResources(
            gpu_memory_per_replica_gb=total_memory_gb,
            min_gpus_per_replica=gpus_per_replica,
            recommended_gpu_type=recommended_gpu,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            total_gpus=total_gpus,
        )
    
    def calculate_throughput_metrics(
        self,
        workload: LLMWorkloadInput,
        model_config: ModelConfig,
        gpu_resources: GPUResources,
        gpu_config: Optional[GPUConfig] = None
    ) -> ThroughputMetrics:
        """Calculate throughput and latency metrics."""
        tps_per_replica, ttft_ms, itl_ms = self.calculate_throughput(
            model_config, gpu_config
        )
        
        # Adjust for multi-GPU
        if gpu_resources.min_gpus_per_replica > 1:
            tps_per_replica *= (1 + 0.7 * (gpu_resources.min_gpus_per_replica - 1))
        
        # Total capacity
        total_tps = tps_per_replica * gpu_resources.min_replicas
        max_rps = total_tps / workload.avg_output_tokens
        
        # Check latency SLA
        e2e_latency = ttft_ms + (workload.avg_output_tokens * itl_ms)
        meets_sla = True
        if model_config.target_latency_p99_ms:
            meets_sla = e2e_latency <= model_config.target_latency_p99_ms
        
        return ThroughputMetrics(
            tps_per_replica=tps_per_replica,
            total_tps_capacity=total_tps,
            max_rps_capacity=max_rps,
            estimated_ttft_ms=ttft_ms,
            estimated_itl_ms=itl_ms,
            meets_latency_sla=meets_sla,
        )
    
    def calculate_storage_resources(
        self,
        model_config: ModelConfig,
        gpu_resources: GPUResources
    ) -> LLMStorageResources:
        """Calculate storage requirements."""
        model_params = model_config.model_params
        bytes_per_param = model_config.bytes_per_param
        
        # Model storage (with overhead)
        model_storage_gb = (
            (model_params * bytes_per_param) / (1024 ** 3) * 
            self.config.MODEL_STORAGE_MULTIPLIER
        )
        
        # Cache storage (KV cache on disk if needed)
        gpu_spec = self._get_gpu_spec(gpu_resources.recommended_gpu_type)
        cache_storage_gb = (
            gpu_spec["memory_gb"] * gpu_resources.total_gpus * 
            self.config.CACHE_STORAGE_PER_GB_MEMORY
        )
        
        # Log storage
        log_storage_gb = self.config.LOG_STORAGE_GB
        
        total_storage_gb = model_storage_gb + cache_storage_gb + log_storage_gb
        
        return LLMStorageResources(
            model_storage_gb=model_storage_gb,
            cache_storage_gb=cache_storage_gb,
            log_storage_gb=log_storage_gb,
            total_storage_gb=total_storage_gb,
        )
    
    def calculate_network_resources(
        self,
        workload: LLMWorkloadInput,
    ) -> LLMNetworkResources:
        """Calculate network requirements."""
        # Bandwidth: tokens × bytes per token × RPS
        total_tokens_per_request = workload.avg_input_tokens + workload.avg_output_tokens
        bytes_per_request = total_tokens_per_request * self.config.BYTES_PER_TOKEN
        
        peak_rps = workload.requests_per_second * workload.peak_load_multiplier
        bandwidth_mbps = (bytes_per_request * peak_rps * 8) / (1024 * 1024)  # Convert to Mbps
        
        # Add safety margin
        bandwidth_mbps *= (1 + self.config.SAFETY_MARGIN)
        
        # Connection limits
        connection_limit = int(peak_rps * self.config.LLM_CONNECTION_LIMIT_PER_RPS)
        
        # Load balancers
        recommended_lbs = max(1, math.ceil(
            connection_limit / self.config.LLM_LOAD_BALANCER_CAPACITY
        ))
        
        return LLMNetworkResources(
            bandwidth_mbps=bandwidth_mbps,
            connection_limit=connection_limit,
            recommended_load_balancers=recommended_lbs,
        )
    
    def calculate_scaling_recommendations(
        self,
        gpu_resources: GPUResources,
        throughput: ThroughputMetrics
    ) -> LLMScalingRecommendations:
        """Calculate auto-scaling recommendations."""
        return LLMScalingRecommendations(
            recommended_min_replicas=gpu_resources.min_replicas,
            recommended_max_replicas=gpu_resources.max_replicas,
            auto_scale_rps_threshold=throughput.max_rps_capacity * self.config.LLM_AUTO_SCALING_RPS_THRESHOLD,
            auto_scale_gpu_util_threshold=self.config.LLM_AUTO_SCALING_GPU_UTIL_THRESHOLD,
            auto_scale_latency_threshold_ms=self.config.LLM_AUTO_SCALING_LATENCY_THRESHOLD_MS,
        )
    
    def calculate_cost_estimate(
        self,
        gpu_resources: GPUResources,
        throughput: ThroughputMetrics,
        storage: LLMStorageResources,
        network: LLMNetworkResources,
        gpu_config: Optional[GPUConfig] = None,
        service: Optional[ServiceInput] = None,
        include_cost: bool = False
    ) -> Optional[LLMCostEstimate]:
        """Calculate cost estimates."""
        if not include_cost:
            return None
        
        gpu_type = gpu_resources.recommended_gpu_type
        
        # Get GPU cost based on cloud provider
        cloud = service.cloud_provider if service else None
        if cloud and cloud in self.config.CLOUD_GPU_PRICING:
            gpu_cost_per_hour = self.config.CLOUD_GPU_PRICING[cloud].get(
                gpu_type, 
                self.config.GPU_SPECS[gpu_type]["cost_per_hour_usd"]
            )
        else:
            gpu_cost_per_hour = self.config.GPU_SPECS[gpu_type]["cost_per_hour_usd"]
        
        # Monthly GPU cost
        monthly_gpu_cost = (
            gpu_cost_per_hour * self.config.HOURS_PER_MONTH * gpu_resources.total_gpus
        )
        
        # Cost per million tokens
        monthly_tokens = throughput.total_tps_capacity * 3600 * 24 * 30
        if monthly_tokens > 0:
            cost_per_million = (monthly_gpu_cost / monthly_tokens) * 1_000_000
        else:
            cost_per_million = 0
        
        # Storage cost
        monthly_storage_cost = storage.total_storage_gb * self.config.STORAGE_COST_PER_GB_MONTHLY
        
        # Network cost (egress)
        monthly_network_cost = network.bandwidth_mbps * self.config.NETWORK_COST_PER_GB * 30
        
        return LLMCostEstimate(
            monthly_gpu_cost=monthly_gpu_cost,
            cost_per_million_tokens=cost_per_million,
            monthly_storage_cost=monthly_storage_cost,
            monthly_network_cost=monthly_network_cost,
        )
    
    def generate_inference_plan(
        self,
        workload: LLMWorkloadInput,
        model_config: ModelConfig,
        gpu_config: Optional[GPUConfig] = None,
        service: Optional[ServiceInput] = None,
        include_cost: bool = False
    ) -> LLMCapacityPlan:
        """Generate complete LLM inference capacity plan."""
        # Calculate all resources
        gpu_resources = self.calculate_gpu_resources(workload, model_config, gpu_config)
        throughput = self.calculate_throughput_metrics(
            workload, model_config, gpu_resources, gpu_config
        )
        storage = self.calculate_storage_resources(model_config, gpu_resources)
        network = self.calculate_network_resources(workload)
        scaling = self.calculate_scaling_recommendations(gpu_resources, throughput)
        cost = self.calculate_cost_estimate(
            gpu_resources, throughput, storage, network,
            gpu_config, service, include_cost
        )
        
        return LLMCapacityPlan(
            workload_input=workload,
            model_config=model_config,
            gpu_resources=gpu_resources,
            throughput=throughput,
            storage=storage,
            network=network,
            scaling=scaling,
            cost=cost,
            gpu_config=gpu_config,
            service=service,
        )
