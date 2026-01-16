"""
LLM Fine-Tuning/Training Forecasting Engine.

Calculates GPU memory, training duration, and cost for LLM training workloads.
"""

import math
from typing import Optional

from models import (
    TrainingInput,
    ModelConfig,
    GPUConfig,
    ServiceInput,
    TrainingGPUResources,
    TrainingMetrics,
    DataParallelConfig,
    TrainingStorageResources,
    TrainingCostEstimate,
    TrainingCapacityPlan,
)
import config


class TrainingEngine:
    """Engine for calculating LLM fine-tuning/training capacity requirements."""
    
    def __init__(self):
        self.config = config
    
    def _get_model_architecture(self, model_size_b: float) -> dict:
        """Get model architecture parameters based on size."""
        sizes = sorted(self.config.MODEL_ARCHITECTURE.keys())
        closest_size = min(sizes, key=lambda x: abs(x - model_size_b))
        
        arch = self.config.MODEL_ARCHITECTURE[closest_size].copy()
        
        if closest_size != model_size_b:
            scale = math.sqrt(model_size_b / closest_size)
            arch["hidden_dim"] = int(arch["hidden_dim"] * scale)
            arch["num_layers"] = int(arch["num_layers"] * scale)
        
        return arch
    
    def _get_gpu_spec(self, gpu_type: Optional[str]) -> dict:
        """Get GPU specifications."""
        if gpu_type and gpu_type in self.config.GPU_SPECS:
            return self.config.GPU_SPECS[gpu_type]
        return self.config.GPU_SPECS["A100-80GB"]
    
    def _get_precision_bytes(self, precision: str) -> float:
        """Get bytes per parameter for precision."""
        return self.config.PRECISION_BYTES.get(precision, 2.0)
    
    def _get_optimizer_multiplier(self, optimizer: str) -> float:
        """Get optimizer state memory multiplier."""
        return self.config.OPTIMIZER_MEMORY_MULTIPLIER.get(optimizer, 2.0)
    
    def calculate_gpu_memory(
        self,
        training_input: TrainingInput,
        model_config: ModelConfig,
        gpu_config: Optional[GPUConfig] = None
    ) -> tuple[float, float, float, float, float]:
        """
        Calculate GPU memory requirements for training.
        
        Returns: (model_weights_gb, gradients_gb, optimizer_states_gb, activations_gb, total_memory_gb)
        """
        model_params = model_config.model_params
        precision = training_input.training_precision
        bytes_per_param = self._get_precision_bytes(precision)
        
        # Model weights memory
        model_weights_gb = (model_params * bytes_per_param) / (1024 ** 3)
        
        # Gradients memory (same size as model weights)
        gradients_gb = model_weights_gb * self.config.GRADIENT_MEMORY_RATIO
        
        # Optimizer states memory
        optimizer_multiplier = self._get_optimizer_multiplier(training_input.optimizer_type)
        optimizer_states_gb = model_weights_gb * optimizer_multiplier
        
        # Activations memory
        # Activations = sequence_length × batch_size × hidden_dim × num_layers × bytes
        arch = self._get_model_architecture(model_config.model_size_billions)
        activations_bytes = (
            training_input.sequence_length *
            training_input.micro_batch_size *
            arch["hidden_dim"] *
            arch["num_layers"] *
            bytes_per_param
        )
        activations_gb = activations_bytes / (1024 ** 3)
        
        # Apply gradient checkpointing savings if enabled
        if training_input.gradient_checkpointing:
            activations_gb *= (1 - self.config.GRADIENT_CHECKPOINTING_SAVINGS)
        
        # Total with overhead
        total_memory_gb = (
            model_weights_gb + gradients_gb + optimizer_states_gb + activations_gb
        ) * (1 + self.config.MEMORY_OVERHEAD)
        
        return model_weights_gb, gradients_gb, optimizer_states_gb, activations_gb, total_memory_gb
    
    def calculate_required_gpus(
        self,
        training_input: TrainingInput,
        model_config: ModelConfig,
        total_memory_gb: float,
        gpu_config: Optional[GPUConfig] = None
    ) -> tuple[int, str, int, int]:
        """
        Calculate required number of GPUs.
        
        Returns: (required_gpus, recommended_gpu_type, gpus_per_node, num_nodes)
        """
        # Determine GPU type
        gpu_type = gpu_config.gpu_type if gpu_config else None
        target_util = gpu_config.target_gpu_utilization if gpu_config else 0.85
        
        if not gpu_type:
            # Auto-select based on memory requirements
            for candidate in ["A100-80GB", "H100-80GB"]:
                gpu_spec = self.config.GPU_SPECS[candidate]
                if total_memory_gb <= gpu_spec["memory_gb"] * target_util:
                    gpu_type = candidate
                    break
            if not gpu_type:
                gpu_type = "H100-80GB"
        
        gpu_spec = self._get_gpu_spec(gpu_type)
        usable_memory = gpu_spec["memory_gb"] * target_util
        
        # Calculate minimum GPUs needed for memory
        gpus_for_memory = math.ceil(total_memory_gb / usable_memory)
        
        # Calculate GPUs needed for data parallelism
        # Data parallel size = global_batch_size / (micro_batch_size × gradient_accum_steps)
        effective_batch_per_gpu = (
            training_input.micro_batch_size * 
            training_input.gradient_accumulation_steps
        )
        gpus_for_dp = math.ceil(training_input.global_batch_size / effective_batch_per_gpu)
        
        # Take maximum of memory and data parallel requirements
        required_gpus = max(gpus_for_memory, gpus_for_dp)
        
        # Ensure at least 1 GPU
        required_gpus = max(1, required_gpus)
        
        # Calculate nodes (assuming 8 GPUs per node)
        gpus_per_node = 8
        num_nodes = math.ceil(required_gpus / gpus_per_node)
        
        # Adjust to fit node boundaries
        if num_nodes > 1:
            required_gpus = num_nodes * gpus_per_node
        
        return required_gpus, gpu_type, gpus_per_node, num_nodes
    
    def calculate_gpu_resources(
        self,
        training_input: TrainingInput,
        model_config: ModelConfig,
        gpu_config: Optional[GPUConfig] = None
    ) -> TrainingGPUResources:
        """Calculate complete GPU resource requirements for training."""
        # Calculate memory breakdown
        (
            model_weights_gb,
            gradients_gb,
            optimizer_states_gb,
            activations_gb,
            total_memory_gb
        ) = self.calculate_gpu_memory(training_input, model_config, gpu_config)
        
        # Calculate GPU requirements
        required_gpus, gpu_type, gpus_per_node, num_nodes = self.calculate_required_gpus(
            training_input, model_config, total_memory_gb, gpu_config
        )
        
        # Per-GPU memory (distributed across GPUs if using model parallelism)
        gpu_memory_per_gpu = total_memory_gb / max(1, required_gpus // 2)  # Rough estimate
        
        return TrainingGPUResources(
            gpu_memory_per_gpu_gb=gpu_memory_per_gpu,
            model_weights_gb=model_weights_gb,
            gradients_gb=gradients_gb,
            optimizer_states_gb=optimizer_states_gb,
            activations_gb=activations_gb,
            required_gpus=required_gpus,
            recommended_gpu_type=gpu_type,
            gpus_per_node=gpus_per_node,
            num_nodes=num_nodes,
        )
    
    def calculate_training_metrics(
        self,
        training_input: TrainingInput,
        model_config: ModelConfig,
        gpu_resources: TrainingGPUResources,
        gpu_config: Optional[GPUConfig] = None
    ) -> TrainingMetrics:
        """Calculate training throughput and duration metrics."""
        gpu_spec = self._get_gpu_spec(gpu_resources.recommended_gpu_type)
        
        model_params = model_config.model_params
        
        # Tokens per second per GPU (compute-bound estimate)
        # Based on GPU FLOPs and model size
        # FLOPs per token ≈ 6 × model_params (forward + backward)
        flops_per_token = 6 * model_params
        gpu_flops = gpu_spec["fp16_tflops"] * 1e12
        
        tokens_per_sec_per_gpu = (
            gpu_flops * self.config.TRAINING_EFFICIENCY
        ) / flops_per_token
        
        # Apply gradient checkpointing slowdown if enabled
        if training_input.gradient_checkpointing:
            tokens_per_sec_per_gpu *= (1 - self.config.GRADIENT_CHECKPOINTING_SLOWDOWN)
        
        # Total throughput across all GPUs
        # Note: Data parallelism scales nearly linearly
        total_tokens_per_sec = tokens_per_sec_per_gpu * gpu_resources.required_gpus
        
        # Training steps calculation
        tokens_per_step = training_input.global_batch_size * training_input.sequence_length
        steps_per_epoch = math.ceil(training_input.dataset_size_tokens / tokens_per_step)
        total_steps = steps_per_epoch * training_input.num_epochs
        
        # Duration calculation
        total_tokens = training_input.dataset_size_tokens * training_input.num_epochs
        duration_seconds = total_tokens / total_tokens_per_sec
        duration_hours = duration_seconds / 3600
        duration_days = duration_hours / 24
        
        return TrainingMetrics(
            tokens_per_second_per_gpu=tokens_per_sec_per_gpu,
            total_tokens_per_second=total_tokens_per_sec,
            steps_per_epoch=steps_per_epoch,
            total_steps=total_steps,
            estimated_duration_hours=duration_hours,
            estimated_duration_days=duration_days,
        )
    
    def calculate_data_parallel_config(
        self,
        training_input: TrainingInput,
        gpu_resources: TrainingGPUResources
    ) -> DataParallelConfig:
        """Calculate data parallelism configuration."""
        # Data parallel size = number of GPUs participating in data parallelism
        data_parallel_size = gpu_resources.required_gpus
        
        # Effective batch size per step
        effective_batch_size = (
            training_input.micro_batch_size *
            training_input.gradient_accumulation_steps *
            data_parallel_size
        )
        
        return DataParallelConfig(
            data_parallel_size=data_parallel_size,
            effective_batch_size=effective_batch_size,
            gradient_accumulation_steps=training_input.gradient_accumulation_steps,
        )
    
    def calculate_storage_resources(
        self,
        training_input: TrainingInput,
        model_config: ModelConfig
    ) -> TrainingStorageResources:
        """Calculate storage requirements for training."""
        model_params = model_config.model_params
        bytes_per_param = self._get_precision_bytes(training_input.training_precision)
        
        # Model checkpoint storage (model + optimizer state)
        model_size_gb = (model_params * bytes_per_param) / (1024 ** 3)
        optimizer_multiplier = self._get_optimizer_multiplier(training_input.optimizer_type)
        checkpoint_gb = model_size_gb * (1 + optimizer_multiplier) * self.config.MODEL_STORAGE_MULTIPLIER
        
        # Dataset storage (estimate 4 bytes per token average)
        dataset_gb = (training_input.dataset_size_tokens * 4) / (1024 ** 3)
        
        total_storage_gb = checkpoint_gb + dataset_gb
        
        return TrainingStorageResources(
            model_checkpoint_gb=checkpoint_gb,
            dataset_storage_gb=dataset_gb,
            total_storage_gb=total_storage_gb,
        )
    
    def calculate_cost_estimate(
        self,
        gpu_resources: TrainingGPUResources,
        training_metrics: TrainingMetrics,
        service: Optional[ServiceInput] = None,
        include_cost: bool = False
    ) -> Optional[TrainingCostEstimate]:
        """Calculate training cost estimates."""
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
        
        # Total cost for training run
        total_gpu_hours = training_metrics.estimated_duration_hours * gpu_resources.required_gpus
        total_training_cost = gpu_cost_per_hour * total_gpu_hours
        
        # Cost per epoch
        cost_per_epoch = total_training_cost / training_metrics.total_steps * training_metrics.steps_per_epoch
        
        return TrainingCostEstimate(
            gpu_cost_per_hour=gpu_cost_per_hour,
            total_training_cost=total_training_cost,
            cost_per_epoch=cost_per_epoch,
            currency="USD",
        )
    
    def generate_training_plan(
        self,
        training_input: TrainingInput,
        model_config: ModelConfig,
        gpu_config: Optional[GPUConfig] = None,
        service: Optional[ServiceInput] = None,
        include_cost: bool = False
    ) -> TrainingCapacityPlan:
        """Generate complete LLM training capacity plan."""
        # Calculate all resources
        gpu_resources = self.calculate_gpu_resources(
            training_input, model_config, gpu_config
        )
        
        training_metrics = self.calculate_training_metrics(
            training_input, model_config, gpu_resources, gpu_config
        )
        
        data_parallel_config = self.calculate_data_parallel_config(
            training_input, gpu_resources
        )
        
        storage = self.calculate_storage_resources(training_input, model_config)
        
        cost = self.calculate_cost_estimate(
            gpu_resources, training_metrics, service, include_cost
        )
        
        return TrainingCapacityPlan(
            training_input=training_input,
            model_config=model_config,
            gpu_resources=gpu_resources,
            training_metrics=training_metrics,
            data_parallel_config=data_parallel_config,
            storage=storage,
            cost=cost,
            gpu_config=gpu_config,
            service=service,
        )
