"""
Configuration parameters for infrastructure capacity forecasting.

Supports both legacy web application and LLM-specific workloads.
"""

# =============================================================================
# Legacy Web Application Configuration
# =============================================================================

# Default resource requirements per user
CPU_PER_CONCURRENT_USER = 0.01  # cores
MEMORY_PER_CONCURRENT_USER = 50  # MB
AVERAGE_DATA_PER_USER = 10  # MB
AVERAGE_FILE_SIZE_PER_USER = 50  # MB
BANDWIDTH_PER_CONCURRENT_USER = 1  # Mbps

# Base resource requirements
BASE_CPU_CORES = 2.0
BASE_MEMORY_GB = 4.0

# Scaling factors
DEFAULT_PEAK_LOAD_MULTIPLIER = 1.5
SAFETY_MARGIN = 0.25  # 25% safety margin
GROWTH_FACTOR = 1.2  # 20% growth buffer

# Storage factors
BACKUP_MULTIPLIER = 0.3  # 30% of primary storage for backups
DATA_RETENTION_DAILY_GROWTH = 0.001  # 0.1% daily growth

# Network factors
CONNECTION_LIMIT_PER_USER = 10
LOAD_BALANCER_CAPACITY = 10000  # concurrent connections per load balancer

# Instance type recommendations (CPU cores, Memory GB)
INSTANCE_TYPES = {
    "small": (2, 4),
    "medium": (4, 8),
    "large": (8, 16),
    "xlarge": (16, 32),
    "2xlarge": (32, 64),
    "4xlarge": (64, 128),
}

# Auto-scaling thresholds
AUTO_SCALING_CPU_THRESHOLD = 70.0  # percentage
AUTO_SCALING_MEMORY_THRESHOLD = 75.0  # percentage

# Cost estimates (per month, USD) - approximate
COST_PER_CPU_CORE_MONTHLY = 50.0
COST_PER_GB_MEMORY_MONTHLY = 10.0
COST_PER_GB_STORAGE_MONTHLY = 0.10
COST_PER_MBPS_BANDWIDTH_MONTHLY = 0.10
COST_PER_LOAD_BALANCER_MONTHLY = 20.0


# =============================================================================
# LLM / GPU Configuration
# =============================================================================

# GPU Specifications: {gpu_type: (memory_gb, memory_bandwidth_gbps, fp16_tflops, cost_per_hour_usd)}
GPU_SPECS = {
    "A100-40GB": {
        "memory_gb": 40,
        "memory_bandwidth_gbps": 1555,
        "fp16_tflops": 312,
        "cost_per_hour_usd": 1.10,  # AWS on-demand approximate
    },
    "A100-80GB": {
        "memory_gb": 80,
        "memory_bandwidth_gbps": 2039,
        "fp16_tflops": 312,
        "cost_per_hour_usd": 1.60,
    },
    "H100-80GB": {
        "memory_gb": 80,
        "memory_bandwidth_gbps": 3350,
        "fp16_tflops": 990,
        "cost_per_hour_usd": 3.00,
    },
    "H100-SXM": {
        "memory_gb": 80,
        "memory_bandwidth_gbps": 3350,
        "fp16_tflops": 990,
        "cost_per_hour_usd": 3.50,
    },
    "L4": {
        "memory_gb": 24,
        "memory_bandwidth_gbps": 300,
        "fp16_tflops": 121,
        "cost_per_hour_usd": 0.50,
    },
    "T4": {
        "memory_gb": 16,
        "memory_bandwidth_gbps": 320,
        "fp16_tflops": 65,
        "cost_per_hour_usd": 0.35,
    },
    "V100": {
        "memory_gb": 32,
        "memory_bandwidth_gbps": 900,
        "fp16_tflops": 125,
        "cost_per_hour_usd": 0.90,
    },
}

# Cloud Provider GPU Pricing (hourly rates in USD)
CLOUD_GPU_PRICING = {
    "aws": {
        "A100-40GB": 1.10,
        "A100-80GB": 1.60,
        "H100-80GB": 3.00,
        "L4": 0.50,
        "T4": 0.35,
        "V100": 0.90,
    },
    "gcp": {
        "A100-40GB": 1.00,
        "A100-80GB": 1.50,
        "H100-80GB": 2.85,
        "L4": 0.45,
        "T4": 0.30,
        "V100": 0.85,
    },
    "azure": {
        "A100-80GB": 1.70,
        "H100-80GB": 3.20,
        "T4": 0.40,
        "V100": 0.95,
    },
}

# Precision bytes per parameter
PRECISION_BYTES = {
    "FP32": 4.0,
    "FP16": 2.0,
    "BF16": 2.0,
    "INT8": 1.0,
    "INT4": 0.5,
    "FP8": 1.0,
}

# Optimizer memory multipliers (relative to model weights)
# For training: Optimizer States = Model Weights × Multiplier
OPTIMIZER_MEMORY_MULTIPLIER = {
    "Adam": 2.0,      # momentum + variance (2 states)
    "AdamW": 2.0,     # same as Adam
    "Adafactor": 0.5, # factorized second moments
    "SGD": 0.0,       # no optimizer state
}

# LLM Model Architecture Constants (approximate)
# Based on typical transformer architectures
MODEL_ARCHITECTURE = {
    # Model size in billions -> (num_layers, hidden_dim, num_heads)
    0.5: {"num_layers": 24, "hidden_dim": 1024, "num_heads": 16},
    1.0: {"num_layers": 24, "hidden_dim": 2048, "num_heads": 16},
    7: {"num_layers": 32, "hidden_dim": 4096, "num_heads": 32},
    13: {"num_layers": 40, "hidden_dim": 5120, "num_heads": 40},
    30: {"num_layers": 60, "hidden_dim": 6656, "num_heads": 52},
    65: {"num_layers": 80, "hidden_dim": 8192, "num_heads": 64},
    70: {"num_layers": 80, "hidden_dim": 8192, "num_heads": 64},
    405: {"num_layers": 126, "hidden_dim": 16384, "num_heads": 128},
}

# LLM Inference Constants
INFERENCE_EFFICIENCY = 0.7  # GPU utilization efficiency during inference
KV_CACHE_OVERHEAD = 0.1    # 10% overhead for KV cache management
ACTIVATION_MEMORY_RATIO = 0.15  # 15% of model weights for inference activations
MEMORY_OVERHEAD = 0.1      # 10% general memory overhead

# LLM Training Constants
TRAINING_EFFICIENCY = 0.5   # GPU utilization efficiency during training
GRADIENT_MEMORY_RATIO = 1.0  # Gradients same size as weights
ACTIVATION_MEMORY_TRAINING_RATIO = 0.5  # Activations for training (varies with batch/seq)
GRADIENT_CHECKPOINTING_SAVINGS = 0.33  # 33% memory reduction with gradient checkpointing
GRADIENT_CHECKPOINTING_SLOWDOWN = 0.20  # 20% slower with gradient checkpointing

# LLM Auto-scaling Thresholds
LLM_AUTO_SCALING_RPS_THRESHOLD = 0.8      # 80% of capacity
LLM_AUTO_SCALING_GPU_UTIL_THRESHOLD = 0.7  # 70% GPU utilization
LLM_AUTO_SCALING_LATENCY_THRESHOLD_MS = 500  # P99 latency threshold

# LLM Network Constants
BYTES_PER_TOKEN = 4  # Average bytes per token for network calculation
LLM_CONNECTION_LIMIT_PER_RPS = 100  # Concurrent connections per RPS
LLM_LOAD_BALANCER_CAPACITY = 50000  # Connections per LB for LLM workloads

# Storage Constants
MODEL_STORAGE_MULTIPLIER = 1.2  # 20% overhead for model storage
CACHE_STORAGE_PER_GB_MEMORY = 0.5  # Cache storage relative to GPU memory
LOG_STORAGE_GB = 10.0  # Default log storage

# Cost Constants
HOURS_PER_MONTH = 720  # 24 * 30
STORAGE_COST_PER_GB_MONTHLY = 0.08  # General storage cost
NETWORK_COST_PER_GB = 0.09  # Egress cost per GB
