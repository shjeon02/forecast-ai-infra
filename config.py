"""
Configuration parameters for infrastructure capacity forecasting.
"""

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
