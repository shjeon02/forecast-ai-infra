# Requirements

## Functional Requirements

### FR-1: User Input Collection

#### FR-1.1: LLM Workload Inputs (Required)
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1.1.1 | System SHALL accept requests per second (RPS) as float input | P0 |
| FR-1.1.2 | System SHALL accept average input tokens per request as integer | P0 |
| FR-1.1.3 | System SHALL accept average output tokens per request as integer | P0 |
| FR-1.1.4 | System SHALL validate that RPS > 0 | P0 |
| FR-1.1.5 | System SHALL validate that tokens per request > 0 | P0 |

#### FR-1.2: Model Configuration Inputs (Required)
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1.2.1 | System SHALL accept model size in parameters (e.g., 7B, 70B, 405B) | P0 |
| FR-1.2.2 | System SHALL accept model precision/quantization (FP32, FP16, BF16, INT8, INT4) | P0 |
| FR-1.2.3 | System SHOULD accept context window size (default: 4096) | P1 |
| FR-1.2.4 | System SHOULD accept target latency SLA (P50, P99 in ms) | P1 |
| FR-1.2.5 | System SHOULD accept batch size (default: 1) | P1 |

#### FR-1.3: Optional Scaling Inputs
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1.3.1 | System SHOULD accept peak load multiplier as float (1.0-10.0) | P1 |
| FR-1.3.2 | System SHOULD accept growth rate as percentage (0-1000%) | P1 |
| FR-1.3.3 | System SHOULD accept target GPU utilization percentage (default: 70%) | P2 |

#### FR-1.4: GPU/Infrastructure Inputs
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1.4.1 | System SHOULD accept GPU type (A100-40GB, A100-80GB, H100, etc.) | P1 |
| FR-1.4.2 | System SHOULD accept cloud provider (aws, gcp, azure) | P1 |
| FR-1.4.3 | System SHOULD accept region specification | P2 |
| FR-1.4.4 | System SHOULD accept current replica count (for scaling recommendations) | P2 |

#### FR-1.5: Service Metadata Inputs (Advanced Mode)
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1.5.1 | System SHOULD accept service identifier (service_id) | P1 |
| FR-1.5.2 | System SHOULD accept team identifier (team_id) | P2 |
| FR-1.5.3 | System SHOULD accept environment type (prod, staging, dev) | P1 |
| FR-1.5.4 | System SHOULD accept criticality level (high, medium, low) | P1 |

#### FR-1.6: Historical Data Inputs (Time-Series Forecasting)
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1.6.1 | System SHOULD accept historical RPS metrics (time-series) | P1 |
| FR-1.6.2 | System SHOULD accept historical token throughput (TPS) | P1 |
| FR-1.6.3 | System SHOULD accept historical latency metrics (TTFT, ITL) | P1 |
| FR-1.6.4 | System SHOULD accept historical cost data (time-series) | P1 |
| FR-1.6.5 | System SHOULD accept forecast horizon in days/months | P1 |

#### FR-1.7: Scenario Analysis Inputs
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1.7.1 | System SHOULD accept scenario type (optimistic, pessimistic, baseline) | P2 |
| FR-1.7.2 | System SHOULD accept traffic spike simulation parameters | P2 |
| FR-1.7.3 | System SHOULD accept budget constraints for comparison | P2 |

#### FR-1.8: Workload Type Selection
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1.8.1 | System SHALL accept workload type (inference, training) | P0 |
| FR-1.8.2 | System SHALL default to inference mode if not specified | P0 |
| FR-1.8.3 | System SHALL validate inputs based on selected workload type | P0 |

#### FR-1.9: Fine-Tuning Inputs (Training Mode)
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1.9.1 | System SHALL accept dataset size in tokens | P0 |
| FR-1.9.2 | System SHALL accept sequence length | P0 |
| FR-1.9.3 | System SHALL accept number of training epochs | P0 |
| FR-1.9.4 | System SHALL accept global batch size | P0 |
| FR-1.9.5 | System SHOULD accept micro batch size (default: 1) | P1 |
| FR-1.9.6 | System SHOULD accept gradient accumulation steps | P1 |
| FR-1.9.7 | System SHOULD accept optimizer type (Adam, AdamW, Adafactor, SGD) | P1 |
| FR-1.9.8 | System SHOULD accept gradient checkpointing flag (default: false) | P1 |
| FR-1.9.9 | System SHOULD accept training precision (FP32, FP16, BF16, FP8) | P1 |

### FR-2: Capacity Forecasting

#### FR-2.1: GPU Resources
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-2.1.1 | System SHALL calculate required GPU memory | P0 |
| FR-2.1.2 | System SHALL calculate required number of GPUs | P0 |
| FR-2.1.3 | System SHALL recommend appropriate GPU types | P0 |
| FR-2.1.4 | System SHALL calculate minimum and maximum model replicas | P0 |

#### FR-2.2: Throughput Calculations
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-2.2.1 | System SHALL calculate tokens per second (TPS) per replica | P0 |
| FR-2.2.2 | System SHALL calculate total TPS capacity | P0 |
| FR-2.2.3 | System SHALL calculate estimated latency (TTFT, ITL) | P1 |
| FR-2.2.4 | System SHALL validate against latency SLA | P1 |

#### FR-2.3: Storage Resources
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-2.3.1 | System SHALL calculate model storage requirements | P0 |
| FR-2.3.2 | System SHALL calculate log/monitoring storage requirements | P1 |
| FR-2.3.3 | System SHALL calculate cache storage requirements | P2 |

#### FR-2.4: Network Resources
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-2.4.1 | System SHALL calculate bandwidth requirements | P1 |
| FR-2.4.2 | System SHALL calculate connection limits | P1 |
| FR-2.4.3 | System SHALL recommend load balancer count | P1 |

#### FR-2.5: Scaling Recommendations (Inference)
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-2.5.1 | System SHALL provide horizontal scaling recommendations (replicas) | P0 |
| FR-2.5.2 | System SHALL provide auto-scaling thresholds (RPS, GPU util, latency) | P0 |
| FR-2.5.3 | System SHALL recommend batch size for target throughput | P1 |

#### FR-2.6: Fine-Tuning Capacity Calculations (Training Mode)
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-2.6.1 | System SHALL calculate GPU memory for training (weights + gradients + optimizer + activations) | P0 |
| FR-2.6.2 | System SHALL calculate required number of GPUs for training | P0 |
| FR-2.6.3 | System SHALL calculate training throughput (tokens/sec) | P0 |
| FR-2.6.4 | System SHALL estimate training duration (hours/days) | P0 |
| FR-2.6.5 | System SHOULD recommend data parallel configuration | P1 |
| FR-2.6.6 | System SHOULD calculate memory savings with gradient checkpointing | P2 |

### FR-3: Cost Estimation

#### FR-3.1: Inference Cost Estimation
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-3.1.1 | System SHOULD calculate monthly GPU costs (inference) | P1 |
| FR-3.1.2 | System SHOULD calculate cost per 1M tokens (inference) | P1 |
| FR-3.1.3 | System SHOULD calculate monthly infrastructure costs | P2 |

#### FR-3.2: Fine-Tuning Cost Estimation
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-3.2.1 | System SHOULD calculate total training GPU cost | P1 |
| FR-3.2.2 | System SHOULD calculate cost per epoch | P1 |
| FR-3.2.3 | System SHOULD calculate cost per training run | P1 |

### FR-4: User Interface

#### FR-4.1: Interactive Mode (Agent/Chatbot)
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-4.1.1 | System SHALL provide conversational prompts for input collection | P0 |
| FR-4.1.2 | System SHALL display validation errors with clear messages | P0 |
| FR-4.1.3 | System SHALL format output in human-readable format | P0 |
| FR-4.1.4 | System SHALL allow graceful exit via keyboard interrupt | P1 |

#### FR-4.2: Command-Line Mode
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-4.2.1 | System SHALL accept all inputs as command-line arguments | P0 |
| FR-4.2.2 | System SHALL support `--help` for usage information | P0 |
| FR-4.2.3 | System SHALL support non-interactive execution | P0 |

#### FR-4.3: JSON Configuration Mode
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-4.3.1 | System SHALL accept JSON configuration file via `--config` argument | P1 |
| FR-4.3.2 | System SHALL validate JSON schema before processing | P1 |
| FR-4.3.3 | System SHALL allow CLI arguments to override JSON config values | P1 |
| FR-4.3.4 | System SHALL provide `--generate-config` to create template files | P2 |
| FR-4.3.5 | System SHALL support both inference and training JSON schemas | P1 |
| FR-4.3.6 | System SHALL display clear error messages for invalid JSON | P1 |

#### FR-4.4: Input Mode Selection
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-4.4.1 | System SHALL default to interactive mode when no arguments provided | P0 |
| FR-4.4.2 | System SHALL use JSON mode when `--config` is provided | P0 |
| FR-4.4.3 | System SHALL use CLI mode when required arguments are provided | P0 |
| FR-4.4.4 | System SHALL support mixed mode (JSON + CLI overrides) | P1 |

### FR-5: Export Capabilities

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-5.1 | System SHALL export capacity plan to JSON format | P0 |
| FR-5.2 | System SHALL allow custom output filename | P1 |
| FR-5.3 | System SHOULD support CSV export | P2 |
| FR-5.4 | System SHOULD support PDF export | P3 |

---

## Non-Functional Requirements

### NFR-1: Performance

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-1.1 | Calculation time for capacity plan | < 1 second |
| NFR-1.2 | Memory usage during execution | < 50 MB |
| NFR-1.3 | Startup time | < 2 seconds |

### NFR-2: Usability

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-2.1 | Time to complete interactive flow | < 2 minutes |
| NFR-2.2 | Error message clarity | User can correct issue without documentation |
| NFR-2.3 | Output readability | Scannable within 30 seconds |

### NFR-3: Reliability

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-3.1 | Graceful handling of invalid input | 100% coverage |
| NFR-3.2 | No crashes on edge cases | Zero crash rate |
| NFR-3.3 | Consistent output for same input | 100% deterministic |

### NFR-4: Maintainability

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-4.1 | Code documentation coverage | >80% of functions |
| NFR-4.2 | Configuration externalization | All magic numbers in config |
| NFR-4.3 | Modular architecture | Single responsibility per module |

### NFR-5: Portability

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-5.1 | Python version compatibility | 3.10+ |
| NFR-5.2 | OS compatibility | macOS, Linux, Windows |
| NFR-5.3 | External dependencies | Standard library only (core) |

### NFR-6: Security

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-6.1 | No sensitive data in logs | Zero sensitive data exposure |
| NFR-6.2 | Safe file operations | Path traversal prevention |
| NFR-6.3 | Input sanitization | All user inputs validated |

---

## Data Requirements

### Input Data Models

#### LLM Workload Input (Required)
```
LLMWorkloadInput:
  requests_per_second: float (required, > 0)
  avg_input_tokens: int (required, > 0)
  avg_output_tokens: int (required, > 0)
  peak_load_multiplier: float (optional, default 1.5)
  growth_rate: float (optional, 0-1000%)
```

#### Model Configuration (Required)
```
ModelConfig:
  model_size_billions: float (required, e.g., 7, 70, 405)
  precision: string (required, enum: FP32|FP16|BF16|INT8|INT4)
  context_window: int (optional, default 4096)
  batch_size: int (optional, default 1)
  target_latency_p50_ms: float (optional)
  target_latency_p99_ms: float (optional)
```

#### GPU Configuration
```
GPUConfig:
  gpu_type: string (optional, e.g., A100-40GB, A100-80GB, H100-80GB)
  gpu_memory_gb: float (derived from gpu_type if not specified)
  target_gpu_utilization: float (optional, default 0.7)
  current_replicas: int (optional)
```

#### Service Metadata (Advanced Mode)
```
ServiceInput:
  service_id: string (optional, unique identifier)
  service_name: string (optional)
  team_id: string (optional)
  environment: string (optional, enum: prod|staging|dev)
  criticality: string (optional, enum: high|medium|low)
  cloud_provider: string (optional, enum: aws|gcp|azure)
  region: string (optional, e.g., "us-east-1")
```

#### Historical Data (Time-Series Forecasting)
```
LLMUsageMetric:
  service_id: string (reference to service)
  timestamp: datetime
  metric_name: string (enum: rps|tps|ttft_ms|itl_ms|gpu_utilization)
  value: float
  unit: string (e.g., requests/sec, tokens/sec, ms, percent)

CostHistory:
  service_id: string (reference to service)
  timestamp: datetime
  amount: float
  currency: string (default: USD)
  category: string (enum: gpu_compute|storage|network)
  tokens_processed: int (optional, for cost-per-token calculation)
```

#### Forecasting Parameters
```
ForecastConfig:
  horizon_days: int (optional, default 30)
  horizon_months: int (optional, default 12)
  seasonal_period: int (optional, default 7 for weekly patterns)
  scenario: string (optional, enum: baseline|optimistic|pessimistic|spike)
  variance_threshold_pct: float (optional, default 10.0, for retraining trigger)
```

#### Fine-Tuning Input (Training Mode)
```
TrainingInput:
  workload_type: string (required, value: "training")
  dataset_size_tokens: int (required, total tokens in dataset)
  sequence_length: int (required, e.g., 2048, 4096, 8192)
  num_epochs: int (required, e.g., 1, 3, 5)
  global_batch_size: int (required, e.g., 32, 64, 128)
  micro_batch_size: int (optional, default 1)
  gradient_accumulation_steps: int (optional, default 1)
  optimizer_type: string (optional, enum: Adam|AdamW|Adafactor|SGD, default Adam)
  gradient_checkpointing: boolean (optional, default false)
  training_precision: string (optional, enum: FP32|FP16|BF16|FP8, default FP16)
```

### Output Data Model

```
LLMCapacityPlan:
  workload_input: LLMWorkloadInput
  model_config: ModelConfig
  
  gpu_resources:
    gpu_memory_per_replica_gb: float
    min_gpus_per_replica: int
    recommended_gpu_type: string
    min_replicas: int
    max_replicas: int
    total_gpus: int
  
  throughput:
    tps_per_replica: float
    total_tps_capacity: float
    max_rps_capacity: float
    estimated_ttft_ms: float
    estimated_itl_ms: float
    meets_latency_sla: boolean
  
  storage:
    model_storage_gb: float
    cache_storage_gb: float
    total_storage_gb: float
  
  network:
    bandwidth_mbps: float
    connection_limit: int
    recommended_load_balancers: int
  
  scaling:
    recommended_min_replicas: int
    recommended_max_replicas: int
    auto_scale_rps_threshold: float
    auto_scale_gpu_util_threshold: float
    auto_scale_latency_threshold_ms: float
  
  cost:
    monthly_gpu_cost: float
    cost_per_million_tokens: float
    monthly_total_cost: float
    currency: string
```

#### Training Capacity Plan Output (Training Mode)
```
TrainingCapacityPlan:
  training_input: TrainingInput
  model_config: ModelConfig
  
  gpu_resources:
    gpu_memory_per_gpu_gb: float
    model_weights_gb: float
    gradients_gb: float
    optimizer_states_gb: float
    activations_gb: float
    required_gpus: int
    recommended_gpu_type: string
    gpus_per_node: int
    num_nodes: int
  
  training_metrics:
    tokens_per_second_per_gpu: float
    total_tokens_per_second: float
    steps_per_epoch: int
    total_steps: int
    estimated_duration_hours: float
    estimated_duration_days: float
  
  data_parallel_config:
    data_parallel_size: int
    effective_batch_size: int
    gradient_accumulation_steps: int
  
  storage:
    model_checkpoint_gb: float
    dataset_storage_gb: float
    total_storage_gb: float
  
  cost:
    gpu_cost_per_hour: float
    total_training_cost: float
    cost_per_epoch: float
    currency: string
```

---

## Interface Requirements

### CLI Interface

```bash
# Interactive mode (default when no arguments provided)
python main.py

# JSON configuration mode
python commands/forecast.py --config inference_config.json
python commands/forecast.py --config training_config.json

# JSON with CLI overrides (CLI args take precedence)
python commands/forecast.py --config inference_config.json --rps 20.0 --cost

# Generate template configuration files
python commands/forecast.py --generate-config inference  # creates inference_config.json
python commands/forecast.py --generate-config training   # creates training_config.json

# Inference mode (non-interactive CLI args)
python commands/forecast.py \
  --mode inference \
  --rps 10.0 \
  --input-tokens 500 \
  --output-tokens 200 \
  --model-size 70 \
  --precision FP16 \
  [--gpu-type A100-80GB] \
  [--context-window 8192] \
  [--batch-size 4] \
  [--cost] \
  [--output <filepath>]

# Fine-tuning mode (non-interactive CLI args)
python commands/forecast.py \
  --mode training \
  --dataset-size 1000000000 \
  --sequence-length 4096 \
  --epochs 3 \
  --global-batch-size 64 \
  --model-size 70 \
  --precision BF16 \
  [--optimizer Adam] \
  [--gradient-checkpointing] \
  [--gpu-type A100-80GB] \
  [--cost] \
  [--output <filepath>]
```

### JSON Configuration Schema

#### Inference Configuration (`inference_config.json`)
```json
{
  "mode": "inference",
  "workload": {
    "requests_per_second": 10.0,
    "avg_input_tokens": 500,
    "avg_output_tokens": 200,
    "peak_load_multiplier": 1.5,
    "growth_rate": 20.0
  },
  "model": {
    "model_size_billions": 70,
    "precision": "FP16",
    "context_window": 8192,
    "batch_size": 4,
    "target_latency_p50_ms": 100,
    "target_latency_p99_ms": 500
  },
  "gpu": {
    "gpu_type": "A100-80GB",
    "target_gpu_utilization": 0.7,
    "current_replicas": 2
  },
  "service": {
    "service_id": "chat-api",
    "service_name": "Chat API Service",
    "team_id": "team_platform",
    "environment": "prod",
    "criticality": "high",
    "cloud_provider": "aws",
    "region": "us-east-1"
  },
  "options": {
    "include_cost": true,
    "output_file": "capacity_plan.json"
  }
}
```

#### Training Configuration (`training_config.json`)
```json
{
  "mode": "training",
  "training": {
    "dataset_size_tokens": 1000000000,
    "sequence_length": 4096,
    "num_epochs": 3,
    "global_batch_size": 64,
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "optimizer_type": "AdamW",
    "gradient_checkpointing": true,
    "training_precision": "BF16"
  },
  "model": {
    "model_size_billions": 70,
    "precision": "BF16"
  },
  "gpu": {
    "gpu_type": "H100-80GB",
    "target_gpu_utilization": 0.85
  },
  "service": {
    "service_id": "finetuning-job-001",
    "team_id": "team_ml",
    "environment": "prod",
    "cloud_provider": "gcp",
    "region": "us-central1"
  },
  "options": {
    "include_cost": true,
    "output_file": "training_plan.json"
  }
}
```

### Programmatic Interface

```python
from models import LLMWorkloadInput, TrainingInput, ModelConfig
from forecast_engine import LLMForecastEngine

# === Inference Mode ===
workload = LLMWorkloadInput(
    requests_per_second=10.0,
    avg_input_tokens=500,
    avg_output_tokens=200
)

model = ModelConfig(
    model_size_billions=70,
    precision="FP16",
    context_window=8192
)

engine = LLMForecastEngine()
inference_plan = engine.generate_inference_plan(workload, model)

print(inference_plan.gpu_resources.total_gpus)
print(inference_plan.cost.cost_per_million_tokens)

# === Fine-Tuning Mode ===
training = TrainingInput(
    dataset_size_tokens=1_000_000_000,
    sequence_length=4096,
    num_epochs=3,
    global_batch_size=64,
    optimizer_type="Adam",
    gradient_checkpointing=True
)

training_plan = engine.generate_training_plan(training, model)

print(training_plan.gpu_resources.required_gpus)
print(training_plan.training_metrics.estimated_duration_hours)
print(training_plan.cost.total_training_cost)
```
