# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Infrastructure Capacity                       │
│                    Forecasting System                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   CLI Entry  │    │    Agent     │    │   Models     │      │
│  │   Point      │───▶│   Interface  │───▶│              │      │
│  │              │    │              │    │              │      │
│  │ forecast.py  │    │  agent.py    │    │  models.py   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         │                   ▼                   │               │
│         │          ┌──────────────┐             │               │
│         └─────────▶│  Forecasting │◀────────────┘               │
│                    │    Engine    │                             │
│                    │              │                             │
│                    │forecast_     │                             │
│                    │engine.py     │                             │
│                    └──────────────┘                             │
│                           │                                      │
│                           ▼                                      │
│                    ┌──────────────┐                             │
│                    │    Config    │                             │
│                    │              │                             │
│                    │  config.py   │                             │
│                    └──────────────┘                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### Layer Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Presentation Layer                          │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐   │
│  │   Interactive Agent     │  │   Command-Line Interface    │   │
│  │   (agent.py)            │  │   (commands/forecast.py)    │   │
│  └─────────────────────────┘  └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Business Layer                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  Forecast Engine                          │   │
│  │                  (forecast_engine.py)                     │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐            │   │
│  │  │  Compute   │ │  Storage   │ │  Network   │            │   │
│  │  │ Calculator │ │ Calculator │ │ Calculator │            │   │
│  │  └────────────┘ └────────────┘ └────────────┘            │   │
│  │  ┌────────────┐ ┌────────────┐                           │   │
│  │  │  Scaling   │ │    Cost    │                           │   │
│  │  │  Advisor   │ │ Estimator  │                           │   │
│  │  └────────────┘ └────────────┘                           │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Data Layer                                │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐   │
│  │   Data Models           │  │   Configuration             │   │
│  │   (models.py)           │  │   (config.py)               │   │
│  └─────────────────────────┘  └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Module Descriptions

### 1. Entry Points

#### `main.py`
- **Purpose**: Primary entry point for interactive mode
- **Responsibilities**:
  - Initialize agent
  - Start interactive session
- **Dependencies**: `agent.py`

#### `commands/forecast.py`
- **Purpose**: CLI entry point with argument parsing
- **Responsibilities**:
  - Parse command-line arguments
  - Support both interactive and non-interactive modes
  - Handle output file generation
- **Dependencies**: `agent.py`, `models.py`, `forecast_engine.py`

### 2. Presentation Layer

#### `agent.py` - CapacityForecastAgent
- **Purpose**: Conversational interface for user interaction
- **Responsibilities**:
  - Display prompts and messages
  - Collect and validate user input
  - Format output for display
  - Handle export operations
- **Key Methods**:
  - `welcome_message()`: Display greeting
  - `collect_input()`: Interactive input collection
  - `validate_*()`: Input validation methods
  - `format_capacity_plan()`: Output formatting
  - `export_plan()`: JSON export
  - `run()`: Main interaction loop

### 3. Business Layer

#### `forecast_engine.py` - ForecastEngine
- **Purpose**: Core calculation logic
- **Responsibilities**:
  - Calculate compute requirements
  - Calculate storage requirements
  - Calculate network requirements
  - Generate scaling recommendations
  - Estimate costs
- **Key Methods**:
  - `calculate_compute_resources()`: CPU/memory calculations
  - `calculate_storage_resources()`: Storage calculations
  - `calculate_network_resources()`: Bandwidth/LB calculations
  - `calculate_scaling_recommendations()`: Scaling advice
  - `calculate_cost_estimate()`: Cost projections
  - `generate_capacity_plan()`: Orchestrate full plan generation

### 4. Data Layer

#### `models.py`
- **Purpose**: Data structure definitions
- **Contains**:
  - `UserInput`: Input parameters dataclass
  - `ComputeResources`: Compute specs dataclass
  - `StorageResources`: Storage specs dataclass
  - `NetworkResources`: Network specs dataclass
  - `ScalingRecommendations`: Scaling advice dataclass
  - `CostEstimate`: Cost projection dataclass
  - `CapacityPlan`: Complete plan container

#### `config.py`
- **Purpose**: Configuration constants
- **Contains**:
  - Per-user resource coefficients
  - Base resource requirements
  - Scaling factors
  - Instance type definitions
  - Cost parameters

---

## Data Flow

### Interactive Mode Flow

```
┌──────┐    ┌───────┐    ┌────────────┐    ┌────────────┐    ┌────────┐
│ User │───▶│ Agent │───▶│ Validation │───▶│ UserInput  │───▶│ Engine │
└──────┘    └───────┘    └────────────┘    └────────────┘    └────────┘
                                                                  │
   ┌──────────────────────────────────────────────────────────────┘
   │
   ▼
┌────────────┐    ┌────────────┐    ┌───────┐    ┌──────┐
│CapacityPlan│───▶│ Formatting │───▶│ Agent │───▶│ User │
└────────────┘    └────────────┘    └───────┘    └──────┘
                         │
                         ▼
                  ┌────────────┐
                  │ JSON Export│
                  └────────────┘
```

### CLI Mode Flow

```
┌─────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│ CLI Args│───▶│ ArgParser  │───▶│ UserInput  │───▶│   Engine   │
└─────────┘    └────────────┘    └────────────┘    └────────────┘
                                                         │
   ┌─────────────────────────────────────────────────────┘
   │
   ▼
┌────────────┐    ┌────────────┐    ┌────────────┐
│CapacityPlan│───▶│ Formatting │───▶│   stdout   │
└────────────┘    └────────────┘    └────────────┘
                         │
                         ▼
                  ┌────────────┐
                  │ JSON File  │
                  └────────────┘
```

---

## Calculation Pipeline

The system supports two distinct forecasting modes for different LLM workloads:

1. **Inference (Serving)**: Real-time request serving with latency requirements
2. **Fine-Tuning (Training)**: Model training with dataset and epoch requirements

These modes have fundamentally different resource requirements, bottlenecks, and optimization goals.

---

### Mode 1: Inference Capacity Forecasting (Serving)

```
         LLMWorkloadInput              ModelConfig
                │                           │
                ▼                           ▼
        ┌───────────────────────────────────────────┐
        │           Extract Parameters              │
        │  - requests_per_second (RPS)              │
        │  - avg_input_tokens                       │
        │  - avg_output_tokens                      │
        │  - model_size_billions                    │
        │  - precision (FP16/INT8/INT4)            │
        │  - context_window                         │
        │  - batch_size                             │
        └───────────────────────────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │   GPU    │  │Throughput│  │  Network │
    │  Memory  │  │   TPS    │  │   Calc   │
    │   Calc   │  │   Calc   │  │          │
    └──────────┘  └──────────┘  └──────────┘
          │             │             │
          ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │  Model   │  │ Latency  │  │Bandwidth │
    │ Weight + │  │Estimation│  │   Calc   │
    │ KV Cache │  │TTFT, ITL │  │          │
    └──────────┘  └──────────┘  └──────────┘
          │             │             │
          ▼             ▼             │
    ┌──────────┐  ┌──────────┐       │
    │  Replica │  │  SLA     │       │
    │   Count  │  │  Check   │       │
    └──────────┘  └──────────┘       │
          │             │             │
          └─────────────┼─────────────┘
                        ▼
              ┌─────────────────┐
              │ Scaling Advisor │
              │ - min/max replicas │
              │ - auto-scale thresholds │
              └─────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │ Cost Estimator  │
              │ - GPU cost/month │
              │ - cost per 1M tokens │
              └─────────────────┘
                        │
                        ▼
                 LLMCapacityPlan
```

### GPU Memory Calculation Formula

```
GPU Memory = Model Weights + KV Cache + Activations + Overhead

Where:
  Model Weights = model_params × bytes_per_param
    - FP32: 4 bytes
    - FP16/BF16: 2 bytes
    - INT8: 1 byte
    - INT4: 0.5 bytes

  KV Cache = 2 × num_layers × hidden_dim × context_length × batch_size × bytes_per_param

  Activations ≈ 10-20% of model weights (during inference)
  
  Overhead ≈ 10% buffer
```

### Throughput Calculation Formula

```
TPS per Replica = (GPU Memory Bandwidth × Efficiency) / (Model Size × Bytes per Param)

Effective RPS = TPS / avg_output_tokens

Required Replicas = ceil(target_RPS / RPS_per_replica) × safety_margin
```

### Latency Estimation

```
TTFT (Time to First Token) = 
  (Input Tokens × Time per Token for Prefill) + Model Loading Overhead

ITL (Inter-Token Latency) = 
  Model Size / (GPU Memory Bandwidth × Efficiency)

E2E Latency = TTFT + (Output Tokens × ITL)
```

### Mode 2: Fine-Tuning Capacity Forecasting (Training)

```
         TrainingInput              ModelConfig
                │                           │
                ▼                           ▼
        ┌───────────────────────────────────────────┐
        │           Extract Parameters              │
        │  - dataset_size (total tokens)           │
        │  - sequence_length                        │
        │  - global_batch_size                      │
        │  - micro_batch_size                       │
        │  - num_epochs                             │
        │  - model_size_billions                    │
        │  - precision (FP16/BF16/FP8)            │
        │  - optimizer_type (Adam, Adafactor)     │
        │  - gradient_accumulation_steps            │
        └───────────────────────────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │   GPU    │  │Throughput│  │  Network │
    │  Memory  │  │   Calc   │  │   Calc   │
    │   Calc   │  │          │  │          │
    └──────────┘  └──────────┘  └──────────┘
          │             │             │
          ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │  Model   │  │ Training │  │Bandwidth │
    │ Weight + │  │  Speed   │  │   Calc   │
    │ Gradients│  │tokens/sec│  │          │
    │ + Optim  │  │per GPU   │  │          │
    │  States  │  │          │  │          │
    └──────────┘  └──────────┘  └──────────┘
          │             │             │
          ▼             ▼             │
    ┌──────────┐  ┌──────────┐       │
    │  GPU     │  │ Training │       │
    │  Count   │  │ Duration │       │
    │ Required │  │Estimation│       │
    └──────────┘  └──────────┘       │
          │             │             │
          └─────────────┼─────────────┘
                        ▼
              ┌─────────────────┐
              │ Data Parallel   │
              │ Configuration   │
              │ - GPUs per node │
              │ - num_nodes     │
              │ - total GPUs    │
              └─────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │ Cost Estimator  │
              │ - GPU cost/training │
              │ - cost per epoch │
              │ - total training cost │
              └─────────────────┘
                        │
                        ▼
              TrainingCapacityPlan
```

### Fine-Tuning GPU Memory Calculation Formula

```
GPU Memory = Model Weights + Gradients + Optimizer States + Activations + Overhead

Where:
  Model Weights = model_params × bytes_per_param
    - FP32: 4 bytes
    - FP16/BF16: 2 bytes
    - FP8: 1 byte

  Gradients = Model Weights (same size as weights)
  
  Optimizer States:
    - Adam: 2 × Model Weights (momentum + variance)
    - AdamW: 2 × Model Weights
    - Adafactor: ~0.5 × Model Weights (factorized)
    - SGD: 0 (no state)
  
  Activations = sequence_length × batch_size × hidden_dim × num_layers × bytes_per_param
    - Can be reduced with gradient checkpointing (recompute activations)
    - Gradient checkpointing: ~33% memory reduction, ~20% slower
  
  Overhead ≈ 10% buffer
```

### Fine-Tuning Throughput Calculation

```
Tokens per Second per GPU = 
  (GPU Compute FLOPs × Efficiency) / (Model FLOPs per Token)

Where:
  Model FLOPs per Token ≈ 2 × model_params × sequence_length

Training Speed = tokens_per_sec_per_gpu × num_gpus × gradient_accumulation_steps

Training Duration = (dataset_size × num_epochs) / training_speed
```

### Fine-Tuning GPU Count Calculation

```
Required GPUs = ceil(
  (Model Memory + Optimizer Memory) / (Available GPU Memory per GPU)
) × data_parallel_factor

Where:
  data_parallel_factor = ceil(global_batch_size / (micro_batch_size × gradient_accum_steps))
  
  Minimum GPUs for data parallelism = data_parallel_factor
```

### Key Differences: Inference vs Fine-Tuning

| Aspect | Inference (Serving) | Fine-Tuning (Training) |
|--------|---------------------|------------------------|
| **Primary Bottleneck** | GPU Memory (KV Cache) | GPU Memory (Optimizer States) |
| **Memory Components** | Weights + KV Cache + Activations | Weights + Gradients + Optimizer + Activations |
| **Scaling Strategy** | Horizontal (Replicas) | Data Parallelism (More GPUs) |
| **Throughput Metric** | TPS, RPS | Tokens/sec per GPU, Training speed |
| **Time Metric** | Latency (TTFT, ITL) | Training duration (hours/days) |
| **Cost Model** | Per token, per request | Per training run, per epoch |
| **Batch Size Impact** | Throughput vs Latency tradeoff | Memory vs Speed tradeoff |
| **Optimization Focus** | Low latency, high throughput | Fast training, memory efficiency |

---

### Mode 3: Time-Series Forecasting (Advanced)

```
                Historical Data
        ┌───────────────────────────────┐
        │  - RPS metrics (time-series)  │
        │  - TPS metrics (time-series)  │
        │  - Latency (TTFT, ITL)        │
        │  - GPU utilization            │
        │  - Cost history               │
        │  - Training metrics (optional) │
        └───────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │     STL Decomposition         │
        │  - Trend extraction           │
        │  - Seasonal pattern (daily/weekly) │
        │  - Residual analysis          │
        └───────────────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │  ARIMA   │  │   ETS    │  │ Ensemble │
    │  Model   │  │  Model   │  │ Weights  │
    └──────────┘  └──────────┘  └──────────┘
          │             │             │
          └─────────────┼─────────────┘
                        ▼
        ┌───────────────────────────────┐
        │     Ensemble Forecast         │
        │  - Weighted combination       │
        │  - Confidence intervals (P80, P95) │
        └───────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │     Scenario Adjustment       │
        │  - optimistic: 0.85x          │
        │  - pessimistic: 1.15x         │
        │  - spike: 2.0-3.0x            │
        └───────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │  Capacity Recommendation      │
        │  - Future replica needs       │
        │  - GPU scaling timeline       │
        │  - Cost projection            │
        └───────────────────────────────┘
                        │
                        ▼
                  ForecastOutput
        ┌───────────────────────────────┐
        │  - forecast: List[float]      │
        │  - intervals: {p80, p95}      │
        │  - components: {trend,        │
        │                 seasonal,     │
        │                 residual}     │
        │  - explanations: List[str]    │
        │  - replica_recommendations    │
        │  - gpu_scaling_recommendations│
        └───────────────────────────────┘
```

---

## Mode Selection Strategy

The system determines which forecasting mode to use based on input parameters:

### Inference Mode Selection
- **Trigger**: `workload_type="inference"` OR presence of `requests_per_second`
- **Use Case**: Real-time serving, API endpoints, chatbots
- **Output**: Replicas, TPS, latency, cost per token

### Fine-Tuning Mode Selection
- **Trigger**: `workload_type="training"` OR presence of `dataset_size` and `num_epochs`
- **Use Case**: Model training, fine-tuning, continued pre-training
- **Output**: GPU count, training duration, cost per training run

### Combined Forecasting
- **Use Case**: Organizations running both inference and training
- **Approach**: Run both modes separately and combine results
- **Output**: Separate capacity plans for each workload type

---

## File Structure

```
resource_forecast/
├── spec-kit/                    # Specification documents
│   ├── context.md              # Problem context and goals
│   ├── requirements.md         # Functional/non-functional requirements
│   ├── architecture.md         # This document
│   ├── work_plan.md            # Implementation plan
│   └── acceptance_criteria.md  # Testing criteria
│
├── commands/                    # CLI commands
│   └── forecast.py             # Main CLI entry point
│
├── models.py                    # Data models
├── config.py                    # Configuration
├── forecast_engine.py           # Core logic
├── agent.py                     # Chatbot interface
├── main.py                      # Interactive entry point
│
├── requirements.txt             # Dependencies
└── README.md                    # User documentation
```

---

## Design Decisions

### D1: Dataclasses for Models
**Decision**: Use Python dataclasses for all data models
**Rationale**: 
- Type safety with minimal boilerplate
- Built-in `__init__`, `__repr__`, `__eq__`
- No external dependencies
- Good IDE support

### D2: Configuration as Module
**Decision**: Store configuration as Python constants in `config.py`
**Rationale**:
- Simple to modify
- Type-checked at import time
- No file parsing needed
- Easy to document with comments

### D3: Engine Pattern
**Decision**: Separate calculation logic into `ForecastEngine` class
**Rationale**:
- Single responsibility principle
- Testable in isolation
- Swappable for different forecasting strategies
- Clear interface boundary

### D4: Agent Pattern for UI
**Decision**: Use agent/chatbot pattern for user interaction
**Rationale**:
- Conversational feel matches use case
- Flexible input collection
- Easy to extend with NLP capabilities
- Familiar pattern for users

### D5: No External Dependencies (Core)
**Decision**: Core functionality uses only standard library
**Rationale**:
- Easy installation
- No dependency conflicts
- Portable across environments
- Optional dependencies for enhanced features

---

## Extension Points

### Adding New Resource Types
1. Create new dataclass in `models.py`
2. Add calculation method in `forecast_engine.py`
3. Include in `CapacityPlan` dataclass
4. Update formatting in `agent.py`

### Adding New Input Parameters
1. Add field to `UserInput` dataclass
2. Add validation method in `agent.py`
3. Add prompt in collection flow
4. Add CLI argument in `commands/forecast.py`
5. Use in calculation methods

### Adding New Output Formats
1. Add format method in `agent.py` or new formatter module
2. Add CLI flag for format selection
3. Update export logic

### Adding Cloud Provider Specifics
1. Create provider-specific config module
2. Add provider parameter to `UserInput`
3. Adjust instance type recommendations
4. Update cost calculations
