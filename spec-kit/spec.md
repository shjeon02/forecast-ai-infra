# LLM Infrastructure Capacity Forecasting System - Specification

> Consolidated specification combining context, requirements, and architecture.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Problem & Goals](#2-problem--goals)
3. [Functional Requirements](#3-functional-requirements)
4. [Non-Functional Requirements](#4-non-functional-requirements)
5. [System Architecture](#5-system-architecture)
6. [Data Models](#6-data-models)
7. [Calculation Formulas](#7-calculation-formulas)
8. [User Interfaces](#8-user-interfaces)
9. [Glossary](#9-glossary)

---

## 1. Overview

### 1.1 Purpose

An intelligent forecasting system that generates accurate infrastructure capacity plans for LLM inference and fine-tuning workloads.

### 1.2 Target Users

| User Type | Primary Use Case |
|-----------|------------------|
| ML Engineers | Size GPU clusters for model serving |
| DevOps Engineers | Generate capacity specs for new deployments |
| Technical Architects | Validate architecture choices against projected load |
| Product Managers | Understand infrastructure implications of growth |
| Platform Engineers | Configure auto-scaling thresholds |

### 1.3 Supported Workloads

| Workload Type | Description | Key Metrics |
|---------------|-------------|-------------|
| **Inference** | Real-time model serving | RPS, TPS, TTFT, ITL |
| **Fine-Tuning** | Model training/adaptation | Duration, Cost per epoch |

---

## 2. Problem & Goals

### 2.1 Problem Statement

Organizations deploying LLM applications face challenges in accurately forecasting infrastructure capacity:
- Under-provisioning → poor latency, request failures
- Over-provisioning → unnecessary GPU costs

Current approaches (spreadsheets, trial-and-error, generic calculators) lack LLM-specific considerations.

### 2.2 Primary Goal

Generate accurate infrastructure capacity plans based on:
- LLM workload metrics (RPS, tokens per request, model parameters)
- Model deployment specs (replicas, GPU types, batch sizes)
- Service metadata (environment, criticality, cloud provider)

### 2.3 Success Metrics

| Metric | Target |
|--------|--------|
| Forecast Accuracy | ±20% of actual GPU requirements |
| User Completion Rate | >90% complete forecasting flow |
| Time to Generate Plan | <5 seconds |
| Export Success Rate | 100% valid JSON output |

---

## 3. Functional Requirements

### 3.1 Input Requirements

#### FR-1: LLM Workload Inputs (Inference)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1.1 | Accept requests per second (RPS) as float | P0 |
| FR-1.2 | Accept average input tokens per request | P0 |
| FR-1.3 | Accept average output tokens per request | P0 |
| FR-1.4 | Validate RPS > 0 | P0 |

#### FR-2: Model Configuration

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-2.1 | Accept model size in parameters (7B, 70B, 405B) | P0 |
| FR-2.2 | Accept precision (FP32, FP16, BF16, INT8, INT4, FP8) | P0 |
| FR-2.3 | Accept context window size (default: 4096) | P1 |
| FR-2.4 | Accept batch size (default: 1) | P1 |

#### FR-3: Fine-Tuning Inputs (Training)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-3.1 | Accept dataset size in tokens | P0 |
| FR-3.2 | Accept sequence length | P0 |
| FR-3.3 | Accept number of epochs | P0 |
| FR-3.4 | Accept global batch size | P0 |
| FR-3.5 | Accept optimizer type (Adam, AdamW, Adafactor, SGD) | P1 |
| FR-3.6 | Accept gradient checkpointing flag | P1 |

#### FR-4: GPU/Infrastructure Inputs

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-4.1 | Accept GPU type (A100-40GB, A100-80GB, H100-80GB, H100-SXM) | P1 |
| FR-4.2 | Accept target GPU utilization (default: 70%) | P2 |
| FR-4.3 | Accept cloud provider (aws, gcp, azure) | P1 |

### 3.2 Output Requirements

#### FR-5: Inference Capacity Plan

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-5.1 | Calculate required GPU memory | P0 |
| FR-5.2 | Calculate number of GPUs | P0 |
| FR-5.3 | Recommend GPU type | P0 |
| FR-5.4 | Calculate TPS per replica | P0 |
| FR-5.5 | Estimate latency (TTFT, ITL) | P1 |
| FR-5.6 | Provide scaling recommendations (min/max replicas) | P0 |
| FR-5.7 | Estimate monthly GPU cost | P1 |
| FR-5.8 | Calculate cost per million tokens | P1 |

#### FR-6: Training Capacity Plan

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-6.1 | Calculate GPU memory (weights + gradients + optimizer + activations) | P0 |
| FR-6.2 | Calculate required number of GPUs | P0 |
| FR-6.3 | Calculate training throughput (tokens/sec) | P0 |
| FR-6.4 | Estimate training duration (hours/days) | P0 |
| FR-6.5 | Recommend data parallel configuration | P1 |
| FR-6.6 | Calculate total training cost | P1 |

### 3.3 User Interface Requirements

#### FR-7: Input Modes

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-7.1 | Support **Interactive Mode** (conversational prompts) | P0 |
| FR-7.2 | Support **CLI Mode** (command-line arguments) | P0 |
| FR-7.3 | Support **JSON Config Mode** (`--config file.json`) | P1 |
| FR-7.4 | Allow CLI arguments to override JSON config values | P1 |
| FR-7.5 | Provide `--generate-config` to create templates | P2 |

#### FR-8: Export Capabilities

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-8.1 | Export capacity plan to JSON format | P0 |
| FR-8.2 | Allow custom output filename | P1 |

---

## 4. Non-Functional Requirements

### 4.1 Performance

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-1.1 | Calculation time | < 1 second |
| NFR-1.2 | Memory usage | < 50 MB |
| NFR-1.3 | Startup time | < 2 seconds |

### 4.2 Usability

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-2.1 | Time to complete interactive flow | < 2 minutes |
| NFR-2.2 | Error message clarity | Self-explanatory |
| NFR-2.3 | Output readability | Scannable in 30 seconds |

### 4.3 Reliability

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-3.1 | Graceful invalid input handling | 100% coverage |
| NFR-3.2 | No crashes on edge cases | Zero crash rate |
| NFR-3.3 | Deterministic output | 100% consistent |

### 4.4 Portability

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-4.1 | Python version | 3.10+ |
| NFR-4.2 | OS compatibility | macOS, Linux, Windows |
| NFR-4.3 | Core dependencies | Standard library only |

---

## 5. System Architecture

### 5.1 System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│               LLM Capacity Forecasting System                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   CLI Entry  │    │    Agent     │    │   Models     │       │
│  │   Point      │───▶│   Interface  │───▶│              │       │
│  │              │    │              │    │              │       │
│  │ forecast.py  │    │ llm_agent.py │    │  models.py   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         │                   ▼                   │                │
│         │          ┌──────────────┐             │                │
│         └─────────▶│  Forecasting │◀────────────┘                │
│                    │    Engines   │                              │
│                    │              │                              │
│                    │inference_    │                              │
│                    │engine.py     │                              │
│                    │training_     │                              │
│                    │engine.py     │                              │
│                    └──────────────┘                              │
│                           │                                      │
│                           ▼                                      │
│                    ┌──────────────┐                              │
│                    │    Config    │                              │
│                    │   config.py  │                              │
│                    └──────────────┘                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Presentation Layer                          │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐   │
│  │   Interactive Agent     │  │   Command-Line Interface    │   │
│  │   (llm_agent.py)        │  │   (commands/forecast.py)    │   │
│  └─────────────────────────┘  └─────────────────────────────┘   │
│                               ┌─────────────────────────────┐   │
│                               │   Config Loader             │   │
│                               │   (config_loader.py)        │   │
│                               └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Business Layer                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  Forecasting Engines                        │ │
│  │  ┌────────────────────┐  ┌────────────────────┐            │ │
│  │  │  Inference Engine  │  │  Training Engine   │            │ │
│  │  │  (inference_       │  │  (training_        │            │ │
│  │  │   engine.py)       │  │   engine.py)       │            │ │
│  │  └────────────────────┘  └────────────────────┘            │ │
│  │  ┌────────────────────┐  ┌────────────────────┐            │ │
│  │  │  GPU Calculator    │  │  Cost Estimator    │            │ │
│  │  └────────────────────┘  └────────────────────┘            │ │
│  └────────────────────────────────────────────────────────────┘ │
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

### 5.3 Module Responsibilities

| Module | Purpose |
|--------|---------|
| `main.py` | Interactive mode entry point |
| `commands/forecast.py` | CLI entry point with argument parsing |
| `llm_agent.py` | Conversational interface for LLM workloads |
| `config_loader.py` | JSON config loading and validation |
| `inference_engine.py` | Inference capacity calculations |
| `training_engine.py` | Training capacity calculations |
| `models.py` | Data structure definitions |
| `config.py` | Configuration constants and GPU specs |

### 5.4 Input Mode Selection Flow

```
                         ┌─────────────────┐
                         │   User Input    │
                         └─────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   Mode Selection Logic  │
                    │   (determine_input_mode)│
                    └─────────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
    ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
    │   Interactive    │ │   JSON Config    │ │   CLI Arguments  │
    │      Mode        │ │      Mode        │ │      Mode        │
    │                  │ │                  │ │                  │
    │ (no args given)  │ │ (--config file)  │ │ (--rps etc.)     │
    └──────────────────┘ └──────────────────┘ └──────────────────┘
              │                   │                   │
              └───────────────────┼───────────────────┘
                                  ▼
                    ┌─────────────────────────┐
                    │   Forecasting Engine    │
                    │   (Inference/Training)  │
                    └─────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   Capacity Plan Output  │
                    └─────────────────────────┘
```

---

## 6. Data Models

### 6.1 Input Models

#### LLMWorkloadInput (Inference)
```python
@dataclass
class LLMWorkloadInput:
    requests_per_second: float      # Required, > 0
    avg_input_tokens: int           # Required, > 0
    avg_output_tokens: int          # Required, > 0
    peak_load_multiplier: float     # Optional, default 1.5
    growth_rate: float              # Optional, 0-1000%
```

#### ModelConfig
```python
@dataclass
class ModelConfig:
    model_size_billions: float      # Required (7, 70, 405)
    precision: str                  # Required (FP32|FP16|BF16|INT8|INT4|FP8)
    context_window: int             # Optional, default 4096
    batch_size: int                 # Optional, default 1
```

#### TrainingInput (Fine-Tuning)
```python
@dataclass
class TrainingInput:
    dataset_size_tokens: int        # Required
    sequence_length: int            # Required
    num_epochs: int                 # Required
    global_batch_size: int          # Required
    micro_batch_size: int           # Optional, default 1
    gradient_accumulation_steps: int # Optional, default 1
    optimizer_type: str             # Optional (Adam|AdamW|Adafactor|SGD)
    gradient_checkpointing: bool    # Optional, default False
```

#### GPUConfig
```python
@dataclass
class GPUConfig:
    gpu_type: str                   # Optional (A100-40GB, H100-80GB, etc.)
    target_gpu_utilization: float   # Optional, default 0.7
```

### 6.2 Output Models

#### LLMCapacityPlan (Inference)
```python
@dataclass
class LLMCapacityPlan:
    gpu_resources:
        gpu_memory_per_replica_gb: float
        min_gpus_per_replica: int
        recommended_gpu_type: str
        min_replicas: int
        max_replicas: int
        total_gpus: int
    
    throughput:
        tps_per_replica: float
        total_tps_capacity: float
        max_rps_capacity: float
        estimated_ttft_ms: float
        estimated_itl_ms: float
        meets_latency_sla: bool
    
    scaling:
        recommended_min_replicas: int
        recommended_max_replicas: int
        auto_scale_rps_threshold: float
        auto_scale_gpu_util_threshold: float
    
    cost:
        monthly_gpu_cost: float
        cost_per_million_tokens: float
        monthly_total_cost: float
```

#### TrainingCapacityPlan (Fine-Tuning)
```python
@dataclass
class TrainingCapacityPlan:
    gpu_resources:
        gpu_memory_per_gpu_gb: float
        model_weights_gb: float
        gradients_gb: float
        optimizer_states_gb: float
        activations_gb: float
        required_gpus: int
        recommended_gpu_type: str
    
    training_metrics:
        tokens_per_second_per_gpu: float
        total_tokens_per_second: float
        steps_per_epoch: int
        total_steps: int
        estimated_duration_hours: float
        estimated_duration_days: float
    
    cost:
        gpu_cost_per_hour: float
        total_training_cost: float
        cost_per_epoch: float
```

---

## 7. Calculation Formulas

### 7.1 Inference Calculations

#### GPU Memory
```
GPU Memory = Model Weights + KV Cache + Activations + Overhead

Model Weights = model_params × bytes_per_param
  - FP32: 4 bytes
  - FP16/BF16: 2 bytes
  - INT8: 1 byte
  - INT4: 0.5 bytes

KV Cache = 2 × num_layers × hidden_dim × context_length × batch_size × bytes_per_param

Activations ≈ 10-20% of model weights

Overhead ≈ 10% buffer
```

#### Throughput
```
TPS per Replica = (GPU Memory Bandwidth × Efficiency) / (Model Size × Bytes per Param)

Effective RPS = TPS / avg_output_tokens

Required Replicas = ceil(target_RPS / RPS_per_replica) × safety_margin
```

#### Latency
```
TTFT = (Input Tokens × Time per Token for Prefill) + Model Loading Overhead

ITL = Model Size / (GPU Memory Bandwidth × Efficiency)

E2E Latency = TTFT + (Output Tokens × ITL)
```

### 7.2 Fine-Tuning Calculations

#### GPU Memory
```
GPU Memory = Model Weights + Gradients + Optimizer States + Activations + Overhead

Model Weights = model_params × bytes_per_param

Gradients = Model Weights (same size)

Optimizer States:
  - Adam/AdamW: 2 × Model Weights (momentum + variance)
  - Adafactor: ~0.5 × Model Weights (factorized)
  - SGD: 0 (no state)

Activations = sequence_length × batch_size × hidden_dim × num_layers × bytes_per_param
  - Gradient checkpointing: ~33% memory reduction, ~20% slower
```

#### Training Throughput
```
Tokens per Second per GPU = (GPU Compute FLOPs × Efficiency) / (Model FLOPs per Token)

Model FLOPs per Token ≈ 2 × model_params × sequence_length

Training Speed = tokens_per_sec_per_gpu × num_gpus × gradient_accumulation_steps

Training Duration = (dataset_size × num_epochs) / training_speed
```

#### GPU Count
```
Required GPUs = ceil(
  (Model Memory + Optimizer Memory) / (Available GPU Memory per GPU)
) × data_parallel_factor

data_parallel_factor = ceil(global_batch_size / (micro_batch_size × gradient_accum_steps))
```

### 7.3 Inference vs Fine-Tuning Comparison

| Aspect | Inference | Fine-Tuning |
|--------|-----------|-------------|
| **Primary Bottleneck** | KV Cache Memory | Optimizer States |
| **Memory Components** | Weights + KV Cache + Activations | Weights + Gradients + Optimizer + Activations |
| **Scaling Strategy** | Horizontal (Replicas) | Data Parallelism (More GPUs) |
| **Throughput Metric** | TPS, RPS | Tokens/sec per GPU |
| **Time Metric** | Latency (TTFT, ITL) | Training duration |
| **Cost Model** | Per token, per request | Per training run, per epoch |

---

## 8. User Interfaces

### 8.1 CLI Usage

```bash
# Interactive mode (default)
python main.py

# JSON configuration mode
python commands/forecast.py --config inference_config.json
python commands/forecast.py --config training_config.json

# JSON with CLI overrides
python commands/forecast.py --config config.json --rps 20.0 --cost

# Generate template config
python commands/forecast.py --generate-config inference
python commands/forecast.py --generate-config training

# Inference mode (CLI args)
python commands/forecast.py \
  --mode inference \
  --rps 10.0 \
  --input-tokens 500 \
  --output-tokens 200 \
  --model-size 70 \
  --precision FP16 \
  --gpu-type H100-80GB \
  --cost

# Fine-tuning mode (CLI args)
python commands/forecast.py \
  --mode training \
  --dataset-size 1000000000 \
  --sequence-length 4096 \
  --epochs 3 \
  --global-batch-size 64 \
  --model-size 70 \
  --precision BF16 \
  --optimizer AdamW \
  --gradient-checkpointing \
  --cost
```

### 8.2 JSON Configuration Schema

#### Inference Config
```json
{
  "mode": "inference",
  "workload": {
    "requests_per_second": 10.0,
    "avg_input_tokens": 500,
    "avg_output_tokens": 200,
    "peak_load_multiplier": 1.5
  },
  "model": {
    "model_size_billions": 70,
    "precision": "FP16",
    "context_window": 8192,
    "batch_size": 4
  },
  "gpu": {
    "gpu_type": "H100-80GB",
    "target_gpu_utilization": 0.7
  },
  "options": {
    "include_cost": true,
    "output_file": "capacity_plan.json"
  }
}
```

#### Training Config
```json
{
  "mode": "training",
  "training": {
    "dataset_size_tokens": 1000000000,
    "sequence_length": 4096,
    "num_epochs": 3,
    "global_batch_size": 64,
    "optimizer_type": "AdamW",
    "gradient_checkpointing": true
  },
  "model": {
    "model_size_billions": 70,
    "precision": "BF16"
  },
  "gpu": {
    "gpu_type": "H100-80GB",
    "target_gpu_utilization": 0.85
  },
  "options": {
    "include_cost": true,
    "output_file": "training_plan.json"
  }
}
```

### 8.3 Programmatic Interface

```python
from models import LLMWorkloadInput, TrainingInput, ModelConfig
from inference_engine import InferenceEngine
from training_engine import TrainingEngine

# Inference
workload = LLMWorkloadInput(
    requests_per_second=10.0,
    avg_input_tokens=500,
    avg_output_tokens=200
)
model = ModelConfig(model_size_billions=70, precision="FP16")
engine = InferenceEngine()
plan = engine.generate_plan(workload, model)

# Training
training = TrainingInput(
    dataset_size_tokens=1_000_000_000,
    sequence_length=4096,
    num_epochs=3,
    global_batch_size=64
)
engine = TrainingEngine()
plan = engine.generate_plan(training, model)
```

---

## 9. Glossary

| Term | Definition |
|------|------------|
| **RPS** | Requests Per Second - primary throughput metric |
| **TPS** | Tokens Per Second - token generation throughput |
| **TTFT** | Time To First Token - latency from request to first token |
| **ITL** | Inter-Token Latency - time between consecutive tokens |
| **KV Cache** | Key-Value cache storing attention states |
| **Context Window** | Maximum tokens a model can process |
| **Model Replica** | Single instance of a model serving requests |
| **Quantization** | Reducing precision (FP16→INT8→INT4) |
| **Gradient Checkpointing** | Recomputing activations to save memory |
| **Optimizer States** | Additional memory for Adam (2x weights) |
| **Data Parallelism** | Distributing batches across GPUs |
| **Gradient Accumulation** | Simulating larger batches |
| **Peak Load Multiplier** | Factor for peak usage periods |

---

## File Structure

```
resource_forecast/
├── spec-kit/                    # Specification documents
│   ├── spec.md                 # This consolidated specification
│   ├── context.md              # Problem context and goals
│   ├── requirements.md         # Detailed requirements
│   ├── architecture.md         # Architecture details
│   ├── work_plan.md            # Implementation plan
│   └── acceptance_criteria.md  # Testing criteria
│
├── commands/
│   └── forecast.py             # CLI entry point
│
├── models.py                    # Data models
├── config.py                    # Configuration & GPU specs
├── config_loader.py             # JSON config loader
├── inference_engine.py          # Inference calculations
├── training_engine.py           # Training calculations
├── llm_agent.py                 # Interactive agent
├── main.py                      # Interactive entry point
│
├── requirements.txt             # Dependencies
└── README.md                    # User documentation
```

---

*Last Updated: January 2026*
