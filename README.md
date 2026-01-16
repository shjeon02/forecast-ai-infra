# LLM Infrastructure Capacity Forecasting

GPU 및 인프라 용량을 예측하는 도구입니다. LLM 추론(Inference) 및 파인튜닝(Training) 워크로드에 대한 리소스 요구사항을 계산합니다.

## 주요 기능

- 🧠 **LLM 추론 용량 예측**: GPU 메모리, TPS, 레이턴시, 레플리카 수 계산
- 🎓 **파인튜닝 용량 예측**: 학습 시간, GPU 수, 메모리 분석, 비용 추정
- 📈 **시계열 예측**: 과거 데이터 기반 미래 리소스 수요 예측 (STL, ARIMA, ETS)
- 💰 **비용 추정**: AWS, GCP, Azure GPU 가격 기반 비용 계산
- 📋 **서비스 관리**: 서비스 메타데이터, 용량 요청 워크플로우

## 설치

```bash
# 의존성 설치
pip install -r requirements.txt
```

## 사용법

### 1. Interactive 모드 (대화형)

```bash
python main.py
# 또는
python commands/forecast.py
```

단계별로 질문에 답변하며 용량 계획을 생성합니다:
- 워크로드 타입 선택 (inference/training)
- 모델 설정 (크기, 정밀도)
- 워크로드 파라미터 입력
- GPU 설정 (선택)

### 2. JSON Config 모드

```bash
# 템플릿 생성
python commands/forecast.py --generate-config inference
python commands/forecast.py --generate-config training

# 설정 파일로 실행
python commands/forecast.py --config inference_config.json

# CLI 옵션으로 오버라이드
python commands/forecast.py --config inference_config.json --rps 20.0
```

#### Inference 설정 예시 (`inference_config.json`)

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
    "gpu_type": "A100-80GB",
    "target_gpu_utilization": 0.7
  },
  "options": {
    "include_cost": true
  }
}
```

#### Training 설정 예시 (`training_config.json`)

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
    "gpu_type": "H100-80GB"
  }
}
```

### 3. CLI Args 모드

#### Inference 모드

```bash
python commands/forecast.py \
  --mode inference \
  --rps 10 \
  --input-tokens 500 \
  --output-tokens 200 \
  --model-size 70 \
  --precision FP16 \
  --gpu-type A100-80GB \
  --cost \
  --output inference_plan.json
```

#### Training 모드

```bash
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
  --cost \
  --output training_plan.json
```

## 출력 예시

### Inference Capacity Plan

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 LLM INFERENCE CAPACITY PLAN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 INPUT SUMMARY:
   Requests per Second: 10.0
   Input Tokens: 500
   Output Tokens: 200
   Model Size: 70B
   Precision: FP16

🖥️ GPU RESOURCES:
   GPU Memory per Replica: 175.94 GB
   GPUs per Replica: 4
   Recommended GPU: H100-80GB
   Replicas: 59 - 89
   Total GPUs: 236

⚡ THROUGHPUT:
   TPS per Replica: 33.94
   Total TPS Capacity: 2002.17
   Max RPS Capacity: 10.01

⏱️ LATENCY:
   Est. TTFT: 50.64 ms
   Est. ITL: 63.95 ms

💰 COST ESTIMATES:
   Monthly GPU Cost: $509,760.00
   Cost per 1M Tokens: $98.2268
```

### Training Capacity Plan

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 LLM TRAINING CAPACITY PLAN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 INPUT SUMMARY:
   Dataset Size: 1,000,000,000 tokens
   Sequence Length: 4096
   Epochs: 3
   Global Batch Size: 64
   Model Size: 70B
   Optimizer: Adam

🖥️ GPU MEMORY BREAKDOWN:
   Model Weights: 130.39 GB
   Gradients: 130.39 GB
   Optimizer States: 260.77 GB
   Activations: 36.51 GB
   Total per GPU: 18.10 GB

🔧 GPU REQUIREMENTS:
   Required GPUs: 64
   Recommended GPU: H100-80GB
   Nodes: 8 × 8 GPUs

⏱️ TRAINING ESTIMATES:
   Duration: 11.0 hours (0.5 days)

💰 COST ESTIMATES:
   Total Training Cost: $2,121.21
   Cost per Epoch: $707.07
```

## Programmatic Usage

```python
from llm_forecast_engine import LLMForecastEngine
from models import LLMWorkloadInput, TrainingInput, ModelConfig, GPUConfig

engine = LLMForecastEngine()

# === Inference Mode ===
workload = LLMWorkloadInput(
    requests_per_second=10.0,
    avg_input_tokens=500,
    avg_output_tokens=200
)
model = ModelConfig(
    model_size_billions=70,
    precision="FP16"
)

inference_plan = engine.generate_inference_plan(workload, model, include_cost=True)
print(f"Total GPUs: {inference_plan.gpu_resources.total_gpus}")
print(f"Cost/M tokens: ${inference_plan.cost.cost_per_million_tokens:.4f}")

# === Training Mode ===
training = TrainingInput(
    dataset_size_tokens=1_000_000_000,
    sequence_length=4096,
    num_epochs=3,
    global_batch_size=64
)

training_plan = engine.generate_training_plan(training, model, include_cost=True)
print(f"Required GPUs: {training_plan.gpu_resources.required_gpus}")
print(f"Duration: {training_plan.training_metrics.estimated_duration_hours:.1f} hours")
print(f"Cost: ${training_plan.cost.total_training_cost:,.2f}")

# === GPU Comparison ===
comparison = engine.compare_gpu_options(workload, model)
for c in comparison:
    print(f"{c['gpu_type']}: {c['total_gpus']} GPUs, ${c['monthly_gpu_cost']:,.0f}/month")

# === Time-Series Forecasting ===
from datetime import datetime, timedelta

rps_history = [
    (datetime.now() - timedelta(days=30-i), 10 + i * 0.1)
    for i in range(30)
]
forecast = engine.forecast_future_needs(
    rps_history, model, horizon_days=30, scenario="pessimistic"
)
print(f"Scaling recommendation: {forecast['scaling_recommendations'][0]}")
```

## 프로젝트 구조

```
resource_forecast/
├── commands/
│   └── forecast.py          # CLI 엔트리포인트
├── models.py                 # 데이터 모델 (Input/Output)
├── config.py                 # GPU 스펙, 가격, 설정값
├── config_loader.py          # JSON 설정 로더
├── inference_engine.py       # 추론 용량 예측 엔진
├── training_engine.py        # 학습 용량 예측 엔진
├── forecasting.py            # 시계열 예측 엔진
├── services.py               # 서비스 관리
├── llm_forecast_engine.py    # 통합 엔진
├── llm_agent.py              # 인터랙티브 에이전트
├── main.py                   # 메인 엔트리포인트
└── spec-kit/                 # 스펙 문서
    ├── requirements.md
    ├── architecture.md
    ├── acceptance_criteria.md
    └── work_plan.md
```

## 지원 GPU

| GPU | 메모리 | 가격/시간 (AWS) |
|-----|--------|-----------------|
| A100-40GB | 40 GB | $1.10 |
| A100-80GB | 80 GB | $1.60 |
| H100-80GB | 80 GB | $3.00 |
| L4 | 24 GB | $0.50 |
| T4 | 16 GB | $0.35 |
| V100 | 32 GB | $0.90 |

## 지원 정밀도

- FP32: 4 bytes/param
- FP16: 2 bytes/param
- BF16: 2 bytes/param
- INT8: 1 byte/param
- INT4: 0.5 bytes/param

## 계산 공식

### GPU 메모리 (Inference)

```
GPU Memory = Model Weights + KV Cache + Activations + Overhead

Model Weights = model_params × bytes_per_param
KV Cache = 2 × num_layers × hidden_dim × context_length × batch_size × bytes_per_param
Activations ≈ 15% of model weights
Overhead ≈ 10%
```

### GPU 메모리 (Training)

```
GPU Memory = Model Weights + Gradients + Optimizer States + Activations + Overhead

Gradients = Model Weights
Optimizer States:
  - Adam: 2 × Model Weights
  - Adafactor: 0.5 × Model Weights
  - SGD: 0
Activations = sequence_length × batch_size × hidden_dim × num_layers × bytes
```

### 학습 시간

```
Training Duration = (Dataset Size × Epochs) / (Tokens per Second per GPU × Number of GPUs)
```

## 라이선스

MIT License
