# LLM Infrastructure Capacity Forecasting System

An intelligent forecasting system that generates accurate infrastructure capacity plans for LLM/Agent applications based on workload metrics (requests per second, tokens, model parameters) rather than user counts.

## Features

- 🤖 **Interactive Chatbot Interface**: Conversational agent that guides you through the forecasting process
- 🎯 **LLM-Specific Planning**: Calculates GPU resources, model replicas, and throughput (TPS/RPS)
- 📊 **Comprehensive Capacity Planning**: GPU memory, latency estimation (TTFT/ITL), and scaling recommendations
- 💰 **Token-Based Cost Estimates**: Cost per million tokens and monthly GPU costs
- 📈 **Scaling Recommendations**: Horizontal scaling (replicas) and auto-scaling thresholds
- 📄 **Export Capabilities**: Export capacity plans to JSON format

## Quick Start

### Installation

1. Clone or navigate to the project directory:
```bash
cd resource_forecast
```

2. (Optional) Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies (currently uses only standard library):
```bash
pip install -r requirements.txt
```

### Usage

#### Interactive Mode (Recommended)

Run the main application:
```bash
python main.py
```

Or use the command interface:
```bash
python commands/forecast.py
```

The agent will prompt you for:
- Requests per second (RPS)
- Average input tokens per request
- Average output tokens per request
- Model size (e.g., 7B, 70B, 405B)
- Model precision (FP16, INT8, INT4)
- Optional: Context window, batch size, GPU type

#### Non-Interactive Mode

Provide inputs directly via command line:
```bash
python commands/forecast.py \
  --rps 10.0 \
  --input-tokens 500 \
  --output-tokens 200 \
  --model-size 70 \
  --precision FP16 \
  --context-window 8192 \
  --batch-size 4 \
  --gpu-type A100-80GB \
  --cost \
  --output my_plan.json
```

## Project Structure

```
resource_forecast/
├── spec-kit/                    # Specification documents
│   ├── context.md              # Problem context, goals, assumptions
│   ├── requirements.md         # Functional & non-functional requirements
│   ├── architecture.md         # System design, components, data flow
│   ├── work_plan.md            # Implementation phases and tasks
│   └── acceptance_criteria.md  # Test scenarios and sign-off criteria
│
├── commands/
│   └── forecast.py             # Command-line entry point
│
├── models.py                    # Data models and structures
├── config.py                    # Configuration parameters
├── forecast_engine.py           # Core forecasting logic
├── agent.py                     # Chatbot/agent interface
├── main.py                      # Main entry point
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## How It Works

### Input Parameters

#### Required Inputs
1. **Requests Per Second (RPS)** (required): Target throughput in requests per second
2. **Average Input Tokens** (required): Average number of input tokens per request
3. **Average Output Tokens** (required): Average number of output tokens per request
4. **Model Size** (required): Model size in billions of parameters (e.g., 7, 70, 405)
5. **Model Precision** (required): Quantization level (FP16, INT8, INT4)

#### Optional Inputs
- **Context Window**: Maximum context length (default: 4096)
- **Batch Size**: Requests processed simultaneously per replica (default: 1)
- **GPU Type**: Specific GPU model (A100-40GB, A100-80GB, H100, etc.)
- **Target Latency**: P50/P99 latency requirements in milliseconds
- **Peak Load Multiplier**: Factor for peak traffic (default: 1.5)
- **Growth Rate**: Expected traffic growth percentage

### Output: LLM Infrastructure Capacity Plan

The system generates a comprehensive plan including:

- **GPU Resources**: GPU memory requirements, recommended GPU types, number of GPUs per replica
- **Model Replicas**: Minimum and maximum replica counts for horizontal scaling
- **Throughput Metrics**: Tokens per second (TPS) per replica, total TPS capacity, effective RPS
- **Latency Estimation**: Time to first token (TTFT), inter-token latency (ITL), end-to-end latency
- **Scaling Recommendations**: Auto-scaling thresholds based on RPS, GPU utilization, and latency
- **Cost Estimates** (optional): Monthly GPU costs and cost per million tokens

### Forecasting Methodology

The system uses LLM-specific formulas to calculate requirements:

#### GPU Memory Calculation
```
GPU Memory = Model Weights + KV Cache + Activations + Overhead

Model Weights = model_params × bytes_per_param
  - FP16/BF16: 2 bytes per parameter
  - INT8: 1 byte per parameter
  - INT4: 0.5 bytes per parameter

KV Cache = 2 × num_layers × hidden_dim × context_length × batch_size × bytes_per_param
```

#### Throughput Calculation
```
TPS per Replica = (GPU Memory Bandwidth × Efficiency) / (Model Size × Bytes per Param)
Effective RPS = TPS / avg_output_tokens
Required Replicas = ceil(target_RPS / RPS_per_replica) × safety_margin
```

#### Latency Estimation
```
TTFT = (Input Tokens × Prefill Time per Token) + Model Loading Overhead
ITL = Model Size / (GPU Memory Bandwidth × Efficiency)
E2E Latency = TTFT + (Output Tokens × ITL)
```

Default parameters can be adjusted in `config.py`.

## Example Output

```
📋 LLM INFRASTRUCTURE CAPACITY PLAN
============================================================

📊 INPUT SUMMARY:
   Requests Per Second: 10.0
   Average Input Tokens: 500
   Average Output Tokens: 200
   Model Size: 70B parameters
   Precision: FP16
   Context Window: 8192

🎮 GPU RESOURCES:
   GPU Memory per Replica: 140.0 GB
   Recommended GPU Type: A100-80GB
   GPUs per Replica: 2
   Minimum Replicas: 2
   Maximum Replicas: 8
   Total GPUs: 4 - 16

⚡ THROUGHPUT:
   TPS per Replica: 1,200 tokens/sec
   Total TPS Capacity: 2,400 - 9,600 tokens/sec
   Effective RPS Capacity: 12 - 48 requests/sec
   Estimated TTFT: 150 ms
   Estimated ITL: 25 ms
   Meets Latency SLA: ✓

📈 SCALING RECOMMENDATIONS:
   Recommended Min Replicas: 2 (for high availability)
   Recommended Max Replicas: 8 (for peak load)
   Auto-scale RPS Threshold: 8.0 requests/sec
   Auto-scale GPU Util Threshold: 70%
   Auto-scale Latency Threshold: 500 ms

💰 COST ESTIMATES:
   Monthly GPU Cost: $8,000 - $32,000
   Cost per Million Tokens: $0.50 - $2.00
   Total Monthly Cost: $8,000 - $32,000 USD
```

## Configuration

Edit `config.py` to customize:
- GPU memory bandwidth and efficiency factors
- Safety margins and scaling factors
- GPU type specifications and costs
- Latency targets and thresholds
- Auto-scaling parameters

## Use Cases

### LLM Inference (Serving)
- Real-time chatbot applications
- API endpoints serving LLM requests
- Multi-tenant model serving platforms

### LLM Agent Applications
- Autonomous agents with multiple LLM calls per request
- Chain-of-thought reasoning systems
- Tool-using agents with variable compute intensity

## Specification

See `spec-kit/` directory for comprehensive documentation:

| Document | Description |
|----------|-------------|
| `context.md` | Problem statement, goals, target users, LLM-specific assumptions |
| `requirements.md` | Functional and non-functional requirements |
| `architecture.md` | System design, component diagrams, calculation pipelines |
| `work_plan.md` | Implementation phases, task breakdown, milestones |
| `acceptance_criteria.md` | Test scenarios, edge cases, sign-off criteria |

## Development

### Adding New Features

1. **Custom Forecasting Logic**: Modify `forecast_engine.py`
2. **New Input Parameters**: Update `models.py` and `agent.py`
3. **Different Output Formats**: Extend `agent.py` formatting methods
4. **Fine-Tuning Support**: Add training-specific calculations (see architecture.md)

### Testing

Run the agent interactively to test:
```bash
python main.py
```

## Future Enhancements

- Fine-tuning capacity forecasting (training workloads)
- Time-series forecasting with historical RPS/TPS data
- Multi-cloud provider GPU recommendations
- Real-time monitoring integration
- Web-based interface
- API endpoints
- Cost optimization recommendations

## License

This project is provided as-is for LLM infrastructure capacity planning purposes.

## Support

For questions or issues, refer to the specification in `spec-kit/` or review the code documentation.
