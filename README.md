# Infrastructure Capacity Forecasting System

A chatbot/agent application that forecasts infrastructure capacity requirements based on user metrics (concurrent users and registered users).

## Features

- 🤖 **Interactive Chatbot Interface**: Conversational agent that guides you through the forecasting process
- 📊 **Comprehensive Capacity Planning**: Calculates compute, storage, and network requirements
- 📈 **Scaling Recommendations**: Provides horizontal and vertical scaling guidance
- 💰 **Cost Estimates**: Optional cost calculations for infrastructure resources
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
- Concurrent users (peak)
- Registered users (total)
- Optional: Growth rate (%)

#### Non-Interactive Mode

Provide inputs directly via command line:
```bash
python commands/forecast.py \
  --concurrent-users 1000 \
  --registered-users 50000 \
  --growth-rate 20 \
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

1. **Concurrent Users** (required): Number of users actively using the system at peak times
2. **Registered Users** (required): Total number of registered users in the system
3. **Growth Rate** (optional): Expected user growth rate percentage

### Output: Infrastructure Capacity Plan

The system generates a comprehensive plan including:

- **Compute Resources**: CPU cores, memory, recommended instance types
- **Storage Resources**: Database, file, and backup storage requirements
- **Network Resources**: Bandwidth and load balancer recommendations
- **Scaling Recommendations**: Horizontal/vertical scaling guidance with auto-scaling thresholds
- **Cost Estimates** (optional): Monthly cost breakdown by resource type

### Forecasting Methodology

The system uses configurable formulas to calculate requirements:

- **CPU**: Base CPU + (Concurrent Users × CPU per user) × Peak Multiplier × Safety Margin
- **Memory**: Base Memory + (Concurrent Users × Memory per user) × Peak Multiplier × Safety Margin
- **Storage**: Registered Users × Average Data per User × Growth Factor × Safety Margin
- **Network**: Concurrent Users × Bandwidth per User × Peak Multiplier × Safety Margin

Default parameters can be adjusted in `config.py`.

## Example Output

```
📋 INFRASTRUCTURE CAPACITY PLAN
============================================================

📊 INPUT SUMMARY:
   Concurrent Users: 1,000
   Registered Users: 50,000

💻 COMPUTE RESOURCES:
   CPU Cores: 18.75
   Memory: 9.38 GB
   Recommended Instance Types: large, xlarge
   Instance Range: 3 - 10 instances

💾 STORAGE RESOURCES:
   Database Storage: 625.00 GB
   File Storage: 2,343.75 GB
   Backup Storage: 890.63 GB
   Total Storage: 3,859.38 GB

🌐 NETWORK RESOURCES:
   Bandwidth: 1,875.00 Mbps
   Connection Limit: 10,000
   Load Balancers: 1

📈 SCALING RECOMMENDATIONS:
   Horizontal Scaling: Recommended: Start with 3 instances for high availability...
   Auto-scaling: 3 - 10 instances
   CPU Threshold: 70.0%
   Memory Threshold: 75.0%
```

## Configuration

Edit `config.py` to customize:
- Resource requirements per user
- Safety margins and scaling factors
- Instance type recommendations
- Cost estimates
- Auto-scaling thresholds

## Specification

See `spec-kit/` directory for comprehensive documentation:

| Document | Description |
|----------|-------------|
| `context.md` | Problem statement, goals, target users, assumptions |
| `requirements.md` | Functional and non-functional requirements |
| `architecture.md` | System design, component diagrams, data flow |
| `work_plan.md` | Implementation phases, task breakdown, milestones |
| `acceptance_criteria.md` | Test scenarios, edge cases, sign-off criteria |

## Development

### Adding New Features

1. **Custom Forecasting Logic**: Modify `forecast_engine.py`
2. **New Input Parameters**: Update `models.py` and `agent.py`
3. **Different Output Formats**: Extend `agent.py` formatting methods

### Testing

Run the agent interactively to test:
```bash
python main.py
```

## Future Enhancements

- Machine learning-based forecasting
- Historical data analysis
- Multi-cloud provider recommendations
- Real-time monitoring integration
- Web-based interface
- API endpoints

## License

This project is provided as-is for infrastructure capacity planning purposes.

## Support

For questions or issues, refer to the specification in `commands/spec.md` or review the code documentation.
