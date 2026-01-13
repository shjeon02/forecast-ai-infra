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

```
                    UserInput
                        │
                        ▼
        ┌───────────────────────────────┐
        │     Extract Parameters        │
        │  - concurrent_users           │
        │  - registered_users           │
        │  - peak_load_multiplier       │
        │  - growth_rate                │
        └───────────────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ Compute  │  │ Storage  │  │ Network  │
    │  Calc    │  │  Calc    │  │  Calc    │
    └──────────┘  └──────────┘  └──────────┘
          │             │             │
          ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ Apply    │  │ Apply    │  │ Apply    │
    │ Multiplier│ │ Growth   │  │ Multiplier│
    └──────────┘  └──────────┘  └──────────┘
          │             │             │
          ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ Apply    │  │ Apply    │  │ Apply    │
    │ Safety   │  │ Safety   │  │ Safety   │
    │ Margin   │  │ Margin   │  │ Margin   │
    └──────────┘  └──────────┘  └──────────┘
          │             │             │
          └─────────────┼─────────────┘
                        ▼
              ┌─────────────────┐
              │ Scaling Advisor │
              └─────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │ Cost Estimator  │
              │   (optional)    │
              └─────────────────┘
                        │
                        ▼
                  CapacityPlan
```

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
