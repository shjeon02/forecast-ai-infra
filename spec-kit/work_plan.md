# Work Plan

## Implementation Phases

### Phase 0: Project Setup
**Duration**: 0.5 day  
**Status**: ✅ Complete

| Task | Description | Deliverable |
|------|-------------|-------------|
| 0.1 | Create project directory structure | Directory tree |
| 0.2 | Initialize specification documents | `spec-kit/*.md` |
| 0.3 | Create README and requirements.txt | Documentation files |

---

### Phase 1: Data Layer
**Duration**: 0.5 day  
**Status**: ✅ Complete

| Task | Description | Deliverable | Dependencies |
|------|-------------|-------------|--------------|
| 1.1 | Define input data model (`UserInput`) | `models.py` | - |
| 1.2 | Define compute resources model | `models.py` | - |
| 1.3 | Define storage resources model | `models.py` | - |
| 1.4 | Define network resources model | `models.py` | - |
| 1.5 | Define scaling recommendations model | `models.py` | - |
| 1.6 | Define cost estimate model | `models.py` | - |
| 1.7 | Define capacity plan container | `models.py` | 1.1-1.6 |
| 1.8 | Create configuration constants | `config.py` | - |

**Acceptance Criteria**:
- [ ] All dataclasses defined with type hints
- [ ] `to_dict()` method on CapacityPlan works correctly
- [ ] Configuration values are reasonable defaults

---

### Phase 2: Business Logic
**Duration**: 1 day  
**Status**: ✅ Complete

| Task | Description | Deliverable | Dependencies |
|------|-------------|-------------|--------------|
| 2.1 | Implement compute calculation | `forecast_engine.py` | 1.2, 1.8 |
| 2.2 | Implement storage calculation | `forecast_engine.py` | 1.3, 1.8 |
| 2.3 | Implement network calculation | `forecast_engine.py` | 1.4, 1.8 |
| 2.4 | Implement scaling advisor | `forecast_engine.py` | 1.5, 2.1 |
| 2.5 | Implement cost estimator | `forecast_engine.py` | 1.6, 2.1-2.3 |
| 2.6 | Implement instance type recommender | `forecast_engine.py` | 1.8 |
| 2.7 | Implement plan generator | `forecast_engine.py` | 2.1-2.6 |

**Acceptance Criteria**:
- [ ] All calculations produce valid numeric results
- [ ] Safety margins applied correctly
- [ ] Peak multipliers applied correctly
- [ ] Instance recommendations are appropriate for load

---

### Phase 3: Presentation Layer
**Duration**: 1 day  
**Status**: ✅ Complete

| Task | Description | Deliverable | Dependencies |
|------|-------------|-------------|--------------|
| 3.1 | Implement welcome message | `agent.py` | - |
| 3.2 | Implement input prompts | `agent.py` | - |
| 3.3 | Implement input validation | `agent.py` | 1.1 |
| 3.4 | Implement input collection flow | `agent.py` | 3.1-3.3 |
| 3.5 | Implement output formatting | `agent.py` | 1.7 |
| 3.6 | Implement JSON export | `agent.py` | 1.7 |
| 3.7 | Implement main interaction loop | `agent.py` | 3.4-3.6, 2.7 |

**Acceptance Criteria**:
- [ ] Interactive flow completes without errors
- [ ] Invalid inputs are rejected with clear messages
- [ ] Output is well-formatted and readable
- [ ] JSON export produces valid JSON

---

### Phase 4: CLI Interface
**Duration**: 0.5 day  
**Status**: ✅ Complete

| Task | Description | Deliverable | Dependencies |
|------|-------------|-------------|--------------|
| 4.1 | Implement argument parser | `commands/forecast.py` | - |
| 4.2 | Implement non-interactive mode | `commands/forecast.py` | 2.7 |
| 4.3 | Implement file output option | `commands/forecast.py` | 3.6 |
| 4.4 | Create main entry point | `main.py` | 3.7 |

**Acceptance Criteria**:
- [ ] `--help` displays usage information
- [ ] Non-interactive mode works with all required args
- [ ] File output creates valid JSON file
- [ ] Interactive mode launches correctly from main.py

---

### Phase 4b: JSON Configuration Mode
**Duration**: 1 day  
**Status**: ✅ Complete

| Task | Description | Deliverable | Dependencies |
|------|-------------|-------------|--------------|
| 4b.1 | Define JSON schema for inference config | `schemas/inference_schema.json` | 5.1 |
| 4b.2 | Define JSON schema for training config | `schemas/training_schema.json` | 5.2 |
| 4b.3 | Implement JSON config loader | `config_loader.py` | 4b.1, 4b.2 |
| 4b.4 | Implement JSON schema validation | `config_loader.py` | 4b.3 |
| 4b.5 | Implement CLI argument override merging | `config_loader.py` | 4b.3 |
| 4b.6 | Implement input mode selection logic | `commands/forecast.py` | 4.1 |
| 4b.7 | Implement `--generate-config` command | `commands/forecast.py` | 4b.1, 4b.2 |
| 4b.8 | Create template config files | `templates/*.json` | 4b.1, 4b.2 |
| 4b.9 | Add JSON config error handling | `config_loader.py` | 4b.3 |

**Acceptance Criteria**:
- [ ] JSON config files parsed correctly
- [ ] Schema validation catches missing required fields
- [ ] CLI arguments override JSON values correctly
- [ ] Invalid JSON shows clear error message
- [ ] File not found shows clear error message
- [ ] `--generate-config` creates valid template files
- [ ] Mode selection priority works correctly (no args → interactive, --config → JSON, CLI args → CLI)

---

### Phase 5: LLM-Specific Data Models
**Duration**: 1.5 days  
**Status**: ✅ Complete

| Task | Description | Deliverable | Dependencies |
|------|-------------|-------------|--------------|
| 5.1 | Define LLMWorkloadInput model (inference) | `models.py` | 1.1 |
| 5.2 | Define TrainingInput model (fine-tuning) | `models.py` | - |
| 5.3 | Define ModelConfig model | `models.py` | - |
| 5.4 | Define GPUConfig model | `models.py` | - |
| 5.5 | Define LLMCapacityPlan model (inference output) | `models.py` | 5.1, 5.3 |
| 5.6 | Define TrainingCapacityPlan model (training output) | `models.py` | 5.2, 5.3 |
| 5.7 | Define GPU specs configuration (A100, H100, etc.) | `config.py` | - |
| 5.8 | Define optimizer memory multipliers | `config.py` | - |

**Acceptance Criteria**:
- [ ] All LLM-specific dataclasses defined with type hints
- [ ] Inference and training models are separate
- [ ] GPU specs accurately reflect real hardware
- [ ] Serialization to dict/JSON works correctly

---

### Phase 6: Inference Forecasting Engine
**Duration**: 2 days  
**Status**: ✅ Complete

| Task | Description | Deliverable | Dependencies |
|------|-------------|-------------|--------------|
| 6.1 | Implement GPU memory calculation (inference) | `inference_engine.py` | 5.3, 5.4, 5.7 |
| 6.2 | Implement KV cache calculation | `inference_engine.py` | 6.1 |
| 6.3 | Implement throughput (TPS) calculation | `inference_engine.py` | 6.1 |
| 6.4 | Implement latency estimation (TTFT, ITL) | `inference_engine.py` | 6.3 |
| 6.5 | Implement replica count calculation | `inference_engine.py` | 6.3, 6.4 |
| 6.6 | Implement GPU type recommendation | `inference_engine.py` | 6.1, 5.7 |
| 6.7 | Implement inference cost estimation | `inference_engine.py` | 6.5 |
| 6.8 | Implement generate_inference_plan() | `inference_engine.py` | 6.1-6.7 |

**Acceptance Criteria**:
- [ ] GPU memory accurately calculated for model + KV cache
- [ ] TPS calculation matches expected throughput
- [ ] Latency estimation is reasonable
- [ ] Replica recommendations meet target RPS
- [ ] Cost per token calculation is accurate

---

### Phase 7: Fine-Tuning Forecasting Engine
**Duration**: 2 days  
**Status**: ✅ Complete

| Task | Description | Deliverable | Dependencies |
|------|-------------|-------------|--------------|
| 7.1 | Implement GPU memory calculation (training) | `training_engine.py` | 5.3, 5.4, 5.8 |
| 7.2 | Implement gradient memory calculation | `training_engine.py` | 7.1 |
| 7.3 | Implement optimizer states calculation | `training_engine.py` | 7.1, 5.8 |
| 7.4 | Implement activations memory calculation | `training_engine.py` | 7.1 |
| 7.5 | Implement gradient checkpointing savings | `training_engine.py` | 7.4 |
| 7.6 | Implement training throughput calculation | `training_engine.py` | 7.1 |
| 7.7 | Implement training duration estimation | `training_engine.py` | 7.6 |
| 7.8 | Implement data parallel configuration | `training_engine.py` | 7.1 |
| 7.9 | Implement training cost estimation | `training_engine.py` | 7.7 |
| 7.10 | Implement generate_training_plan() | `training_engine.py` | 7.1-7.9 |

**Acceptance Criteria**:
- [ ] GPU memory accurately includes weights + gradients + optimizer + activations
- [ ] Optimizer memory matches expected (Adam: 2x weights)
- [ ] Gradient checkpointing reduces memory by ~33%
- [ ] Training duration estimation is reasonable
- [ ] Cost per training run calculation is accurate

---

### Phase 8: Time-Series Forecasting Engine
**Duration**: 2 days  
**Status**: ✅ Complete

| Task | Description | Deliverable | Dependencies |
|------|-------------|-------------|--------------|
| 8.1 | Implement STL decomposition | `forecasting.py` | 5.3, 5.4 |
| 8.2 | Implement ARIMA model fitting | `forecasting.py` | 8.1 |
| 8.3 | Implement ETS model fitting | `forecasting.py` | 8.1 |
| 8.4 | Implement ensemble weighting | `forecasting.py` | 8.2, 8.3 |
| 8.5 | Implement confidence intervals (P80, P95) | `forecasting.py` | 8.4 |
| 8.6 | Implement scenario adjustments | `forecasting.py` | 8.4 |
| 8.7 | Implement attribution & explanations | `forecasting.py` | 8.4 |
| 8.8 | Implement rolling update/retraining | `forecasting.py` | 8.4 |

**Acceptance Criteria**:
- [ ] STL decomposition extracts trend, seasonal, residual
- [ ] ARIMA produces valid forecasts with AIC selection
- [ ] ETS produces valid forecasts
- [ ] Ensemble combines models with configurable weights
- [ ] Scenarios apply correct multipliers

---

### Phase 9: Service & Resource Management
**Duration**: 1 day  
**Status**: ✅ Complete

| Task | Description | Deliverable | Dependencies |
|------|-------------|-------------|--------------|
| 9.1 | Implement service metadata handling | `services.py` | 5.1 |
| 9.2 | Implement cloud provider configuration | `config.py` | - |
| 9.3 | Add provider-specific GPU pricing | `config.py` | 9.2 |
| 9.4 | Implement mode selection logic | `forecast_engine.py` | 6.8, 7.10 |

**Acceptance Criteria**:
- [ ] Service metadata properly validated
- [ ] Cloud provider GPU pricing loaded correctly
- [ ] Mode selection (inference vs training) works correctly

---

### Phase 10: Testing & Validation
**Duration**: 2 days  
**Status**: ✅ Complete

| Task | Description | Deliverable | Dependencies |
|------|-------------|-------------|--------------|
| 10.1 | Manual testing of inference mode | Test report | 6.8 |
| 10.2 | Manual testing of fine-tuning mode | Test report | 7.10 |
| 10.3 | Manual testing of CLI mode | Test report | 4.2 |
| 10.4 | Edge case testing | Test report | 10.1-10.3 |
| 10.5 | Cross-platform validation | Test report | 10.1-10.4 |
| 10.6 | GPU memory calculation accuracy testing | Test report | 6.1, 7.1 |
| 10.7 | Time-series forecasting accuracy testing | Test report | 8.8 |
| 10.8 | Scenario analysis validation | Test report | 8.6 |

**Acceptance Criteria**:
- [ ] All test scenarios pass for both modes
- [ ] GPU memory calculations match expected values
- [ ] No crashes on edge cases
- [ ] Performance meets NFR targets
- [ ] Works on macOS, Linux, Windows
- [ ] Time-series forecasts within expected accuracy

---

### Phase 11: Documentation
**Duration**: 1 day  
**Status**: ✅ Complete

| Task | Description | Deliverable | Dependencies |
|------|-------------|-------------|--------------|
| 11.1 | Update README with inference/training modes | `README.md` | 6.8, 7.10 |
| 11.2 | Document GPU configurations | `config.py` comments | 5.7 |
| 11.3 | Add docstrings | All `.py` files | - |
| 11.4 | Update specification docs | `spec-kit/*.md` | - |
| 11.5 | Create API documentation | `docs/api.md` | 6.8, 7.10 |
| 11.6 | Create inference usage examples | `docs/inference_examples.md` | 11.1 |
| 11.7 | Create fine-tuning usage examples | `docs/training_examples.md` | 11.1 |

**Acceptance Criteria**:
- [ ] README covers both inference and fine-tuning modes
- [ ] All public functions have docstrings
- [ ] GPU configuration parameters documented
- [ ] Inference and fine-tuning documented with examples

---

## Task Dependencies Graph

```
Phase 0 ──▶ Phase 1 ──▶ Phase 2 ──▶ Phase 3 ──▶ Phase 4 ──┐
                │                      │                   │
                └──────────────────────┴───────────────────┤
                                                           ▼
                                                    Phase 5 (LLM Models)
                                                           │
                                                           ├──────────────────────┐
                                                           │                      │
                                                           ▼                      ▼
                                                 Phase 4b (JSON Config)    Phase 6 (Inference)
                                                           │                      │
                                                           │              ┌───────┴───────┐
                                                           │              ▼               ▼
                                                           │    Phase 7 (Fine-Tuning)  Phase 8 (Time-Series)
                                                           │              │               │
                                                           └──────────────┼───────────────┘
                                                                          ▼
                                                                   Phase 9 (Services)
                                                                          │
                                                                          ▼
                                                                   Phase 10 (Testing)
                                                                          │
                                                                          ▼
                                                                   Phase 11 (Docs)
```

---

## Detailed Task Breakdown

### Critical Path

```
1.1 UserInput ──▶ 2.1 Compute ──▶ 2.7 Generator ──▶ 3.7 Agent Loop ──▶ 4.4 Main
      │              │
      │              ├──▶ 2.2 Storage
      │              │
      │              ├──▶ 2.3 Network
      │              │
      │              └──▶ 2.4 Scaling
      │
      └──▶ 3.3 Validation ──▶ 3.4 Collection
```

### Parallel Work Streams

**Stream A: Data Models**
- 1.1, 1.2, 1.3, 1.4, 1.5, 1.6 (can be done in parallel)
- 1.7 (depends on 1.1-1.6)
- 1.8 (independent)

**Stream B: Calculations**
- 2.1, 2.2, 2.3, 2.6 (can be done in parallel after Phase 1)
- 2.4 (depends on 2.1)
- 2.5 (depends on 2.1-2.3)
- 2.7 (depends on 2.1-2.6)

**Stream C: UI**
- 3.1, 3.2 (independent)
- 3.3 (depends on 1.1)
- 3.4 (depends on 3.1-3.3)
- 3.5 (depends on 1.7)
- 3.6 (depends on 1.7)
- 3.7 (depends on 3.4-3.6, 2.7)

---

## Milestones

| Milestone | Description | Target Date | Status |
|-----------|-------------|-------------|--------|
| M1 | Basic data models complete | Day 1 | ✅ |
| M2 | Formula-based engine complete | Day 2 | ✅ |
| M3 | Interactive mode working | Day 3 | ✅ |
| M4 | CLI mode working | Day 3.5 | ✅ |
| M5 | LLM-specific data models complete | Day 5 | ✅ |
| M5b | JSON config mode complete | Day 6 | ✅ |
| M6 | Inference forecasting engine complete | Day 8 | ✅ |
| M7 | Fine-tuning forecasting engine complete | Day 10 | ✅ |
| M8 | Time-series forecasting complete | Day 12 | ✅ |
| M9 | Service management complete | Day 13 | ✅ |
| M10 | Testing complete | Day 15 | ✅ |
| M11 | Documentation complete | Day 16 | ✅ |

---

## Risk Register

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Calculation formula inaccuracy | High | Medium | Validate against real-world data |
| Cross-platform compatibility issues | Medium | Low | Test on multiple OS early |
| Performance issues with large numbers | Low | Low | Use efficient algorithms |
| User confusion with prompts | Medium | Medium | User testing and iteration |

---

## Future Enhancements (Backlog)

| ID | Enhancement | Priority | Effort | Status |
|----|-------------|----------|--------|--------|
| E1 | Inference capacity forecasting | P0 | Large | 🔲 Phase 6 |
| E2 | Fine-tuning capacity forecasting | P0 | Large | 🔲 Phase 7 |
| E3 | JSON configuration input mode | P1 | Medium | ✅ Phase 4b |
| E4 | Time-series forecasting (STL, ARIMA, ETS) | P1 | Large | ✅ Phase 8 |
| E5 | Historical data import | P1 | Medium | ✅ Phase 5 |
| E6 | Multi-cloud GPU pricing | P1 | Medium | ✅ Phase 9 |
| E7 | Scenario-based what-if analysis | P1 | Medium | ✅ Phase 8 |
| E8 | Tensor parallelism support | P2 | Medium | Backlog |
| E9 | Pipeline parallelism support | P2 | Medium | Backlog |
| E10 | LoRA/QLoRA memory calculation | P2 | Medium | Backlog |
| E11 | Web-based interface | P2 | Large | Backlog |
| E12 | API endpoints (FastAPI) | P2 | Medium | Backlog |
| E13 | PDF export | P3 | Small | Backlog |
| E14 | Interactive visualization | P3 | Medium | Backlog |
| E15 | Real-time monitoring integration | P3 | Large | Backlog |
| E16 | Cost optimization recommendations | P2 | Medium | Backlog |
| E17 | YAML configuration support | P3 | Small | Backlog |
