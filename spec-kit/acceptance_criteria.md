# Acceptance Criteria

## Test Scenarios

### TS-1: Interactive Mode - Inference Happy Path

**Preconditions**: System is installed and ready to run

**Steps**:
1. Run `python main.py`
2. Select `inference` mode
3. Enter `10.0` for RPS
4. Enter `500` for input tokens
5. Enter `200` for output tokens
6. Enter `70` for model size (billions)
7. Enter `FP16` for precision
8. Press Enter to skip optional inputs
9. Enter `n` to skip export

**Expected Results**:
- [ ] Welcome message displays correctly
- [ ] Mode selection prompt appears
- [ ] Prompts appear in correct order
- [ ] Validation passes for all inputs
- [ ] Capacity plan displays with all sections:
  - [ ] Input Summary (RPS, tokens, model config)
  - [ ] GPU Resources (memory, count, type, replicas)
  - [ ] Throughput (TPS, latency)
  - [ ] Storage Resources
  - [ ] Network Resources
  - [ ] Scaling Recommendations
  - [ ] Cost Estimation
- [ ] All numeric values are positive and reasonable
- [ ] Exit without errors

---

### TS-2: Interactive Mode - Invalid Input Handling

**Preconditions**: System is installed and ready to run

**Steps**:
1. Run `python main.py`
2. Select `inference` mode
3. Enter `abc` for RPS
4. Enter `-10` for RPS
5. Enter `10.0` for RPS
6. Enter `0` for input tokens
7. Enter `500` for input tokens
8. Enter `invalid` for precision
9. Enter `FP16` for precision
10. Complete the flow

**Expected Results**:
- [ ] Error message for non-numeric input: `❌ Please enter a valid number.`
- [ ] Error message for negative RPS: `❌ RPS must be greater than 0.`
- [ ] Success message after valid input: `✓ Valid`
- [ ] Error message for zero tokens
- [ ] Precision validation works correctly (FP32, FP16, BF16, INT8, INT4)
- [ ] Flow continues after valid corrections
- [ ] Final capacity plan generated successfully

---

### TS-3: Interactive Mode - Keyboard Interrupt

**Preconditions**: System is installed and ready to run

**Steps**:
1. Run `python main.py`
2. Press Ctrl+C during any prompt

**Expected Results**:
- [ ] Exit message displays: `👋 Goodbye! Exiting...`
- [ ] No stack trace or error message
- [ ] Clean process exit (exit code 0 or 130)

---

### TS-4: CLI Mode - Inference Non-Interactive

**Preconditions**: System is installed and ready to run

**Steps**:
1. Run `python commands/forecast.py --mode inference --rps 10.0 --input-tokens 500 --output-tokens 200 --model-size 70 --precision FP16`

**Expected Results**:
- [ ] No interactive prompts appear
- [ ] Capacity plan displays with all sections
- [ ] JSON export file created (`capacity_plan.json`)
- [ ] Exit code 0

---

### TS-5: CLI Mode - Fine-Tuning Non-Interactive

**Preconditions**: System is installed and ready to run

**Steps**:
1. Run `python commands/forecast.py --mode training --dataset-size 1000000000 --sequence-length 4096 --epochs 3 --global-batch-size 64 --model-size 70 --precision BF16`

**Expected Results**:
- [ ] No interactive prompts appear
- [ ] Training capacity plan displays with all sections:
  - [ ] GPU memory breakdown (weights, gradients, optimizer, activations)
  - [ ] Training duration estimate
  - [ ] GPU count recommendation
  - [ ] Training cost estimate
- [ ] JSON export file created
- [ ] Exit code 0

---

### TS-6: CLI Mode - Help

**Preconditions**: System is installed and ready to run

**Steps**:
1. Run `python commands/forecast.py --help`

**Expected Results**:
- [ ] Usage information displays
- [ ] All inference arguments documented:
  - [ ] `--mode` (inference/training)
  - [ ] `--rps`
  - [ ] `--input-tokens`, `--output-tokens`
  - [ ] `--model-size`, `--precision`
  - [ ] `--gpu-type`
- [ ] All training arguments documented:
  - [ ] `--dataset-size`
  - [ ] `--sequence-length`
  - [ ] `--epochs`
  - [ ] `--global-batch-size`
  - [ ] `--optimizer`
  - [ ] `--gradient-checkpointing`
- [ ] Exit code 0

---

### TS-7: JSON Export Validation (Inference)

**Preconditions**: Inference capacity plan generated

**Steps**:
1. Generate an inference capacity plan with export
2. Read the exported JSON file
3. Parse with JSON parser
4. Validate structure

**Expected Results**:
- [ ] File is valid JSON (parseable)
- [ ] Contains `workload_input` object with:
  - [ ] `requests_per_second` (float)
  - [ ] `avg_input_tokens` (integer)
  - [ ] `avg_output_tokens` (integer)
- [ ] Contains `model_config` object with:
  - [ ] `model_size_billions` (float)
  - [ ] `precision` (string)
  - [ ] `context_window` (integer)
- [ ] Contains `gpu_resources` object with:
  - [ ] `gpu_memory_per_replica_gb` (float)
  - [ ] `min_gpus_per_replica` (integer)
  - [ ] `recommended_gpu_type` (string)
  - [ ] `min_replicas` (integer)
  - [ ] `max_replicas` (integer)
  - [ ] `total_gpus` (integer)
- [ ] Contains `throughput` object with:
  - [ ] `tps_per_replica` (float)
  - [ ] `total_tps_capacity` (float)
  - [ ] `estimated_ttft_ms` (float)
  - [ ] `estimated_itl_ms` (float)
- [ ] Contains `scaling` object
- [ ] Contains `cost` object with:
  - [ ] `cost_per_million_tokens` (float)

---

### TS-8: Inference GPU Memory Calculation - 7B Model

**Input**: 
- Model size: 7B
- Precision: FP16
- Context window: 4096
- Batch size: 1

**Expected Ranges**:
- [ ] Model weights: ~14 GB (7B × 2 bytes)
- [ ] KV cache: ~0.5-1 GB
- [ ] Total GPU memory: 15-20 GB
- [ ] Recommended GPU: A100-40GB or A100-80GB
- [ ] GPUs per replica: 1

---

### TS-9: Inference GPU Memory Calculation - 70B Model

**Input**: 
- Model size: 70B
- Precision: FP16
- Context window: 8192
- Batch size: 4

**Expected Ranges**:
- [ ] Model weights: ~140 GB (70B × 2 bytes)
- [ ] KV cache: ~8-16 GB
- [ ] Total GPU memory: 150-180 GB
- [ ] Recommended GPU: H100-80GB
- [ ] GPUs per replica: 2-3 (for tensor parallelism)

---

### TS-10: Fine-Tuning GPU Memory Calculation - 7B Model

**Input**: 
- Model size: 7B
- Precision: BF16
- Sequence length: 4096
- Batch size: 1
- Optimizer: Adam

**Expected Ranges**:
- [ ] Model weights: ~14 GB
- [ ] Gradients: ~14 GB
- [ ] Optimizer states: ~28 GB (Adam: 2x weights)
- [ ] Activations: ~8-16 GB
- [ ] Total GPU memory: 60-80 GB
- [ ] Required GPUs: 1-2 (A100-80GB)

---

### TS-11: Fine-Tuning Duration Estimation

**Input**:
- Model size: 7B
- Dataset size: 1B tokens
- Sequence length: 4096
- Epochs: 3
- Global batch size: 64

**Expected Results**:
- [ ] Tokens per second estimation is reasonable
- [ ] Training duration in hours/days format
- [ ] Duration scales linearly with epochs
- [ ] Duration scales inversely with GPU count

---

### TS-12: Inference Cost Estimation

**Input**:
- Model size: 70B
- RPS: 10
- GPU type: A100-80GB
- Cost estimation enabled

**Expected Results**:
- [ ] Monthly GPU cost > $0
- [ ] Cost per million tokens > $0
- [ ] Monthly total cost calculated
- [ ] Currency is USD

---

### TS-12b: Fine-Tuning Cost Estimation

**Input**:
- Model size: 70B
- Dataset size: 1B tokens
- Epochs: 3
- GPU type: A100-80GB
- Cost estimation enabled

**Expected Results**:
- [ ] Total training cost > $0
- [ ] Cost per epoch calculated
- [ ] Cost = GPU hourly rate × duration × GPU count
- [ ] Currency is USD

---

## Edge Cases

### EC-1: Minimum Inference Values

**Input**: RPS=0.1, input_tokens=1, output_tokens=1, model_size=0.5B

**Expected Results**:
- [ ] Calculation completes without error
- [ ] Minimum 1 replica recommended
- [ ] All resource values > 0

---

### EC-2: Very Large Model (405B)

**Input**: model_size=405B, precision=FP16

**Expected Results**:
- [ ] Calculation completes without overflow
- [ ] GPU memory: ~810 GB calculated
- [ ] Multi-GPU recommendation (8+ GPUs)
- [ ] Tensor parallelism recommendation

---

### EC-3: High RPS Scenario

**Input**: RPS=1000, model_size=7B

**Expected Results**:
- [ ] Many replicas recommended
- [ ] Auto-scaling thresholds configured
- [ ] Cost estimation reasonable

---

### EC-4: INT4 Quantization Memory

**Input**: model_size=70B, precision=INT4

**Expected Results**:
- [ ] GPU memory: ~35 GB (70B × 0.5 bytes)
- [ ] Fits in single A100-40GB
- [ ] Lower throughput noted

---

### EC-5: Zero Growth Rate (Fixed Issue)

**Input**: growth_rate=0

**Expected Results**:
- [ ] Treated as "no growth" (multiplier = 1.0)
- [ ] NOT treated as falsy value
- [ ] Calculation completes normally with 0% growth

---

### EC-6: Gradient Checkpointing Memory Savings

**Input**: model_size=70B, gradient_checkpointing=true

**Expected Results**:
- [ ] Activations memory reduced by ~33%
- [ ] Training speed note (~20% slower)
- [ ] Overall memory fits in fewer GPUs

---

## Advanced Features Test Scenarios

### TS-13: Service Metadata Input

**Preconditions**: System supports advanced mode

**Input**:
```python
ServiceInput(
    service_id="svc_001",
    service_name="Search API",
    team_id="team_platform",
    environment="prod",
    criticality="high",
    cloud_provider="aws",
    region="us-east-1"
)
```

**Expected Results**:
- [ ] Service metadata validated correctly
- [ ] Environment enum validated (prod, staging, dev)
- [ ] Criticality enum validated (high, medium, low)
- [ ] Cloud provider enum validated (aws, gcp, azure)
- [ ] Region accepted as free-form string

---

### TS-14: Historical Data Input (LLM Metrics)

**Preconditions**: Time-series forecasting engine available

**Input**:
```python
usage_metrics = [
    LLMUsageMetric(service_id="svc_001", timestamp=..., metric_name="rps", value=10.5, unit="requests/sec"),
    LLMUsageMetric(service_id="svc_001", timestamp=..., metric_name="tps", value=1500.0, unit="tokens/sec"),
    LLMUsageMetric(service_id="svc_001", timestamp=..., metric_name="ttft_ms", value=85.0, unit="ms"),
    LLMUsageMetric(service_id="svc_001", timestamp=..., metric_name="gpu_utilization", value=72.5, unit="percent"),
    # ... 12 months of data
]
```

**Expected Results**:
- [ ] Historical data parsed correctly
- [ ] Timestamps validated as datetime
- [ ] LLM-specific metric names validated (rps, tps, ttft_ms, itl_ms, gpu_utilization)
- [ ] Values validated as positive floats
- [ ] Minimum 6 data points required for forecasting

---

### TS-15: Time-Series Forecasting (STL + Ensemble)

**Preconditions**: 12+ months of historical cost data

**Input**: Monthly cost data with seasonal pattern

**Expected Results**:
- [ ] STL decomposition produces trend, seasonal, residual
- [ ] ARIMA model fits without error
- [ ] ETS model fits without error
- [ ] Ensemble combines models with weights
- [ ] Forecast output includes:
  - [ ] `forecast`: List of predicted values
  - [ ] `intervals`: P80 and P95 confidence bands
  - [ ] `components`: Trend, seasonal, residual
  - [ ] `explanations`: Human-readable insights
  - [ ] `attributions`: Impact breakdown

---

### TS-16: Scenario Analysis

**Preconditions**: Base forecast generated

**Input**: scenario="pessimistic"

**Expected Results**:
- [ ] Baseline forecast calculated first
- [ ] Pessimistic scenario applies 1.15x multiplier
- [ ] Optimistic scenario applies 0.85x multiplier
- [ ] Spike scenario applies 1.5x multiplier
- [ ] Scenario comparison output generated

---

### TS-17: Forecast with Confidence Intervals

**Preconditions**: Time-series forecast generated

**Expected Results**:
- [ ] P80 interval contains ~80% of actual values
- [ ] P95 interval contains ~95% of actual values
- [ ] Lower bound ≤ Forecast ≤ Upper bound
- [ ] Intervals widen as horizon increases

---

### TS-18: Rolling Update / Retraining

**Preconditions**: Prior forecast exists, new actuals available

**Input**:
- Prior forecast for 6 months
- Actual values for first 3 months
- Variance threshold: 10%

**Expected Results**:
- [ ] MAPE calculated correctly
- [ ] If MAPE > threshold, model retrains
- [ ] Updated forecast returned
- [ ] Alert generated if retraining triggered

---

### TS-19: Capacity Request Workflow

**Preconditions**: Service exists in system

**Input**:
```python
CapacityRequest(
    service_id="svc_001",
    requester_email="engineer@company.com",
    justification="Expected traffic increase for product launch",
    workload_type="inference",
    requested_resources=GPUResourceRequest(
        gpu_type="A100-80GB",
        current_replicas=2,
        requested_replicas=5,
        model_size_billions=70,
        target_rps=50.0
    )
)
```

**Expected Results**:
- [ ] Request created with status "draft"
- [ ] Request can be submitted (status -> "submitted")
- [ ] Request can be approved/rejected
- [ ] GPU cost delta calculated
- [ ] Audit log entry created for each action

---

### TS-20: Multi-Cloud GPU Provider Support

**Preconditions**: Cloud provider GPU configurations loaded

**Input**: cloud_provider="gcp" with equivalent requirements

**Expected Results**:
- [ ] GCP GPU types recommended (A100, L4, T4)
- [ ] GCP-specific GPU pricing applied
- [ ] Region names validated for GCP format
- [ ] Cross-provider GPU cost comparison available
- [ ] GPU availability by region noted

---

### TS-21: Fine-Tuning Mode - Interactive

**Preconditions**: System supports training mode

**Steps**:
1. Run `python main.py`
2. Select `training` mode
3. Enter dataset size, sequence length, epochs, batch size
4. Enter model configuration
5. Complete the flow

**Expected Results**:
- [ ] Training mode prompts are different from inference
- [ ] GPU memory breakdown shows:
  - [ ] Model weights
  - [ ] Gradients
  - [ ] Optimizer states
  - [ ] Activations
- [ ] Training duration estimate displayed
- [ ] GPU count recommendation displayed
- [ ] Training cost estimate displayed

---

### TS-22: Fine-Tuning with Gradient Checkpointing

**Input**:
- Model size: 70B
- Gradient checkpointing: enabled

**Expected Results**:
- [ ] Activations memory reduced by ~33%
- [ ] Training speed note displayed (~20% slower)
- [ ] Total GPU memory reduced
- [ ] Possibly fits in fewer GPUs

---

### TS-23: Optimizer Memory Comparison

**Input**: Same model, different optimizers (Adam vs Adafactor)

**Expected Results**:
- [ ] Adam: Optimizer states = 2x weights
- [ ] Adafactor: Optimizer states = 0.5x weights
- [ ] Total memory difference reflected
- [ ] GPU count may differ

---

### TS-24: JSON Export Validation (Training)

**Preconditions**: Training capacity plan generated

**Expected Results**:
- [ ] Contains `training_input` object with:
  - [ ] `dataset_size_tokens`
  - [ ] `sequence_length`
  - [ ] `num_epochs`
  - [ ] `global_batch_size`
  - [ ] `optimizer_type`
- [ ] Contains `gpu_resources` with training breakdown
- [ ] Contains `training_metrics` object with:
  - [ ] `tokens_per_second_per_gpu`
  - [ ] `estimated_duration_hours`
  - [ ] `estimated_duration_days`
- [ ] Contains `cost` object with:
  - [ ] `total_training_cost`
  - [ ] `cost_per_epoch`

---

## Advanced Edge Cases

### EC-7: Insufficient Historical Data

**Input**: Only 3 months of historical data (< 6 minimum)

**Expected Results**:
- [ ] Warning message displayed
- [ ] Falls back to formula-based forecasting
- [ ] No crash or error

---

### EC-8: Missing Seasonal Pattern

**Input**: Flat historical data with no seasonality

**Expected Results**:
- [ ] STL handles gracefully (seasonal = 0)
- [ ] Forecast still generated
- [ ] Trend-only forecast produced

---

### EC-9: Extreme Outliers in Historical Data

**Input**: Historical data with 10x spike in one month

**Expected Results**:
- [ ] Outlier detected and handled
- [ ] Forecast not skewed by single outlier
- [ ] Warning in explanations about outlier

---

### EC-10: Invalid Service Metadata Enums

**Input**: environment="production" (not "prod")

**Expected Results**:
- [ ] Validation error with clear message
- [ ] Suggests valid values: prod, staging, dev
- [ ] Does not proceed with invalid enum

---

### EC-11: Invalid Precision Value

**Input**: precision="float16" (not "FP16")

**Expected Results**:
- [ ] Validation error with clear message
- [ ] Suggests valid values: FP32, FP16, BF16, INT8, INT4
- [ ] Does not proceed with invalid precision

---

### EC-12: Model Too Large for Single GPU

**Input**: model_size=70B, precision=FP16, gpu_type=A100-40GB

**Expected Results**:
- [ ] Detects model won't fit (140GB > 40GB)
- [ ] Recommends multi-GPU or larger GPU
- [ ] Suggests alternative: A100-80GB or tensor parallelism

---

## Performance Criteria

### PC-1: Calculation Speed

**Test**: Generate 100 capacity plans in sequence

**Expected Result**: 
- [ ] Total time < 10 seconds
- [ ] Average time < 100ms per plan

---

### PC-2: Memory Usage

**Test**: Generate capacity plan for 10M concurrent users

**Expected Result**:
- [ ] Peak memory usage < 100 MB
- [ ] No memory leaks (memory freed after completion)

---

### PC-3: Startup Time

**Test**: Time from command execution to first prompt

**Expected Result**:
- [ ] Startup time < 2 seconds

---

## Compatibility Criteria

### CC-1: Python Version

**Test**: Run on Python 3.10, 3.11, 3.12, 3.13

**Expected Result**:
- [ ] All tests pass on each version

---

### CC-2: Operating Systems

**Test**: Run full test suite on macOS, Linux, Windows

**Expected Result**:
- [ ] All tests pass on each OS
- [ ] Output formatting correct on each OS

---

## Regression Test Checklist

After any code change, verify:

### Basic Mode
- [ ] `python main.py` launches without error
- [ ] Interactive flow completes with valid inputs
- [ ] CLI mode works with required arguments
- [ ] JSON export produces valid JSON
- [ ] All imports resolve correctly
- [ ] No linter errors
- [ ] Edge cases still handled correctly
- [ ] growth_rate=0 handled correctly (not treated as None)

### Advanced Mode
- [ ] Service metadata input validates correctly
- [ ] Historical data input parses correctly
- [ ] Time-series forecasting produces valid output
- [ ] STL decomposition runs without error
- [ ] ARIMA/ETS ensemble works correctly
- [ ] Scenario analysis applies correct multipliers
- [ ] Confidence intervals calculated correctly
- [ ] Capacity request workflow functions correctly

---

## Sign-Off Criteria

### Ready for Release (Inference Mode)

- [ ] All TS-1 to TS-9 test scenarios pass (inference)
- [ ] TS-12 cost estimation pass
- [ ] All EC-1 to EC-5 edge cases handled
- [ ] All PC-* performance criteria met
- [ ] All CC-* compatibility criteria met
- [ ] Basic documentation complete
- [ ] No known blocking issues
- [ ] Regression tests pass

### Ready for Release (Fine-Tuning Mode)

- [ ] TS-5, TS-10, TS-11, TS-12b test scenarios pass (training)
- [ ] TS-21 to TS-24 fine-tuning scenarios pass
- [ ] EC-6 gradient checkpointing edge case handled
- [ ] Training duration calculation verified
- [ ] Optimizer memory calculation verified

### Ready for Release (Advanced Mode)

- [ ] All TS-13 to TS-20 test scenarios pass
- [ ] All EC-7 to EC-12 edge cases handled
- [ ] Time-series forecasting accuracy validated
- [ ] Service management workflow tested
- [ ] Multi-cloud GPU support validated
- [ ] Advanced documentation complete
- [ ] Integration with inference/training modes verified

### Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Developer | | | |
| Reviewer | | | |
| Product Owner | | | |
