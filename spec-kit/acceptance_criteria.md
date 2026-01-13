# Acceptance Criteria

## Test Scenarios

### TS-1: Interactive Mode - Happy Path

**Preconditions**: System is installed and ready to run

**Steps**:
1. Run `python main.py`
2. Enter `1000` for concurrent users
3. Enter `50000` for registered users
4. Press Enter to skip growth rate
5. Enter `n` to skip export

**Expected Results**:
- [ ] Welcome message displays correctly
- [ ] Prompts appear in correct order
- [ ] Validation passes for both inputs
- [ ] Capacity plan displays with all sections:
  - [ ] Input Summary
  - [ ] Compute Resources (CPU, Memory, Instance Types)
  - [ ] Storage Resources (Database, File, Backup, Total)
  - [ ] Network Resources (Bandwidth, Connection Limit, Load Balancers)
  - [ ] Scaling Recommendations
- [ ] All numeric values are positive and reasonable
- [ ] Exit without errors

---

### TS-2: Interactive Mode - Invalid Input Handling

**Preconditions**: System is installed and ready to run

**Steps**:
1. Run `python main.py`
2. Enter `abc` for concurrent users
3. Enter `-100` for concurrent users
4. Enter `1000` for concurrent users
5. Enter `0` for registered users
6. Enter `50000` for registered users
7. Enter `invalid` for growth rate
8. Enter `25` for growth rate
9. Complete the flow

**Expected Results**:
- [ ] Error message for non-numeric input: `❌ Please enter a valid number.`
- [ ] Error message for negative number: `❌ Concurrent users must be greater than 0.`
- [ ] Success message after valid input: `✓ Valid`
- [ ] Error message for zero registered users
- [ ] Growth rate validation works correctly
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

### TS-4: CLI Mode - Non-Interactive

**Preconditions**: System is installed and ready to run

**Steps**:
1. Run `python commands/forecast.py --concurrent-users 500 --registered-users 10000`

**Expected Results**:
- [ ] No interactive prompts appear
- [ ] Capacity plan displays with all sections
- [ ] JSON export file created (`capacity_plan.json`)
- [ ] Exit code 0

---

### TS-5: CLI Mode - With All Options

**Preconditions**: System is installed and ready to run

**Steps**:
1. Run `python commands/forecast.py --concurrent-users 2000 --registered-users 100000 --growth-rate 30 --cost --output my_plan.json`

**Expected Results**:
- [ ] Capacity plan includes cost estimates section
- [ ] Growth rate of 30% applied to calculations
- [ ] Export file named `my_plan.json` created
- [ ] JSON file contains valid, parseable JSON
- [ ] All expected fields present in JSON output

---

### TS-6: CLI Mode - Help

**Preconditions**: System is installed and ready to run

**Steps**:
1. Run `python commands/forecast.py --help`

**Expected Results**:
- [ ] Usage information displays
- [ ] All arguments documented:
  - [ ] `--concurrent-users`
  - [ ] `--registered-users`
  - [ ] `--growth-rate`
  - [ ] `--cost`
  - [ ] `--output`
- [ ] Exit code 0

---

### TS-7: JSON Export Validation

**Preconditions**: Capacity plan generated

**Steps**:
1. Generate a capacity plan with export
2. Read the exported JSON file
3. Parse with JSON parser
4. Validate structure

**Expected Results**:
- [ ] File is valid JSON (parseable)
- [ ] Contains `user_input` object with:
  - [ ] `concurrent_users` (integer)
  - [ ] `registered_users` (integer)
- [ ] Contains `compute` object with:
  - [ ] `cpu_cores` (number)
  - [ ] `memory_gb` (number)
  - [ ] `recommended_instance_types` (array)
  - [ ] `min_instances` (integer)
  - [ ] `max_instances` (integer)
- [ ] Contains `storage` object with:
  - [ ] `database_storage_gb` (number)
  - [ ] `file_storage_gb` (number)
  - [ ] `backup_storage_gb` (number)
  - [ ] `total_storage_gb` (number)
- [ ] Contains `network` object with:
  - [ ] `bandwidth_mbps` (number)
  - [ ] `connection_limit` (integer)
  - [ ] `recommended_load_balancers` (integer)
- [ ] Contains `scaling` object

---

### TS-8: Calculation Accuracy - Small Scale

**Input**: 
- Concurrent users: 100
- Registered users: 1000

**Expected Ranges**:
- [ ] CPU cores: 2-5
- [ ] Memory: 4-10 GB
- [ ] Database storage: 10-50 GB
- [ ] Bandwidth: 100-250 Mbps
- [ ] Min instances: 2-3

---

### TS-9: Calculation Accuracy - Medium Scale

**Input**: 
- Concurrent users: 1000
- Registered users: 50000

**Expected Ranges**:
- [ ] CPU cores: 15-25
- [ ] Memory: 90-150 GB
- [ ] Database storage: 500-1000 GB
- [ ] Bandwidth: 1000-2500 Mbps
- [ ] Min instances: 3-5

---

### TS-10: Calculation Accuracy - Large Scale

**Input**: 
- Concurrent users: 10000
- Registered users: 500000

**Expected Ranges**:
- [ ] CPU cores: 150-250
- [ ] Memory: 900-1500 GB
- [ ] Database storage: 5000-10000 GB
- [ ] Bandwidth: 10000-25000 Mbps
- [ ] Min instances: 10-25

---

### TS-11: Growth Rate Impact

**Steps**:
1. Generate plan with 1000 concurrent, 50000 registered, no growth rate
2. Generate plan with 1000 concurrent, 50000 registered, 50% growth rate
3. Compare storage values

**Expected Results**:
- [ ] Storage with growth rate > Storage without growth rate
- [ ] Increase factor approximately 1.5x for storage

---

### TS-12: Cost Estimation

**Input**:
- Concurrent users: 1000
- Registered users: 50000
- Cost estimation enabled

**Expected Results**:
- [ ] Monthly compute cost > $0
- [ ] Monthly storage cost > $0
- [ ] Monthly network cost > $0
- [ ] Total = compute + storage + network
- [ ] Currency is USD

---

## Edge Cases

### EC-1: Minimum Values

**Input**: concurrent_users=1, registered_users=1

**Expected Results**:
- [ ] Calculation completes without error
- [ ] Minimum 2 instances recommended (for HA)
- [ ] All resource values > 0

---

### EC-2: Very Large Numbers

**Input**: concurrent_users=1000000, registered_users=100000000

**Expected Results**:
- [ ] Calculation completes without overflow
- [ ] All resource values are positive
- [ ] Instance recommendations are reasonable

---

### EC-3: Concurrent > Registered (Warning Case)

**Input**: concurrent_users=10000, registered_users=5000

**Expected Results**:
- [ ] Calculation still proceeds
- [ ] Warning may be displayed (optional)
- [ ] Results reflect input values

---

### EC-4: Maximum Growth Rate

**Input**: growth_rate=1000 (1000%)

**Expected Results**:
- [ ] Accepted as valid input
- [ ] Storage calculations apply 11x multiplier
- [ ] No overflow or errors

---

### EC-5: Zero Optional Values

**Input**: growth_rate=0

**Expected Results**:
- [ ] Treated as "no growth"
- [ ] Default growth factor still applied
- [ ] Calculation completes normally

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

### TS-14: Historical Data Input

**Preconditions**: Time-series forecasting engine available

**Input**:
```python
usage_metrics = [
    UsageMetric(service_id="svc_001", timestamp=..., metric_name="cpu_utilization", value=45.5, unit="percent"),
    UsageMetric(service_id="svc_001", timestamp=..., metric_name="cpu_utilization", value=52.3, unit="percent"),
    # ... 12 months of data
]
```

**Expected Results**:
- [ ] Historical data parsed correctly
- [ ] Timestamps validated as datetime
- [ ] Metric names validated (cpu_utilization, memory_utilization, etc.)
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
    requested_resources=[ResourceInput(resource_type="ec2", instance_type="m5.xlarge", current_count=5)]
)
```

**Expected Results**:
- [ ] Request created with status "draft"
- [ ] Request can be submitted (status -> "submitted")
- [ ] Request can be approved/rejected
- [ ] Audit log entry created for each action

---

### TS-20: Multi-Cloud Provider Support

**Preconditions**: Cloud provider configurations loaded

**Input**: cloud_provider="gcp" with equivalent requirements

**Expected Results**:
- [ ] GCP instance types recommended (n1-standard, e2-medium, etc.)
- [ ] GCP-specific cost estimates applied
- [ ] Region names validated for GCP format
- [ ] Cross-provider comparison available

---

## Advanced Edge Cases

### EC-6: Insufficient Historical Data

**Input**: Only 3 months of historical data (< 6 minimum)

**Expected Results**:
- [ ] Warning message displayed
- [ ] Falls back to formula-based forecasting
- [ ] No crash or error

---

### EC-7: Missing Seasonal Pattern

**Input**: Flat historical data with no seasonality

**Expected Results**:
- [ ] STL handles gracefully (seasonal = 0)
- [ ] Forecast still generated
- [ ] Trend-only forecast produced

---

### EC-8: Extreme Outliers in Historical Data

**Input**: Historical data with 10x spike in one month

**Expected Results**:
- [ ] Outlier detected and handled
- [ ] Forecast not skewed by single outlier
- [ ] Warning in explanations about outlier

---

### EC-9: Invalid Service Metadata Enums

**Input**: environment="production" (not "prod")

**Expected Results**:
- [ ] Validation error with clear message
- [ ] Suggests valid values: prod, staging, dev
- [ ] Does not proceed with invalid enum

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

### Ready for Release (Basic Mode)

- [ ] All TS-1 to TS-12 test scenarios pass
- [ ] All EC-1 to EC-5 edge cases handled
- [ ] All PC-* performance criteria met
- [ ] All CC-* compatibility criteria met
- [ ] Basic documentation complete
- [ ] No known blocking issues
- [ ] Regression tests pass

### Ready for Release (Advanced Mode)

- [ ] All TS-13 to TS-20 test scenarios pass
- [ ] All EC-6 to EC-9 edge cases handled
- [ ] Time-series forecasting accuracy validated
- [ ] Service management workflow tested
- [ ] Multi-cloud support validated
- [ ] Advanced documentation complete
- [ ] Integration with basic mode verified

### Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Developer | | | |
| Reviewer | | | |
| Product Owner | | | |
