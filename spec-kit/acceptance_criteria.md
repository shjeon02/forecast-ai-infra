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

- [ ] `python main.py` launches without error
- [ ] Interactive flow completes with valid inputs
- [ ] CLI mode works with required arguments
- [ ] JSON export produces valid JSON
- [ ] All imports resolve correctly
- [ ] No linter errors
- [ ] Edge cases still handled correctly

---

## Sign-Off Criteria

### Ready for Release

- [ ] All TS-* test scenarios pass
- [ ] All EC-* edge cases handled
- [ ] All PC-* performance criteria met
- [ ] All CC-* compatibility criteria met
- [ ] Documentation complete
- [ ] No known blocking issues
- [ ] Regression tests pass

### Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Developer | | | |
| Reviewer | | | |
| Product Owner | | | |
