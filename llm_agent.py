"""
LLM Capacity Forecasting Agent.

Interactive chatbot interface for LLM inference and training capacity planning.
"""

import json
from typing import Optional, Union

from models import (
    LLMWorkloadInput,
    TrainingInput,
    ModelConfig,
    GPUConfig,
    ServiceInput,
    LLMCapacityPlan,
    TrainingCapacityPlan,
)
from inference_engine import InferenceEngine
from training_engine import TrainingEngine


class LLMCapacityAgent:
    """Agent interface for LLM capacity forecasting."""
    
    VALID_PRECISION = ["FP32", "FP16", "BF16", "INT8", "INT4"]
    VALID_GPU_TYPES = ["A100-40GB", "A100-80GB", "H100-80GB", "L4", "T4", "V100"]
    VALID_OPTIMIZERS = ["Adam", "AdamW", "Adafactor", "SGD"]
    
    def __init__(self):
        self.inference_engine = InferenceEngine()
        self.training_engine = TrainingEngine()
    
    def welcome_message(self) -> str:
        """Display welcome message."""
        return (
            "\n🤖 Welcome to LLM Capacity Forecasting Agent!\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "I'll help you forecast GPU and infrastructure requirements\n"
            "for your LLM inference or fine-tuning workloads.\n"
        )
    
    def _validate_positive_float(self, value: str, field_name: str) -> tuple[bool, Optional[float], str]:
        """Validate positive float input."""
        try:
            num = float(value.strip())
            if num <= 0:
                return False, None, f"❌ {field_name} must be greater than 0."
            return True, num, "✓ Valid"
        except ValueError:
            return False, None, "❌ Please enter a valid number."
    
    def _validate_positive_int(self, value: str, field_name: str) -> tuple[bool, Optional[int], str]:
        """Validate positive integer input."""
        try:
            num = int(value.strip())
            if num <= 0:
                return False, None, f"❌ {field_name} must be greater than 0."
            return True, num, "✓ Valid"
        except ValueError:
            return False, None, "❌ Please enter a valid integer."
    
    def _validate_enum(self, value: str, valid_values: list, field_name: str) -> tuple[bool, Optional[str], str]:
        """Validate enum input."""
        value = value.strip().upper() if value.strip() else ""
        if not value:
            return True, None, "✓ Skipped (using default)"
        if value in valid_values:
            return True, value, f"✓ {field_name}: {value}"
        return False, None, f"❌ Invalid {field_name}. Valid options: {', '.join(valid_values)}"
    
    def _collect_with_validation(self, prompt: str, validator, *args) -> any:
        """Collect input with validation loop."""
        while True:
            user_input = input(prompt)
            valid, value, message = validator(user_input, *args)
            print(message)
            if valid:
                return value
    
    def collect_mode_selection(self) -> str:
        """Collect workload type selection."""
        print("\n📋 Step 1: Select Workload Type")
        print("   1) Inference (Real-time serving)")
        print("   2) Training (Fine-tuning)")
        
        while True:
            choice = input("\n   Your choice [1/2]: ").strip()
            if choice in ["1", "inference"]:
                print("✓ Mode: Inference")
                return "inference"
            elif choice in ["2", "training"]:
                print("✓ Mode: Training")
                return "training"
            else:
                print("❌ Please enter 1 or 2")
    
    def collect_model_config(self) -> ModelConfig:
        """Collect model configuration."""
        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("🧠 Step 2: Model Configuration")
        
        # Model size
        model_size = self._collect_with_validation(
            "   Model size in billions (e.g., 7, 70, 405): ",
            self._validate_positive_float,
            "Model size"
        )
        
        # Precision
        precision = None
        while precision is None:
            precision_input = input(f"   Precision [{'/'.join(self.VALID_PRECISION)}] (default: FP16): ").strip()
            if not precision_input:
                precision = "FP16"
                print("✓ Precision: FP16 (default)")
            else:
                valid, precision, msg = self._validate_enum(precision_input, self.VALID_PRECISION, "Precision")
                print(msg)
        
        # Context window
        context_input = input("   Context window size (default: 4096): ").strip()
        if context_input:
            valid, context_window, msg = self._validate_positive_int(context_input, "Context window")
            print(msg)
            if not valid:
                context_window = 4096
        else:
            context_window = 4096
            print("✓ Context window: 4096 (default)")
        
        # Batch size
        batch_input = input("   Batch size (default: 1): ").strip()
        if batch_input:
            valid, batch_size, msg = self._validate_positive_int(batch_input, "Batch size")
            print(msg)
            if not valid:
                batch_size = 1
        else:
            batch_size = 1
            print("✓ Batch size: 1 (default)")
        
        return ModelConfig(
            model_size_billions=model_size,
            precision=precision,
            context_window=context_window,
            batch_size=batch_size,
        )
    
    def collect_inference_workload(self) -> LLMWorkloadInput:
        """Collect inference workload parameters."""
        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("📊 Step 3: Workload Parameters")
        
        # RPS
        rps = self._collect_with_validation(
            "   Requests per second (RPS): ",
            self._validate_positive_float,
            "RPS"
        )
        
        # Input tokens
        input_tokens = self._collect_with_validation(
            "   Average input tokens per request: ",
            self._validate_positive_int,
            "Input tokens"
        )
        
        # Output tokens
        output_tokens = self._collect_with_validation(
            "   Average output tokens per request: ",
            self._validate_positive_int,
            "Output tokens"
        )
        
        # Peak load multiplier
        peak_input = input("   Peak load multiplier (default: 1.5): ").strip()
        if peak_input:
            valid, peak_multiplier, msg = self._validate_positive_float(peak_input, "Peak multiplier")
            print(msg)
            if not valid:
                peak_multiplier = 1.5
        else:
            peak_multiplier = 1.5
            print("✓ Peak multiplier: 1.5 (default)")
        
        # Growth rate
        growth_input = input("   Expected growth rate % (optional): ").strip()
        growth_rate = None
        if growth_input:
            valid, growth_rate, msg = self._validate_positive_float(growth_input, "Growth rate")
            print(msg)
        
        return LLMWorkloadInput(
            requests_per_second=rps,
            avg_input_tokens=input_tokens,
            avg_output_tokens=output_tokens,
            peak_load_multiplier=peak_multiplier,
            growth_rate=growth_rate,
        )
    
    def collect_training_input(self) -> TrainingInput:
        """Collect training parameters."""
        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("📊 Step 3: Training Parameters")
        
        # Dataset size
        dataset_size = self._collect_with_validation(
            "   Dataset size in tokens (e.g., 1000000000 for 1B): ",
            self._validate_positive_int,
            "Dataset size"
        )
        
        # Sequence length
        seq_input = input("   Sequence length (default: 4096): ").strip()
        if seq_input:
            valid, sequence_length, msg = self._validate_positive_int(seq_input, "Sequence length")
            print(msg)
            if not valid:
                sequence_length = 4096
        else:
            sequence_length = 4096
            print("✓ Sequence length: 4096 (default)")
        
        # Epochs
        epochs = self._collect_with_validation(
            "   Number of epochs: ",
            self._validate_positive_int,
            "Epochs"
        )
        
        # Global batch size
        batch_size = self._collect_with_validation(
            "   Global batch size: ",
            self._validate_positive_int,
            "Global batch size"
        )
        
        # Micro batch size
        micro_input = input("   Micro batch size (default: 1): ").strip()
        if micro_input:
            valid, micro_batch, msg = self._validate_positive_int(micro_input, "Micro batch size")
            print(msg)
            if not valid:
                micro_batch = 1
        else:
            micro_batch = 1
            print("✓ Micro batch size: 1 (default)")
        
        # Optimizer
        optimizer = None
        opt_input = input(f"   Optimizer [{'/'.join(self.VALID_OPTIMIZERS)}] (default: Adam): ").strip()
        if opt_input:
            valid, optimizer, msg = self._validate_enum(opt_input, self.VALID_OPTIMIZERS, "Optimizer")
            print(msg)
        if not optimizer:
            optimizer = "Adam"
            print("✓ Optimizer: Adam (default)")
        
        # Gradient checkpointing
        gc_input = input("   Enable gradient checkpointing? [y/N]: ").strip().lower()
        gradient_checkpointing = gc_input in ["y", "yes"]
        print(f"✓ Gradient checkpointing: {'Enabled' if gradient_checkpointing else 'Disabled'}")
        
        return TrainingInput(
            dataset_size_tokens=dataset_size,
            sequence_length=sequence_length,
            num_epochs=epochs,
            global_batch_size=batch_size,
            micro_batch_size=micro_batch,
            optimizer_type=optimizer,
            gradient_checkpointing=gradient_checkpointing,
        )
    
    def collect_gpu_config(self) -> Optional[GPUConfig]:
        """Collect GPU configuration (optional)."""
        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("🖥️ Step 4: GPU Configuration (Optional - press Enter to auto-select)")
        
        # GPU type
        gpu_input = input(f"   GPU type [{'/'.join(self.VALID_GPU_TYPES)}]: ").strip()
        gpu_type = None
        if gpu_input:
            # Allow partial matches
            gpu_upper = gpu_input.upper()
            for valid_gpu in self.VALID_GPU_TYPES:
                if gpu_upper in valid_gpu or valid_gpu.startswith(gpu_upper):
                    gpu_type = valid_gpu
                    break
            if gpu_type:
                print(f"✓ GPU type: {gpu_type}")
            else:
                print(f"⚠️ Unknown GPU type, will auto-select based on requirements")
        else:
            print("✓ GPU type: Auto-select")
        
        # Target utilization
        util_input = input("   Target GPU utilization % (default: 70): ").strip()
        if util_input:
            try:
                utilization = float(util_input) / 100
                print(f"✓ Target utilization: {utilization * 100}%")
            except ValueError:
                utilization = 0.7
                print("✓ Target utilization: 70% (default)")
        else:
            utilization = 0.7
            print("✓ Target utilization: 70% (default)")
        
        return GPUConfig(
            gpu_type=gpu_type,
            target_gpu_utilization=utilization,
        )
    
    def format_inference_plan(self, plan: LLMCapacityPlan) -> str:
        """Format inference capacity plan for display."""
        output = []
        output.append("\n" + "━" * 60)
        output.append("📋 LLM INFERENCE CAPACITY PLAN")
        output.append("━" * 60)
        
        # Input Summary
        output.append("\n📊 INPUT SUMMARY:")
        output.append(f"   Requests per Second: {plan.workload_input.requests_per_second}")
        output.append(f"   Input Tokens: {plan.workload_input.avg_input_tokens}")
        output.append(f"   Output Tokens: {plan.workload_input.avg_output_tokens}")
        output.append(f"   Model Size: {plan.model_config.model_size_billions}B")
        output.append(f"   Precision: {plan.model_config.precision}")
        
        # GPU Resources
        output.append("\n🖥️ GPU RESOURCES:")
        output.append(f"   GPU Memory per Replica: {plan.gpu_resources.gpu_memory_per_replica_gb:.2f} GB")
        output.append(f"   GPUs per Replica: {plan.gpu_resources.min_gpus_per_replica}")
        output.append(f"   Recommended GPU: {plan.gpu_resources.recommended_gpu_type}")
        output.append(f"   Replicas: {plan.gpu_resources.min_replicas} - {plan.gpu_resources.max_replicas}")
        output.append(f"   Total GPUs: {plan.gpu_resources.total_gpus}")
        
        # Throughput
        output.append("\n⚡ THROUGHPUT:")
        output.append(f"   TPS per Replica: {plan.throughput.tps_per_replica:.2f}")
        output.append(f"   Total TPS Capacity: {plan.throughput.total_tps_capacity:.2f}")
        output.append(f"   Max RPS Capacity: {plan.throughput.max_rps_capacity:.2f}")
        
        # Latency
        output.append("\n⏱️ LATENCY:")
        output.append(f"   Est. TTFT: {plan.throughput.estimated_ttft_ms:.2f} ms")
        output.append(f"   Est. ITL: {plan.throughput.estimated_itl_ms:.2f} ms")
        sla_status = "✓ Meets SLA" if plan.throughput.meets_latency_sla else "⚠️ May not meet SLA"
        output.append(f"   SLA Status: {sla_status}")
        
        # Storage
        output.append("\n💾 STORAGE:")
        output.append(f"   Model Storage: {plan.storage.model_storage_gb:.2f} GB")
        output.append(f"   Cache Storage: {plan.storage.cache_storage_gb:.2f} GB")
        output.append(f"   Total Storage: {plan.storage.total_storage_gb:.2f} GB")
        
        # Scaling
        output.append("\n📈 AUTO-SCALING:")
        output.append(f"   Min/Max Replicas: {plan.scaling.recommended_min_replicas} / {plan.scaling.recommended_max_replicas}")
        output.append(f"   RPS Threshold: {plan.scaling.auto_scale_rps_threshold:.2f}")
        output.append(f"   GPU Util Threshold: {plan.scaling.auto_scale_gpu_util_threshold * 100:.0f}%")
        
        # Cost
        if plan.cost:
            output.append("\n💰 COST ESTIMATES:")
            output.append(f"   Monthly GPU Cost: ${plan.cost.monthly_gpu_cost:,.2f}")
            output.append(f"   Cost per 1M Tokens: ${plan.cost.cost_per_million_tokens:.4f}")
            output.append(f"   Total Monthly Cost: ${plan.cost.monthly_total_cost:,.2f} {plan.cost.currency}")
        
        output.append("\n" + "━" * 60)
        output.append("✅ Inference capacity plan generated successfully!")
        output.append("━" * 60 + "\n")
        
        return "\n".join(output)
    
    def format_training_plan(self, plan: TrainingCapacityPlan) -> str:
        """Format training capacity plan for display."""
        output = []
        output.append("\n" + "━" * 60)
        output.append("📋 LLM TRAINING CAPACITY PLAN")
        output.append("━" * 60)
        
        # Input Summary
        output.append("\n📊 INPUT SUMMARY:")
        output.append(f"   Dataset Size: {plan.training_input.dataset_size_tokens:,} tokens")
        output.append(f"   Sequence Length: {plan.training_input.sequence_length}")
        output.append(f"   Epochs: {plan.training_input.num_epochs}")
        output.append(f"   Global Batch Size: {plan.training_input.global_batch_size}")
        output.append(f"   Model Size: {plan.model_config.model_size_billions}B")
        output.append(f"   Optimizer: {plan.training_input.optimizer_type}")
        if plan.training_input.gradient_checkpointing:
            output.append("   Gradient Checkpointing: ✓ Enabled")
        
        # GPU Memory Breakdown
        output.append("\n🖥️ GPU MEMORY BREAKDOWN:")
        output.append(f"   Model Weights: {plan.gpu_resources.model_weights_gb:.2f} GB")
        output.append(f"   Gradients: {plan.gpu_resources.gradients_gb:.2f} GB")
        output.append(f"   Optimizer States: {plan.gpu_resources.optimizer_states_gb:.2f} GB")
        output.append(f"   Activations: {plan.gpu_resources.activations_gb:.2f} GB")
        output.append(f"   Total per GPU: {plan.gpu_resources.gpu_memory_per_gpu_gb:.2f} GB")
        
        # GPU Requirements
        output.append("\n🔧 GPU REQUIREMENTS:")
        output.append(f"   Required GPUs: {plan.gpu_resources.required_gpus}")
        output.append(f"   Recommended GPU: {plan.gpu_resources.recommended_gpu_type}")
        output.append(f"   Nodes: {plan.gpu_resources.num_nodes} × {plan.gpu_resources.gpus_per_node} GPUs")
        
        # Training Metrics
        output.append("\n⏱️ TRAINING ESTIMATES:")
        output.append(f"   Tokens/sec/GPU: {plan.training_metrics.tokens_per_second_per_gpu:.2f}")
        output.append(f"   Total Tokens/sec: {plan.training_metrics.total_tokens_per_second:.2f}")
        output.append(f"   Steps per Epoch: {plan.training_metrics.steps_per_epoch:,}")
        output.append(f"   Total Steps: {plan.training_metrics.total_steps:,}")
        output.append(f"   Duration: {plan.training_metrics.estimated_duration_hours:.1f} hours ({plan.training_metrics.estimated_duration_days:.1f} days)")
        
        # Data Parallel Config
        output.append("\n🔀 DATA PARALLEL CONFIG:")
        output.append(f"   Data Parallel Size: {plan.data_parallel_config.data_parallel_size}")
        output.append(f"   Effective Batch Size: {plan.data_parallel_config.effective_batch_size}")
        
        # Storage
        output.append("\n💾 STORAGE:")
        output.append(f"   Checkpoint Size: {plan.storage.model_checkpoint_gb:.2f} GB")
        output.append(f"   Dataset Size: {plan.storage.dataset_storage_gb:.2f} GB")
        output.append(f"   Total Storage: {plan.storage.total_storage_gb:.2f} GB")
        
        # Cost
        if plan.cost:
            output.append("\n💰 COST ESTIMATES:")
            output.append(f"   GPU Cost/Hour: ${plan.cost.gpu_cost_per_hour:.2f}")
            output.append(f"   Cost per Epoch: ${plan.cost.cost_per_epoch:,.2f}")
            output.append(f"   Total Training Cost: ${plan.cost.total_training_cost:,.2f} {plan.cost.currency}")
        
        output.append("\n" + "━" * 60)
        output.append("✅ Training capacity plan generated successfully!")
        output.append("━" * 60 + "\n")
        
        return "\n".join(output)
    
    def export_plan(
        self, 
        plan: Union[LLMCapacityPlan, TrainingCapacityPlan], 
        filename: str = "capacity_plan.json"
    ) -> str:
        """Export capacity plan to JSON file."""
        plan_dict = plan.to_dict()
        with open(filename, 'w') as f:
            json.dump(plan_dict, f, indent=2)
        return f"📄 Capacity plan exported to {filename}"
    
    def run(self, include_cost: bool = True) -> Optional[Union[LLMCapacityPlan, TrainingCapacityPlan]]:
        """Run the agent interactively."""
        try:
            print(self.welcome_message())
            
            # Step 1: Mode selection
            mode = self.collect_mode_selection()
            
            # Step 2: Model configuration
            model_config = self.collect_model_config()
            
            # Step 3: Workload/Training parameters
            if mode == "inference":
                workload = self.collect_inference_workload()
            else:
                workload = self.collect_training_input()
            
            # Step 4: GPU configuration
            gpu_config = self.collect_gpu_config()
            
            # Ask about cost estimation
            cost_input = input("\n💰 Include cost estimates? [Y/n]: ").strip().lower()
            include_cost = cost_input not in ["n", "no"]
            
            # Generate plan
            print("\n⏳ Calculating capacity requirements...")
            
            if mode == "inference":
                plan = self.inference_engine.generate_inference_plan(
                    workload, model_config, gpu_config, include_cost=include_cost
                )
                print(self.format_inference_plan(plan))
                default_filename = "inference_plan.json"
            else:
                plan = self.training_engine.generate_training_plan(
                    workload, model_config, gpu_config, include_cost=include_cost
                )
                print(self.format_training_plan(plan))
                default_filename = "training_plan.json"
            
            # Offer export
            export = input("💾 Export plan to JSON file? [Y/n]: ").strip().lower()
            if export not in ["n", "no"]:
                filename = input(f"   Enter filename (default: {default_filename}): ").strip()
                if not filename:
                    filename = default_filename
                if not filename.endswith('.json'):
                    filename += '.json'
                print(self.export_plan(plan, filename))
            
            return plan
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Exiting...")
            return None
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
