#!/usr/bin/env python3
"""
Command-line entry point for LLM infrastructure capacity forecasting.

Supports three input modes:
1. Interactive mode: Conversational prompts (default when no args)
2. JSON config mode: Load configuration from JSON file (--config)
3. CLI args mode: Direct command-line arguments (--mode --rps etc.)
"""

import sys
import os
import json
import argparse
from typing import Optional

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_agent import LLMCapacityAgent
from inference_engine import InferenceEngine
from training_engine import TrainingEngine
from config_loader import (
    ConfigLoader,
    ConfigValidationError,
    save_config_template,
)
from models import (
    LLMWorkloadInput,
    TrainingInput,
    ModelConfig,
    GPUConfig,
    ServiceInput,
)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with all options."""
    parser = argparse.ArgumentParser(
        description="LLM Infrastructure Capacity Forecasting Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python commands/forecast.py

  # JSON config mode
  python commands/forecast.py --config inference_config.json

  # Generate config template
  python commands/forecast.py --generate-config inference

  # CLI mode - Inference
  python commands/forecast.py --mode inference --rps 10 --input-tokens 500 \\
      --output-tokens 200 --model-size 70 --precision FP16

  # CLI mode - Training
  python commands/forecast.py --mode training --dataset-size 1000000000 \\
      --sequence-length 4096 --epochs 3 --global-batch-size 64 \\
      --model-size 70 --precision BF16
        """
    )
    
    # Input mode options
    parser.add_argument(
        '--config',
        type=str,
        metavar='FILE',
        help='Path to JSON configuration file'
    )
    parser.add_argument(
        '--generate-config',
        type=str,
        choices=['inference', 'training'],
        metavar='MODE',
        help='Generate template config file (inference or training)'
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        choices=['inference', 'training'],
        help='Workload type: inference or training'
    )
    
    # Inference workload options
    inference_group = parser.add_argument_group('Inference Options')
    inference_group.add_argument(
        '--rps',
        type=float,
        metavar='N',
        help='Requests per second'
    )
    inference_group.add_argument(
        '--input-tokens',
        type=int,
        metavar='N',
        help='Average input tokens per request'
    )
    inference_group.add_argument(
        '--output-tokens',
        type=int,
        metavar='N',
        help='Average output tokens per request'
    )
    inference_group.add_argument(
        '--peak-multiplier',
        type=float,
        metavar='N',
        default=1.5,
        help='Peak load multiplier (default: 1.5)'
    )
    inference_group.add_argument(
        '--growth-rate',
        type=float,
        metavar='N',
        help='Expected growth rate percentage'
    )
    
    # Training options
    training_group = parser.add_argument_group('Training Options')
    training_group.add_argument(
        '--dataset-size',
        type=int,
        metavar='N',
        help='Dataset size in tokens'
    )
    training_group.add_argument(
        '--sequence-length',
        type=int,
        metavar='N',
        default=4096,
        help='Sequence length (default: 4096)'
    )
    training_group.add_argument(
        '--epochs',
        type=int,
        metavar='N',
        help='Number of training epochs'
    )
    training_group.add_argument(
        '--global-batch-size',
        type=int,
        metavar='N',
        help='Global batch size'
    )
    training_group.add_argument(
        '--micro-batch-size',
        type=int,
        metavar='N',
        default=1,
        help='Micro batch size (default: 1)'
    )
    training_group.add_argument(
        '--optimizer',
        type=str,
        choices=['Adam', 'AdamW', 'Adafactor', 'SGD'],
        default='Adam',
        help='Optimizer type (default: Adam)'
    )
    training_group.add_argument(
        '--gradient-checkpointing',
        action='store_true',
        help='Enable gradient checkpointing'
    )
    
    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        '--model-size',
        type=float,
        metavar='N',
        help='Model size in billions (e.g., 7, 70, 405)'
    )
    model_group.add_argument(
        '--precision',
        type=str,
        choices=['FP32', 'FP16', 'BF16', 'INT8', 'INT4'],
        help='Model precision/quantization'
    )
    model_group.add_argument(
        '--context-window',
        type=int,
        metavar='N',
        default=4096,
        help='Context window size (default: 4096)'
    )
    model_group.add_argument(
        '--batch-size',
        type=int,
        metavar='N',
        default=1,
        help='Batch size for inference (default: 1)'
    )
    
    # GPU configuration
    gpu_group = parser.add_argument_group('GPU Configuration')
    gpu_group.add_argument(
        '--gpu-type',
        type=str,
        choices=['A100-40GB', 'A100-80GB', 'H100-80GB', 'L4', 'T4', 'V100'],
        help='GPU type (default: auto-select)'
    )
    gpu_group.add_argument(
        '--gpu-utilization',
        type=float,
        metavar='N',
        default=0.7,
        help='Target GPU utilization 0-1 (default: 0.7)'
    )
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--cost',
        action='store_true',
        help='Include cost estimates'
    )
    output_group.add_argument(
        '--output',
        type=str,
        metavar='FILE',
        help='Output JSON file path'
    )
    output_group.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed output, only show summary'
    )
    
    return parser


def determine_input_mode(args) -> str:
    """Determine which input mode to use based on arguments."""
    # Generate config mode
    if args.generate_config:
        return "generate"
    
    # JSON config mode
    if args.config:
        return "json"
    
    # CLI mode - check if required args for either mode are provided
    if args.mode == "inference":
        if args.rps and args.input_tokens and args.output_tokens and args.model_size and args.precision:
            return "cli"
    elif args.mode == "training":
        if args.dataset_size and args.epochs and args.global_batch_size and args.model_size and args.precision:
            return "cli"
    
    # Check if any CLI args suggest non-interactive mode
    if args.mode or args.rps or args.dataset_size:
        return "cli_incomplete"
    
    # Default to interactive
    return "interactive"


def run_interactive_mode(args):
    """Run in interactive mode."""
    agent = LLMCapacityAgent()
    plan = agent.run(include_cost=args.cost)
    return plan


def run_json_mode(args):
    """Run with JSON configuration file."""
    loader = ConfigLoader()
    
    try:
        # Load config
        config = loader.load_config(args.config)
        
        # Validate
        is_valid, errors = loader.validate_config(config)
        if not is_valid:
            print("❌ Configuration validation failed:")
            for error in errors:
                print(f"   - {error}")
            return None
        
        # Merge with CLI overrides
        config = loader.merge_with_cli_args(config, args)
        
        # Convert to models
        mode = config.get("mode", "inference")
        workload_or_training, model_config, gpu_config, service, options = loader.config_to_models(config)
        
        # Get options
        include_cost = options.get("include_cost", args.cost)
        output_file = options.get("output_file", args.output)
        
        # Generate plan
        print("⏳ Calculating capacity requirements...")
        
        agent = LLMCapacityAgent()
        
        if mode == "inference":
            engine = InferenceEngine()
            plan = engine.generate_inference_plan(
                workload_or_training, model_config, gpu_config, service, include_cost
            )
            print(agent.format_inference_plan(plan))
        else:
            engine = TrainingEngine()
            plan = engine.generate_training_plan(
                workload_or_training, model_config, gpu_config, service, include_cost
            )
            print(agent.format_training_plan(plan))
        
        # Export if output specified
        if output_file:
            print(agent.export_plan(plan, output_file))
        
        return plan
        
    except ConfigValidationError as e:
        print(f"❌ {e}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def run_cli_mode(args):
    """Run with CLI arguments."""
    # Validate required args based on mode
    if args.mode == "inference":
        if not all([args.rps, args.input_tokens, args.output_tokens, args.model_size, args.precision]):
            print("❌ Missing required arguments for inference mode.")
            print("   Required: --rps, --input-tokens, --output-tokens, --model-size, --precision")
            return None
        
        workload = LLMWorkloadInput(
            requests_per_second=args.rps,
            avg_input_tokens=args.input_tokens,
            avg_output_tokens=args.output_tokens,
            peak_load_multiplier=args.peak_multiplier,
            growth_rate=args.growth_rate,
        )
        
    elif args.mode == "training":
        if not all([args.dataset_size, args.epochs, args.global_batch_size, args.model_size, args.precision]):
            print("❌ Missing required arguments for training mode.")
            print("   Required: --dataset-size, --epochs, --global-batch-size, --model-size, --precision")
            return None
        
        workload = TrainingInput(
            dataset_size_tokens=args.dataset_size,
            sequence_length=args.sequence_length,
            num_epochs=args.epochs,
            global_batch_size=args.global_batch_size,
            micro_batch_size=args.micro_batch_size,
            optimizer_type=args.optimizer,
            gradient_checkpointing=args.gradient_checkpointing,
            training_precision=args.precision,
        )
    else:
        print("❌ Please specify --mode (inference or training)")
        return None
    
    # Model config
    model_config = ModelConfig(
        model_size_billions=args.model_size,
        precision=args.precision,
        context_window=args.context_window,
        batch_size=args.batch_size,
    )
    
    # GPU config
    gpu_config = GPUConfig(
        gpu_type=args.gpu_type,
        target_gpu_utilization=args.gpu_utilization,
    )
    
    # Generate plan
    if not args.quiet:
        print("⏳ Calculating capacity requirements...")
    
    agent = LLMCapacityAgent()
    
    if args.mode == "inference":
        engine = InferenceEngine()
        plan = engine.generate_inference_plan(
            workload, model_config, gpu_config, include_cost=args.cost
        )
        if not args.quiet:
            print(agent.format_inference_plan(plan))
    else:
        engine = TrainingEngine()
        plan = engine.generate_training_plan(
            workload, model_config, gpu_config, include_cost=args.cost
        )
        if not args.quiet:
            print(agent.format_training_plan(plan))
    
    # Export if output specified
    if args.output:
        print(agent.export_plan(plan, args.output))
    elif not args.quiet:
        # Auto-export to default file
        default_file = "inference_plan.json" if args.mode == "inference" else "training_plan.json"
        print(agent.export_plan(plan, default_file))
    
    return plan


def run_generate_config(args):
    """Generate configuration template."""
    mode = args.generate_config
    try:
        filepath = save_config_template(mode)
        print(f"✅ Generated {mode} configuration template: {filepath}")
        print(f"   Edit this file and run: python commands/forecast.py --config {filepath}")
    except Exception as e:
        print(f"❌ Error generating template: {e}")


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Determine input mode
    input_mode = determine_input_mode(args)
    
    if input_mode == "generate":
        run_generate_config(args)
    elif input_mode == "json":
        run_json_mode(args)
    elif input_mode == "cli":
        run_cli_mode(args)
    elif input_mode == "cli_incomplete":
        print("❌ Incomplete CLI arguments provided.")
        print("   For inference mode, provide: --mode inference --rps --input-tokens --output-tokens --model-size --precision")
        print("   For training mode, provide: --mode training --dataset-size --epochs --global-batch-size --model-size --precision")
        print("\n   Or run without arguments for interactive mode.")
        parser.print_help()
    else:
        # Interactive mode
        run_interactive_mode(args)


if __name__ == "__main__":
    main()
