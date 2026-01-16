"""
JSON Configuration Loader for LLM Capacity Forecasting.

Handles loading, validation, and merging of JSON configuration files.
"""

import json
import os
from typing import Optional, Any
from dataclasses import fields

from models import (
    LLMWorkloadInput,
    TrainingInput,
    ModelConfig,
    GPUConfig,
    ServiceInput,
)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigLoader:
    """Loads and validates JSON configuration files."""
    
    # Required fields for inference mode
    INFERENCE_REQUIRED = {
        "workload": ["requests_per_second", "avg_input_tokens", "avg_output_tokens"],
        "model": ["model_size_billions", "precision"],
    }
    
    # Required fields for training mode
    TRAINING_REQUIRED = {
        "training": ["dataset_size_tokens", "sequence_length", "num_epochs", "global_batch_size"],
        "model": ["model_size_billions", "precision"],
    }
    
    # Valid enum values
    VALID_PRECISION = ["FP32", "FP16", "BF16", "INT8", "INT4", "FP8"]
    VALID_OPTIMIZER = ["Adam", "AdamW", "Adafactor", "SGD"]
    VALID_ENVIRONMENT = ["prod", "staging", "dev"]
    VALID_CRITICALITY = ["high", "medium", "low"]
    VALID_CLOUD_PROVIDER = ["aws", "gcp", "azure"]
    VALID_GPU_TYPES = ["A100-40GB", "A100-80GB", "H100-80GB", "H100-SXM", "L4", "T4", "V100"]
    
    def load_config(self, filepath: str) -> dict:
        """Load configuration from JSON file."""
        if not os.path.exists(filepath):
            raise ConfigValidationError(f"Config file not found: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigValidationError(f"Invalid JSON format: {e}")
        
        return config
    
    def validate_config(self, config: dict) -> tuple[bool, list[str]]:
        """
        Validate configuration structure and values.
        
        Returns: (is_valid, list of error messages)
        """
        errors = []
        
        # Check mode
        mode = config.get("mode", "inference")
        if mode not in ["inference", "training"]:
            errors.append(f"Invalid mode: {mode}. Must be 'inference' or 'training'.")
            return False, errors
        
        # Check required fields based on mode
        required = self.INFERENCE_REQUIRED if mode == "inference" else self.TRAINING_REQUIRED
        
        for section, fields_list in required.items():
            if section not in config:
                errors.append(f"Missing required section: '{section}'")
                continue
            
            for field in fields_list:
                if field not in config[section]:
                    errors.append(f"Missing required field: '{section}.{field}'")
        
        # Validate enum values
        if "model" in config:
            precision = config["model"].get("precision")
            if precision and precision not in self.VALID_PRECISION:
                errors.append(
                    f"Invalid precision: {precision}. "
                    f"Valid values: {', '.join(self.VALID_PRECISION)}"
                )
        
        if "training" in config:
            optimizer = config["training"].get("optimizer_type")
            if optimizer and optimizer not in self.VALID_OPTIMIZER:
                errors.append(
                    f"Invalid optimizer: {optimizer}. "
                    f"Valid values: {', '.join(self.VALID_OPTIMIZER)}"
                )
        
        if "gpu" in config:
            gpu_type = config["gpu"].get("gpu_type")
            if gpu_type and gpu_type not in self.VALID_GPU_TYPES:
                errors.append(
                    f"Invalid GPU type: {gpu_type}. "
                    f"Valid values: {', '.join(self.VALID_GPU_TYPES)}"
                )
        
        if "service" in config:
            env = config["service"].get("environment")
            if env and env not in self.VALID_ENVIRONMENT:
                errors.append(
                    f"Invalid environment: {env}. "
                    f"Valid values: {', '.join(self.VALID_ENVIRONMENT)}"
                )
            
            criticality = config["service"].get("criticality")
            if criticality and criticality not in self.VALID_CRITICALITY:
                errors.append(
                    f"Invalid criticality: {criticality}. "
                    f"Valid values: {', '.join(self.VALID_CRITICALITY)}"
                )
            
            cloud = config["service"].get("cloud_provider")
            if cloud and cloud not in self.VALID_CLOUD_PROVIDER:
                errors.append(
                    f"Invalid cloud provider: {cloud}. "
                    f"Valid values: {', '.join(self.VALID_CLOUD_PROVIDER)}"
                )
        
        # Validate numeric ranges
        if "workload" in config:
            rps = config["workload"].get("requests_per_second")
            if rps is not None and rps <= 0:
                errors.append("requests_per_second must be greater than 0")
            
            input_tokens = config["workload"].get("avg_input_tokens")
            if input_tokens is not None and input_tokens <= 0:
                errors.append("avg_input_tokens must be greater than 0")
            
            output_tokens = config["workload"].get("avg_output_tokens")
            if output_tokens is not None and output_tokens <= 0:
                errors.append("avg_output_tokens must be greater than 0")
        
        if "model" in config:
            model_size = config["model"].get("model_size_billions")
            if model_size is not None and model_size <= 0:
                errors.append("model_size_billions must be greater than 0")
        
        if "training" in config:
            dataset_size = config["training"].get("dataset_size_tokens")
            if dataset_size is not None and dataset_size <= 0:
                errors.append("dataset_size_tokens must be greater than 0")
            
            epochs = config["training"].get("num_epochs")
            if epochs is not None and epochs <= 0:
                errors.append("num_epochs must be greater than 0")
        
        return len(errors) == 0, errors
    
    def merge_with_cli_args(self, config: dict, args: Any) -> dict:
        """Merge JSON config with CLI arguments (CLI takes precedence)."""
        merged = config.copy()
        
        # Ensure nested dicts exist
        for section in ["workload", "model", "gpu", "service", "training", "options"]:
            if section not in merged:
                merged[section] = {}
        
        # Map CLI args to config sections
        cli_mappings = {
            # Workload mappings
            "rps": ("workload", "requests_per_second"),
            "input_tokens": ("workload", "avg_input_tokens"),
            "output_tokens": ("workload", "avg_output_tokens"),
            "peak_multiplier": ("workload", "peak_load_multiplier"),
            "growth_rate": ("workload", "growth_rate"),
            
            # Model mappings
            "model_size": ("model", "model_size_billions"),
            "precision": ("model", "precision"),
            "context_window": ("model", "context_window"),
            "batch_size": ("model", "batch_size"),
            
            # GPU mappings
            "gpu_type": ("gpu", "gpu_type"),
            "gpu_utilization": ("gpu", "target_gpu_utilization"),
            
            # Training mappings
            "dataset_size": ("training", "dataset_size_tokens"),
            "sequence_length": ("training", "sequence_length"),
            "epochs": ("training", "num_epochs"),
            "global_batch_size": ("training", "global_batch_size"),
            "micro_batch_size": ("training", "micro_batch_size"),
            "optimizer": ("training", "optimizer_type"),
            "gradient_checkpointing": ("training", "gradient_checkpointing"),
            
            # Options mappings
            "cost": ("options", "include_cost"),
            "output": ("options", "output_file"),
        }
        
        # Apply CLI overrides
        for cli_arg, (section, field) in cli_mappings.items():
            value = getattr(args, cli_arg, None)
            if value is not None:
                merged[section][field] = value
        
        # Handle mode override
        if hasattr(args, "mode") and args.mode:
            merged["mode"] = args.mode
        
        return merged
    
    def config_to_models(self, config: dict) -> tuple:
        """
        Convert config dict to model objects.
        
        Returns different tuples based on mode:
        - Inference: (LLMWorkloadInput, ModelConfig, GPUConfig, ServiceInput, options)
        - Training: (TrainingInput, ModelConfig, GPUConfig, ServiceInput, options)
        """
        mode = config.get("mode", "inference")
        
        # Model config
        model_cfg = config.get("model", {})
        model_config = ModelConfig(
            model_size_billions=model_cfg.get("model_size_billions", 7),
            precision=model_cfg.get("precision", "FP16"),
            context_window=model_cfg.get("context_window", 4096),
            batch_size=model_cfg.get("batch_size", 1),
            target_latency_p50_ms=model_cfg.get("target_latency_p50_ms"),
            target_latency_p99_ms=model_cfg.get("target_latency_p99_ms"),
        )
        
        # GPU config
        gpu_cfg = config.get("gpu", {})
        gpu_config = GPUConfig(
            gpu_type=gpu_cfg.get("gpu_type"),
            target_gpu_utilization=gpu_cfg.get("target_gpu_utilization", 0.7),
            current_replicas=gpu_cfg.get("current_replicas"),
        ) if gpu_cfg else None
        
        # Service config
        svc_cfg = config.get("service", {})
        service_input = ServiceInput(
            service_id=svc_cfg.get("service_id"),
            service_name=svc_cfg.get("service_name"),
            team_id=svc_cfg.get("team_id"),
            environment=svc_cfg.get("environment", "prod"),
            criticality=svc_cfg.get("criticality", "medium"),
            cloud_provider=svc_cfg.get("cloud_provider"),
            region=svc_cfg.get("region"),
        ) if svc_cfg else None
        
        # Options
        options = config.get("options", {})
        
        if mode == "inference":
            workload_cfg = config.get("workload", {})
            workload_input = LLMWorkloadInput(
                requests_per_second=workload_cfg.get("requests_per_second", 1.0),
                avg_input_tokens=workload_cfg.get("avg_input_tokens", 512),
                avg_output_tokens=workload_cfg.get("avg_output_tokens", 256),
                peak_load_multiplier=workload_cfg.get("peak_load_multiplier", 1.5),
                growth_rate=workload_cfg.get("growth_rate"),
            )
            return workload_input, model_config, gpu_config, service_input, options
        
        else:  # training
            training_cfg = config.get("training", {})
            training_input = TrainingInput(
                dataset_size_tokens=training_cfg.get("dataset_size_tokens", 1_000_000_000),
                sequence_length=training_cfg.get("sequence_length", 4096),
                num_epochs=training_cfg.get("num_epochs", 1),
                global_batch_size=training_cfg.get("global_batch_size", 64),
                micro_batch_size=training_cfg.get("micro_batch_size", 1),
                gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
                optimizer_type=training_cfg.get("optimizer_type", "Adam"),
                gradient_checkpointing=training_cfg.get("gradient_checkpointing", False),
                training_precision=training_cfg.get("training_precision", "FP16"),
            )
            return training_input, model_config, gpu_config, service_input, options


def generate_inference_template() -> dict:
    """Generate a template inference configuration."""
    return {
        "mode": "inference",
        "workload": {
            "requests_per_second": 10.0,
            "avg_input_tokens": 500,
            "avg_output_tokens": 200,
            "peak_load_multiplier": 1.5,
            "growth_rate": 20.0
        },
        "model": {
            "model_size_billions": 70,
            "precision": "FP16",
            "context_window": 8192,
            "batch_size": 4,
            "target_latency_p50_ms": 100,
            "target_latency_p99_ms": 500
        },
        "gpu": {
            "gpu_type": "A100-80GB",
            "target_gpu_utilization": 0.7,
            "current_replicas": None
        },
        "service": {
            "service_id": "my-llm-service",
            "service_name": "LLM Inference Service",
            "team_id": "ml-team",
            "environment": "prod",
            "criticality": "high",
            "cloud_provider": "aws",
            "region": "us-east-1"
        },
        "options": {
            "include_cost": True,
            "output_file": "inference_plan.json"
        }
    }


def generate_training_template() -> dict:
    """Generate a template training configuration."""
    return {
        "mode": "training",
        "training": {
            "dataset_size_tokens": 1000000000,
            "sequence_length": 4096,
            "num_epochs": 3,
            "global_batch_size": 64,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "optimizer_type": "AdamW",
            "gradient_checkpointing": True,
            "training_precision": "BF16"
        },
        "model": {
            "model_size_billions": 70,
            "precision": "BF16"
        },
        "gpu": {
            "gpu_type": "H100-80GB",
            "target_gpu_utilization": 0.85
        },
        "service": {
            "service_id": "finetuning-job-001",
            "service_name": "LLM Fine-Tuning Job",
            "team_id": "ml-team",
            "environment": "prod",
            "cloud_provider": "gcp",
            "region": "us-central1"
        },
        "options": {
            "include_cost": True,
            "output_file": "training_plan.json"
        }
    }


def save_config_template(mode: str, filepath: Optional[str] = None) -> str:
    """Save a configuration template to file."""
    if mode == "inference":
        template = generate_inference_template()
        default_filename = "inference_config.json"
    elif mode == "training":
        template = generate_training_template()
        default_filename = "training_config.json"
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'inference' or 'training'.")
    
    output_path = filepath or default_filename
    
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    return output_path
