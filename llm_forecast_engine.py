"""
Unified LLM Forecast Engine.

Provides a single interface for both inference and training capacity forecasting,
with optional time-series based predictions.
"""

from typing import Optional, Union, List, Tuple
from datetime import datetime

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
from forecasting import ForecastingEngine, ForecastResult
from services import ServiceValidator


class LLMForecastEngine:
    """
    Unified LLM Forecast Engine.
    
    Combines inference and training forecasting with optional
    time-series based predictions.
    """
    
    def __init__(self):
        self.inference_engine = InferenceEngine()
        self.training_engine = TrainingEngine()
        self.forecasting_engine = ForecastingEngine(seasonal_period=7)
        self.validator = ServiceValidator()
    
    def generate_inference_plan(
        self,
        workload: LLMWorkloadInput,
        model_config: ModelConfig,
        gpu_config: Optional[GPUConfig] = None,
        service: Optional[ServiceInput] = None,
        include_cost: bool = False
    ) -> LLMCapacityPlan:
        """
        Generate inference capacity plan.
        
        Args:
            workload: Workload parameters (RPS, tokens, etc.)
            model_config: Model configuration
            gpu_config: Optional GPU preferences
            service: Optional service metadata
            include_cost: Whether to include cost estimates
            
        Returns:
            LLMCapacityPlan with complete recommendations
        """
        # Validate inputs
        is_valid, errors = self.validator.validate_inference_workload(workload)
        if not is_valid:
            raise ValueError(f"Invalid workload: {'; '.join(errors)}")
        
        is_valid, errors = self.validator.validate_model_config(model_config)
        if not is_valid:
            raise ValueError(f"Invalid model config: {'; '.join(errors)}")
        
        if gpu_config:
            is_valid, errors = self.validator.validate_gpu_config(gpu_config)
            if not is_valid:
                raise ValueError(f"Invalid GPU config: {'; '.join(errors)}")
        
        if service:
            is_valid, errors = self.validator.validate_service_input(service)
            if not is_valid:
                raise ValueError(f"Invalid service input: {'; '.join(errors)}")
        
        return self.inference_engine.generate_inference_plan(
            workload, model_config, gpu_config, service, include_cost
        )
    
    def generate_training_plan(
        self,
        training_input: TrainingInput,
        model_config: ModelConfig,
        gpu_config: Optional[GPUConfig] = None,
        service: Optional[ServiceInput] = None,
        include_cost: bool = False
    ) -> TrainingCapacityPlan:
        """
        Generate training capacity plan.
        
        Args:
            training_input: Training parameters
            model_config: Model configuration
            gpu_config: Optional GPU preferences
            service: Optional service metadata
            include_cost: Whether to include cost estimates
            
        Returns:
            TrainingCapacityPlan with complete recommendations
        """
        # Validate inputs
        is_valid, errors = self.validator.validate_training_input(training_input)
        if not is_valid:
            raise ValueError(f"Invalid training input: {'; '.join(errors)}")
        
        is_valid, errors = self.validator.validate_model_config(model_config)
        if not is_valid:
            raise ValueError(f"Invalid model config: {'; '.join(errors)}")
        
        if gpu_config:
            is_valid, errors = self.validator.validate_gpu_config(gpu_config)
            if not is_valid:
                raise ValueError(f"Invalid GPU config: {'; '.join(errors)}")
        
        if service:
            is_valid, errors = self.validator.validate_service_input(service)
            if not is_valid:
                raise ValueError(f"Invalid service input: {'; '.join(errors)}")
        
        return self.training_engine.generate_training_plan(
            training_input, model_config, gpu_config, service, include_cost
        )
    
    def forecast_future_needs(
        self,
        rps_history: List[Tuple[datetime, float]],
        model_config: ModelConfig,
        gpu_config: Optional[GPUConfig] = None,
        horizon_days: int = 30,
        scenario: str = "baseline",
        include_cost: bool = True
    ) -> dict:
        """
        Forecast future capacity needs based on historical RPS data.
        
        Args:
            rps_history: List of (timestamp, rps) tuples
            model_config: Model configuration
            gpu_config: Optional GPU preferences
            horizon_days: Number of days to forecast
            scenario: Scenario type (baseline, optimistic, pessimistic, spike)
            include_cost: Whether to include cost estimates
            
        Returns:
            Dictionary with current and forecasted capacity plans
        """
        if len(rps_history) < 6:
            raise ValueError("Insufficient historical data. Need at least 6 data points.")
        
        # Forecast future RPS
        timestamps = [t for t, _ in rps_history]
        values = [v for _, v in rps_history]
        
        rps_forecast = self.forecasting_engine.forecast_metric(
            timestamps, values, "RPS", horizon_days, scenario
        )
        
        # Get current capacity plan based on current RPS
        current_rps = values[-1]
        avg_input_tokens = 500  # Default assumption
        avg_output_tokens = 200  # Default assumption
        
        current_workload = LLMWorkloadInput(
            requests_per_second=current_rps,
            avg_input_tokens=avg_input_tokens,
            avg_output_tokens=avg_output_tokens,
        )
        
        current_plan = self.generate_inference_plan(
            current_workload, model_config, gpu_config, include_cost=include_cost
        )
        
        # Get forecasted capacity plan based on max forecasted RPS (P95)
        max_forecast_rps = max(rps_forecast.upper_95) if rps_forecast.upper_95 else current_rps
        avg_forecast_rps = sum(rps_forecast.forecast) / len(rps_forecast.forecast) if rps_forecast.forecast else current_rps
        
        forecast_workload = LLMWorkloadInput(
            requests_per_second=max_forecast_rps,
            avg_input_tokens=avg_input_tokens,
            avg_output_tokens=avg_output_tokens,
        )
        
        forecast_plan = self.generate_inference_plan(
            forecast_workload, model_config, gpu_config, include_cost=include_cost
        )
        
        # Calculate scaling recommendations
        current_gpus = current_plan.gpu_resources.total_gpus
        forecast_gpus = forecast_plan.gpu_resources.total_gpus
        gpu_change = forecast_gpus - current_gpus
        gpu_change_pct = (gpu_change / current_gpus * 100) if current_gpus > 0 else 0
        
        scaling_recommendations = []
        if gpu_change > 0:
            scaling_recommendations.append(
                f"📈 Scale up from {current_gpus} to {forecast_gpus} GPUs "
                f"(+{gpu_change_pct:.1f}%) to handle forecasted demand"
            )
        elif gpu_change < 0:
            scaling_recommendations.append(
                f"📉 Potential to scale down from {current_gpus} to {forecast_gpus} GPUs "
                f"({gpu_change_pct:.1f}%) based on forecasted demand"
            )
        else:
            scaling_recommendations.append(
                "➡️ Current capacity is sufficient for forecasted demand"
            )
        
        # Add scenario-specific recommendations
        if scenario == "pessimistic":
            scaling_recommendations.append(
                "⚠️ Pessimistic scenario: Consider additional headroom for unexpected spikes"
            )
        elif scenario == "spike":
            scaling_recommendations.append(
                "🚨 Spike scenario: Ensure auto-scaling policies can handle rapid scale-up"
            )
        
        return {
            "current_plan": current_plan,
            "forecast_plan": forecast_plan,
            "rps_forecast": {
                "horizon_days": horizon_days,
                "scenario": scenario,
                "current_rps": current_rps,
                "avg_forecast_rps": avg_forecast_rps,
                "max_forecast_rps": max_forecast_rps,
                "forecast_values": rps_forecast.forecast,
                "timestamps": [t.isoformat() for t in rps_forecast.timestamps],
                "confidence_intervals": {
                    "lower_80": rps_forecast.lower_80,
                    "upper_80": rps_forecast.upper_80,
                    "lower_95": rps_forecast.lower_95,
                    "upper_95": rps_forecast.upper_95,
                },
                "explanations": rps_forecast.explanations,
            },
            "scaling_recommendations": scaling_recommendations,
            "gpu_change": {
                "current_gpus": current_gpus,
                "forecast_gpus": forecast_gpus,
                "delta": gpu_change,
                "delta_pct": gpu_change_pct,
            },
        }
    
    def compare_gpu_options(
        self,
        workload: LLMWorkloadInput,
        model_config: ModelConfig,
        gpu_types: List[str] = None,
        include_cost: bool = True
    ) -> List[dict]:
        """
        Compare different GPU options for a workload.
        
        Args:
            workload: Workload parameters
            model_config: Model configuration
            gpu_types: List of GPU types to compare (default: common types)
            include_cost: Whether to include cost estimates
            
        Returns:
            List of comparison results sorted by total cost
        """
        if gpu_types is None:
            gpu_types = ["A100-40GB", "A100-80GB", "H100-80GB"]
        
        results = []
        
        for gpu_type in gpu_types:
            try:
                gpu_config = GPUConfig(gpu_type=gpu_type)
                plan = self.generate_inference_plan(
                    workload, model_config, gpu_config, include_cost=include_cost
                )
                
                results.append({
                    "gpu_type": gpu_type,
                    "total_gpus": plan.gpu_resources.total_gpus,
                    "replicas": plan.gpu_resources.min_replicas,
                    "gpus_per_replica": plan.gpu_resources.min_gpus_per_replica,
                    "gpu_memory_per_replica_gb": plan.gpu_resources.gpu_memory_per_replica_gb,
                    "tps_capacity": plan.throughput.total_tps_capacity,
                    "meets_latency_sla": plan.throughput.meets_latency_sla,
                    "monthly_gpu_cost": plan.cost.monthly_gpu_cost if plan.cost else None,
                    "cost_per_million_tokens": plan.cost.cost_per_million_tokens if plan.cost else None,
                })
            except Exception as e:
                results.append({
                    "gpu_type": gpu_type,
                    "error": str(e),
                })
        
        # Sort by monthly cost (cheapest first)
        results.sort(key=lambda x: x.get("monthly_gpu_cost", float("inf")) or float("inf"))
        
        return results
    
    def estimate_training_time_and_cost(
        self,
        training_input: TrainingInput,
        model_config: ModelConfig,
        gpu_types: List[str] = None,
    ) -> List[dict]:
        """
        Estimate training time and cost for different GPU options.
        
        Args:
            training_input: Training parameters
            model_config: Model configuration
            gpu_types: List of GPU types to compare
            
        Returns:
            List of training estimates sorted by total cost
        """
        if gpu_types is None:
            gpu_types = ["A100-80GB", "H100-80GB"]
        
        results = []
        
        for gpu_type in gpu_types:
            try:
                gpu_config = GPUConfig(gpu_type=gpu_type, target_gpu_utilization=0.85)
                plan = self.generate_training_plan(
                    training_input, model_config, gpu_config, include_cost=True
                )
                
                results.append({
                    "gpu_type": gpu_type,
                    "required_gpus": plan.gpu_resources.required_gpus,
                    "num_nodes": plan.gpu_resources.num_nodes,
                    "gpu_memory_breakdown": {
                        "model_weights_gb": plan.gpu_resources.model_weights_gb,
                        "gradients_gb": plan.gpu_resources.gradients_gb,
                        "optimizer_states_gb": plan.gpu_resources.optimizer_states_gb,
                        "activations_gb": plan.gpu_resources.activations_gb,
                    },
                    "training_hours": plan.training_metrics.estimated_duration_hours,
                    "training_days": plan.training_metrics.estimated_duration_days,
                    "tokens_per_second": plan.training_metrics.total_tokens_per_second,
                    "total_training_cost": plan.cost.total_training_cost if plan.cost else None,
                    "cost_per_epoch": plan.cost.cost_per_epoch if plan.cost else None,
                })
            except Exception as e:
                results.append({
                    "gpu_type": gpu_type,
                    "error": str(e),
                })
        
        # Sort by total cost (cheapest first)
        results.sort(key=lambda x: x.get("total_training_cost", float("inf")) or float("inf"))
        
        return results
