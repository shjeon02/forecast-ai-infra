"""
Time-Series Forecasting Engine.

Provides STL decomposition, ARIMA, ETS models, and ensemble forecasting
for predicting future resource requirements based on historical data.
"""

import math
from typing import Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class TimeSeriesData:
    """Container for time series data."""
    timestamps: List[datetime]
    values: List[float]
    metric_name: str = "metric"
    unit: str = "value"
    
    def __len__(self):
        return len(self.values)
    
    @property
    def mean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0
    
    @property
    def std(self) -> float:
        if len(self.values) < 2:
            return 0
        mean = self.mean
        variance = sum((x - mean) ** 2 for x in self.values) / len(self.values)
        return math.sqrt(variance)


@dataclass
class STLComponents:
    """STL decomposition components."""
    trend: List[float]
    seasonal: List[float]
    residual: List[float]
    period: int


@dataclass
class ForecastResult:
    """Result of time series forecasting."""
    forecast: List[float]
    timestamps: List[datetime]
    lower_80: List[float]  # P80 lower bound
    upper_80: List[float]  # P80 upper bound
    lower_95: List[float]  # P95 lower bound
    upper_95: List[float]  # P95 upper bound
    components: Optional[STLComponents] = None
    explanations: List[str] = None
    scenario: str = "baseline"
    
    def __post_init__(self):
        if self.explanations is None:
            self.explanations = []


class STLDecomposition:
    """
    Seasonal-Trend decomposition using Loess (STL).
    
    Simplified implementation without external dependencies.
    """
    
    def __init__(self, period: int = 7):
        """
        Initialize STL decomposition.
        
        Args:
            period: Seasonal period (e.g., 7 for weekly, 30 for monthly)
        """
        self.period = period
    
    def _moving_average(self, data: List[float], window: int) -> List[float]:
        """Calculate moving average."""
        result = []
        half_window = window // 2
        
        for i in range(len(data)):
            start = max(0, i - half_window)
            end = min(len(data), i + half_window + 1)
            window_data = data[start:end]
            result.append(sum(window_data) / len(window_data))
        
        return result
    
    def _extract_seasonal(self, detrended: List[float]) -> List[float]:
        """Extract seasonal component."""
        seasonal = [0.0] * len(detrended)
        
        # Calculate average for each position in the period
        for i in range(self.period):
            positions = list(range(i, len(detrended), self.period))
            if positions:
                avg = sum(detrended[p] for p in positions) / len(positions)
                for p in positions:
                    seasonal[p] = avg
        
        # Normalize seasonal component to sum to zero within each period
        seasonal_mean = sum(seasonal[:self.period]) / self.period if self.period > 0 else 0
        seasonal = [s - seasonal_mean for s in seasonal]
        
        return seasonal
    
    def decompose(self, data: TimeSeriesData) -> STLComponents:
        """
        Perform STL decomposition.
        
        Args:
            data: Time series data
            
        Returns:
            STLComponents with trend, seasonal, and residual
        """
        values = data.values
        n = len(values)
        
        if n < self.period * 2:
            # Not enough data for seasonal decomposition
            # Return trend as moving average, no seasonal
            trend = self._moving_average(values, min(n, 3))
            seasonal = [0.0] * n
            residual = [v - t for v, t in zip(values, trend)]
            return STLComponents(trend=trend, seasonal=seasonal, residual=residual, period=self.period)
        
        # Step 1: Extract trend using moving average
        trend = self._moving_average(values, self.period)
        
        # Step 2: Detrend
        detrended = [v - t for v, t in zip(values, trend)]
        
        # Step 3: Extract seasonal component
        seasonal = self._extract_seasonal(detrended)
        
        # Step 4: Calculate residual
        residual = [v - t - s for v, t, s in zip(values, trend, seasonal)]
        
        return STLComponents(
            trend=trend,
            seasonal=seasonal,
            residual=residual,
            period=self.period
        )


class ARIMAModel:
    """
    Simplified ARIMA model implementation.
    
    Uses autoregressive (AR) component for forecasting.
    """
    
    def __init__(self, p: int = 2, d: int = 1, q: int = 0):
        """
        Initialize ARIMA model.
        
        Args:
            p: AR order
            d: Differencing order
            q: MA order (not implemented in this simplified version)
        """
        self.p = p
        self.d = d
        self.q = q
        self.coefficients = []
        self.constant = 0.0
        self.residual_std = 0.0
    
    def _difference(self, data: List[float], order: int = 1) -> List[float]:
        """Apply differencing."""
        result = data.copy()
        for _ in range(order):
            result = [result[i] - result[i-1] for i in range(1, len(result))]
        return result
    
    def _undifference(self, diff_forecast: List[float], last_values: List[float]) -> List[float]:
        """Reverse differencing."""
        result = []
        last = last_values[-1]
        for val in diff_forecast:
            last = last + val
            result.append(last)
        return result
    
    def fit(self, data: TimeSeriesData):
        """
        Fit ARIMA model to data.
        
        Uses ordinary least squares for AR coefficient estimation.
        """
        values = data.values
        
        # Apply differencing
        if self.d > 0:
            diff_values = self._difference(values, self.d)
        else:
            diff_values = values.copy()
        
        if len(diff_values) <= self.p:
            # Not enough data
            self.coefficients = [1.0 / self.p] * self.p
            self.constant = sum(diff_values) / len(diff_values) if diff_values else 0
            return
        
        # Simple AR estimation using correlations
        n = len(diff_values)
        mean = sum(diff_values) / n
        
        # Estimate AR coefficients using autocorrelation
        self.coefficients = []
        for lag in range(1, self.p + 1):
            if n - lag > 0:
                numerator = sum((diff_values[i] - mean) * (diff_values[i - lag] - mean) 
                               for i in range(lag, n))
                denominator = sum((diff_values[i] - mean) ** 2 for i in range(n))
                if denominator > 0:
                    self.coefficients.append(numerator / denominator)
                else:
                    self.coefficients.append(0.0)
            else:
                self.coefficients.append(0.0)
        
        # Normalize coefficients
        coef_sum = sum(abs(c) for c in self.coefficients)
        if coef_sum > 1:
            self.coefficients = [c / coef_sum for c in self.coefficients]
        
        self.constant = mean * (1 - sum(self.coefficients))
        
        # Estimate residual standard deviation
        fitted = self._fit_values(diff_values)
        residuals = [a - f for a, f in zip(diff_values[self.p:], fitted)]
        if residuals:
            self.residual_std = math.sqrt(sum(r ** 2 for r in residuals) / len(residuals))
    
    def _fit_values(self, values: List[float]) -> List[float]:
        """Calculate fitted values."""
        fitted = []
        for i in range(self.p, len(values)):
            pred = self.constant
            for j, coef in enumerate(self.coefficients):
                pred += coef * values[i - j - 1]
            fitted.append(pred)
        return fitted
    
    def forecast(self, data: TimeSeriesData, horizon: int) -> Tuple[List[float], List[float], List[float]]:
        """
        Generate forecast.
        
        Returns: (forecast, lower_95, upper_95)
        """
        values = data.values.copy()
        
        # Apply differencing for forecasting
        if self.d > 0:
            diff_values = self._difference(values, self.d)
        else:
            diff_values = values.copy()
        
        # Generate forecasts
        diff_forecast = []
        history = diff_values[-self.p:] if len(diff_values) >= self.p else diff_values
        
        for h in range(horizon):
            pred = self.constant
            for j, coef in enumerate(self.coefficients):
                if j < len(history):
                    pred += coef * history[-(j + 1)]
            diff_forecast.append(pred)
            history = history[1:] + [pred] if len(history) >= self.p else history + [pred]
        
        # Undifference
        if self.d > 0:
            forecast = self._undifference(diff_forecast, values[-self.d:])
        else:
            forecast = diff_forecast
        
        # Calculate confidence intervals
        z_95 = 1.96
        z_80 = 1.28
        
        lower_95 = []
        upper_95 = []
        lower_80 = []
        upper_80 = []
        
        for h in range(horizon):
            # Uncertainty grows with horizon
            std_h = self.residual_std * math.sqrt(1 + h * 0.1)
            lower_95.append(forecast[h] - z_95 * std_h)
            upper_95.append(forecast[h] + z_95 * std_h)
            lower_80.append(forecast[h] - z_80 * std_h)
            upper_80.append(forecast[h] + z_80 * std_h)
        
        return forecast, lower_80, upper_80, lower_95, upper_95


class ETSModel:
    """
    Exponential Smoothing (ETS) model.
    
    Implements simple exponential smoothing with trend.
    """
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.1):
        """
        Initialize ETS model.
        
        Args:
            alpha: Level smoothing parameter
            beta: Trend smoothing parameter
        """
        self.alpha = alpha
        self.beta = beta
        self.level = 0.0
        self.trend = 0.0
        self.residual_std = 0.0
    
    def fit(self, data: TimeSeriesData):
        """Fit ETS model to data."""
        values = data.values
        n = len(values)
        
        if n < 2:
            self.level = values[0] if values else 0
            self.trend = 0
            return
        
        # Initialize level and trend
        self.level = values[0]
        self.trend = values[1] - values[0] if n > 1 else 0
        
        # Fit using exponential smoothing
        residuals = []
        for i in range(1, n):
            forecast = self.level + self.trend
            residuals.append(values[i] - forecast)
            
            # Update level and trend
            new_level = self.alpha * values[i] + (1 - self.alpha) * (self.level + self.trend)
            new_trend = self.beta * (new_level - self.level) + (1 - self.beta) * self.trend
            
            self.level = new_level
            self.trend = new_trend
        
        # Estimate residual std
        if residuals:
            self.residual_std = math.sqrt(sum(r ** 2 for r in residuals) / len(residuals))
    
    def forecast(self, horizon: int) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
        """
        Generate forecast.
        
        Returns: (forecast, lower_80, upper_80, lower_95, upper_95)
        """
        forecast = []
        for h in range(1, horizon + 1):
            forecast.append(self.level + h * self.trend)
        
        # Confidence intervals
        z_95 = 1.96
        z_80 = 1.28
        
        lower_95 = []
        upper_95 = []
        lower_80 = []
        upper_80 = []
        
        for h in range(horizon):
            std_h = self.residual_std * math.sqrt(1 + h * 0.15)
            lower_95.append(forecast[h] - z_95 * std_h)
            upper_95.append(forecast[h] + z_95 * std_h)
            lower_80.append(forecast[h] - z_80 * std_h)
            upper_80.append(forecast[h] + z_80 * std_h)
        
        return forecast, lower_80, upper_80, lower_95, upper_95


class EnsembleForecaster:
    """
    Ensemble forecaster combining multiple models.
    """
    
    def __init__(
        self,
        arima_weight: float = 0.5,
        ets_weight: float = 0.5,
        seasonal_period: int = 7
    ):
        """
        Initialize ensemble forecaster.
        
        Args:
            arima_weight: Weight for ARIMA model
            ets_weight: Weight for ETS model
            seasonal_period: Period for seasonal decomposition
        """
        self.arima_weight = arima_weight
        self.ets_weight = ets_weight
        self.seasonal_period = seasonal_period
        
        self.stl = STLDecomposition(period=seasonal_period)
        self.arima = ARIMAModel(p=2, d=1, q=0)
        self.ets = ETSModel(alpha=0.3, beta=0.1)
        self.components: Optional[STLComponents] = None
    
    def fit(self, data: TimeSeriesData):
        """Fit all models to data."""
        # STL decomposition
        self.components = self.stl.decompose(data)
        
        # Fit models on deseasonalized data
        deseasonalized_values = [
            v - s for v, s in zip(data.values, self.components.seasonal)
        ]
        deseasonalized = TimeSeriesData(
            timestamps=data.timestamps,
            values=deseasonalized_values,
            metric_name=data.metric_name,
            unit=data.unit
        )
        
        self.arima.fit(deseasonalized)
        self.ets.fit(deseasonalized)
    
    def forecast(
        self,
        data: TimeSeriesData,
        horizon: int,
        scenario: str = "baseline"
    ) -> ForecastResult:
        """
        Generate ensemble forecast.
        
        Args:
            data: Historical data
            horizon: Forecast horizon
            scenario: Scenario type (baseline, optimistic, pessimistic, spike)
            
        Returns:
            ForecastResult with forecasts and confidence intervals
        """
        # Fit models
        self.fit(data)
        
        # Get forecasts from each model
        arima_fc, arima_l80, arima_u80, arima_l95, arima_u95 = self.arima.forecast(data, horizon)
        ets_fc, ets_l80, ets_u80, ets_l95, ets_u95 = self.ets.forecast(horizon)
        
        # Ensemble combination
        forecast = []
        lower_80 = []
        upper_80 = []
        lower_95 = []
        upper_95 = []
        
        for h in range(horizon):
            fc = self.arima_weight * arima_fc[h] + self.ets_weight * ets_fc[h]
            l80 = self.arima_weight * arima_l80[h] + self.ets_weight * ets_l80[h]
            u80 = self.arima_weight * arima_u80[h] + self.ets_weight * ets_u80[h]
            l95 = self.arima_weight * arima_l95[h] + self.ets_weight * ets_l95[h]
            u95 = self.arima_weight * arima_u95[h] + self.ets_weight * ets_u95[h]
            
            forecast.append(fc)
            lower_80.append(l80)
            upper_80.append(u80)
            lower_95.append(l95)
            upper_95.append(u95)
        
        # Add seasonal component back
        if self.components:
            n_seasonal = len(self.components.seasonal)
            for h in range(horizon):
                seasonal_idx = h % n_seasonal
                forecast[h] += self.components.seasonal[seasonal_idx]
                lower_80[h] += self.components.seasonal[seasonal_idx]
                upper_80[h] += self.components.seasonal[seasonal_idx]
                lower_95[h] += self.components.seasonal[seasonal_idx]
                upper_95[h] += self.components.seasonal[seasonal_idx]
        
        # Apply scenario adjustments
        scenario_multiplier = self._get_scenario_multiplier(scenario)
        forecast = [f * scenario_multiplier for f in forecast]
        lower_80 = [l * scenario_multiplier for l in lower_80]
        upper_80 = [u * scenario_multiplier for u in upper_80]
        lower_95 = [l * scenario_multiplier for l in lower_95]
        upper_95 = [u * scenario_multiplier for u in upper_95]
        
        # Generate timestamps for forecast
        last_timestamp = data.timestamps[-1] if data.timestamps else datetime.now()
        forecast_timestamps = []
        for h in range(1, horizon + 1):
            forecast_timestamps.append(last_timestamp + timedelta(days=h))
        
        # Generate explanations
        explanations = self._generate_explanations(data, forecast, scenario)
        
        return ForecastResult(
            forecast=forecast,
            timestamps=forecast_timestamps,
            lower_80=lower_80,
            upper_80=upper_80,
            lower_95=lower_95,
            upper_95=upper_95,
            components=self.components,
            explanations=explanations,
            scenario=scenario
        )
    
    def _get_scenario_multiplier(self, scenario: str) -> float:
        """Get multiplier for scenario adjustment."""
        multipliers = {
            "baseline": 1.0,
            "optimistic": 0.85,
            "pessimistic": 1.15,
            "spike": 1.5,
        }
        return multipliers.get(scenario, 1.0)
    
    def _generate_explanations(
        self,
        data: TimeSeriesData,
        forecast: List[float],
        scenario: str
    ) -> List[str]:
        """Generate human-readable explanations for the forecast."""
        explanations = []
        
        # Trend analysis
        if self.components and len(self.components.trend) > 1:
            trend_start = self.components.trend[0]
            trend_end = self.components.trend[-1]
            trend_change = ((trend_end - trend_start) / trend_start * 100) if trend_start != 0 else 0
            
            if trend_change > 5:
                explanations.append(f"📈 Upward trend detected: {trend_change:.1f}% increase over the period")
            elif trend_change < -5:
                explanations.append(f"📉 Downward trend detected: {abs(trend_change):.1f}% decrease over the period")
            else:
                explanations.append("➡️ Relatively stable trend observed")
        
        # Seasonal pattern
        if self.components and max(self.components.seasonal) - min(self.components.seasonal) > data.std * 0.5:
            explanations.append(f"🔄 Seasonal pattern detected with period of {self.components.period} days")
        
        # Forecast summary
        if forecast:
            forecast_avg = sum(forecast) / len(forecast)
            current_avg = data.mean
            change_pct = ((forecast_avg - current_avg) / current_avg * 100) if current_avg != 0 else 0
            
            if change_pct > 10:
                explanations.append(f"⚠️ Forecast indicates {change_pct:.1f}% increase in resource demand")
            elif change_pct < -10:
                explanations.append(f"💡 Forecast indicates {abs(change_pct):.1f}% decrease in resource demand")
            else:
                explanations.append("📊 Resource demand expected to remain stable")
        
        # Scenario explanation
        if scenario != "baseline":
            scenario_desc = {
                "optimistic": "🌟 Optimistic scenario: 15% lower than baseline",
                "pessimistic": "⚠️ Pessimistic scenario: 15% higher than baseline",
                "spike": "🚨 Spike scenario: 50% higher than baseline for stress testing",
            }
            if scenario in scenario_desc:
                explanations.append(scenario_desc[scenario])
        
        return explanations


class ForecastingEngine:
    """
    Main forecasting engine for capacity planning.
    """
    
    def __init__(self, seasonal_period: int = 7):
        """
        Initialize forecasting engine.
        
        Args:
            seasonal_period: Period for seasonal patterns (7 for weekly)
        """
        self.seasonal_period = seasonal_period
        self.ensemble = EnsembleForecaster(
            arima_weight=0.5,
            ets_weight=0.5,
            seasonal_period=seasonal_period
        )
    
    def forecast_metric(
        self,
        timestamps: List[datetime],
        values: List[float],
        metric_name: str = "metric",
        horizon_days: int = 30,
        scenario: str = "baseline"
    ) -> ForecastResult:
        """
        Forecast a single metric.
        
        Args:
            timestamps: List of timestamps
            values: List of metric values
            metric_name: Name of the metric
            horizon_days: Number of days to forecast
            scenario: Scenario type
            
        Returns:
            ForecastResult
        """
        data = TimeSeriesData(
            timestamps=timestamps,
            values=values,
            metric_name=metric_name
        )
        
        if len(data) < 6:
            # Not enough data for forecasting
            # Return a simple projection based on mean
            mean_val = data.mean
            std_val = data.std if data.std > 0 else mean_val * 0.1
            
            last_timestamp = timestamps[-1] if timestamps else datetime.now()
            forecast_timestamps = [last_timestamp + timedelta(days=h+1) for h in range(horizon_days)]
            
            return ForecastResult(
                forecast=[mean_val] * horizon_days,
                timestamps=forecast_timestamps,
                lower_80=[mean_val - 1.28 * std_val] * horizon_days,
                upper_80=[mean_val + 1.28 * std_val] * horizon_days,
                lower_95=[mean_val - 1.96 * std_val] * horizon_days,
                upper_95=[mean_val + 1.96 * std_val] * horizon_days,
                explanations=["⚠️ Insufficient historical data (<6 points). Using simple average projection."],
                scenario=scenario
            )
        
        return self.ensemble.forecast(data, horizon_days, scenario)
    
    def forecast_capacity_needs(
        self,
        rps_history: List[Tuple[datetime, float]],
        gpu_util_history: Optional[List[Tuple[datetime, float]]] = None,
        horizon_days: int = 30,
        scenario: str = "baseline"
    ) -> dict:
        """
        Forecast capacity needs based on historical metrics.
        
        Args:
            rps_history: List of (timestamp, rps) tuples
            gpu_util_history: Optional list of (timestamp, utilization) tuples
            horizon_days: Forecast horizon
            scenario: Scenario type
            
        Returns:
            Dictionary with forecast results for each metric
        """
        results = {}
        
        # Forecast RPS
        if rps_history:
            timestamps = [t for t, _ in rps_history]
            values = [v for _, v in rps_history]
            results["rps"] = self.forecast_metric(
                timestamps, values, "RPS", horizon_days, scenario
            )
        
        # Forecast GPU utilization
        if gpu_util_history:
            timestamps = [t for t, _ in gpu_util_history]
            values = [v for _, v in gpu_util_history]
            results["gpu_utilization"] = self.forecast_metric(
                timestamps, values, "GPU Utilization", horizon_days, scenario
            )
        
        return results
    
    def calculate_mape(
        self,
        actual: List[float],
        predicted: List[float]
    ) -> float:
        """
        Calculate Mean Absolute Percentage Error (MAPE).
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            MAPE value (0-100 percentage)
        """
        if len(actual) != len(predicted):
            raise ValueError("Actual and predicted must have same length")
        
        if not actual:
            return 0.0
        
        total_error = 0.0
        count = 0
        
        for a, p in zip(actual, predicted):
            if a != 0:
                total_error += abs((a - p) / a)
                count += 1
        
        return (total_error / count * 100) if count > 0 else 0.0
    
    def check_retraining_needed(
        self,
        actual: List[float],
        predicted: List[float],
        threshold_pct: float = 10.0
    ) -> Tuple[bool, float]:
        """
        Check if model retraining is needed based on forecast accuracy.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            threshold_pct: MAPE threshold for retraining
            
        Returns:
            (needs_retraining, mape)
        """
        mape = self.calculate_mape(actual, predicted)
        return mape > threshold_pct, mape
