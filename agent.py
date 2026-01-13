"""
Chatbot/Agent interface for infrastructure capacity forecasting.
"""

import json
from typing import Optional
from models import UserInput, CapacityPlan
from forecast_engine import ForecastEngine


class CapacityForecastAgent:
    """Agent interface for collecting user input and generating capacity plans."""
    
    def __init__(self):
        self.engine = ForecastEngine()
        self.user_input: Optional[UserInput] = None
    
    def welcome_message(self) -> str:
        """Display welcome message."""
        return (
            "🤖 Welcome to Infrastructure Capacity Forecasting Agent!\n"
            "I'll help you forecast your infrastructure capacity requirements.\n"
            "Let's start by gathering some information about your system.\n"
        )
    
    def prompt_concurrent_users(self) -> str:
        """Prompt for concurrent users."""
        return "📊 Please enter the number of concurrent users (peak): "
    
    def prompt_registered_users(self) -> str:
        """Prompt for registered users."""
        return "👥 Please enter the total number of registered users: "
    
    def prompt_optional_params(self) -> str:
        """Prompt for optional parameters."""
        return (
            "\n💡 Optional parameters (press Enter to skip):\n"
            "   - Growth rate (%): "
        )
    
    def validate_concurrent_users(self, value: str) -> tuple[bool, Optional[int], str]:
        """Validate concurrent users input."""
        try:
            num = int(value.strip())
            if num <= 0:
                return False, None, "❌ Concurrent users must be greater than 0."
            if num > 10000000:
                return False, None, "❌ Concurrent users seems unreasonably high. Please verify."
            return True, num, "✓ Valid"
        except ValueError:
            return False, None, "❌ Please enter a valid number."
    
    def validate_registered_users(self, value: str) -> tuple[bool, Optional[int], str]:
        """Validate registered users input."""
        try:
            num = int(value.strip())
            if num <= 0:
                return False, None, "❌ Registered users must be greater than 0."
            if num > 1000000000:
                return False, None, "❌ Registered users seems unreasonably high. Please verify."
            return True, num, "✓ Valid"
        except ValueError:
            return False, None, "❌ Please enter a valid number."
    
    def validate_growth_rate(self, value: str) -> tuple[bool, Optional[float], str]:
        """Validate growth rate input."""
        if not value.strip():
            return True, None, "✓ Skipped"
        try:
            rate = float(value.strip())
            if rate < 0 or rate > 1000:
                return False, None, "❌ Growth rate should be between 0 and 1000%."
            return True, rate, "✓ Valid"
        except ValueError:
            return False, None, "❌ Please enter a valid number."
    
    def collect_input(self) -> UserInput:
        """Collect user input interactively."""
        print(self.welcome_message())
        
        # Collect concurrent users
        while True:
            concurrent_input = input(self.prompt_concurrent_users())
            valid, value, message = self.validate_concurrent_users(concurrent_input)
            print(message)
            if valid and value is not None:
                concurrent_users = value
                break
        
        # Collect registered users
        while True:
            registered_input = input(self.prompt_registered_users())
            valid, value, message = self.validate_registered_users(registered_input)
            print(message)
            if valid and value is not None:
                registered_users = value
                break
        
        # Collect optional parameters
        growth_rate = None
        growth_input = input(self.prompt_optional_params())
        valid, value, message = self.validate_growth_rate(growth_input)
        if valid and value is not None:
            growth_rate = value
        
        return UserInput(
            concurrent_users=concurrent_users,
            registered_users=registered_users,
            growth_rate=growth_rate,
        )
    
    def format_capacity_plan(self, plan: CapacityPlan) -> str:
        """Format capacity plan for display."""
        output = []
        output.append("\n" + "="*60)
        output.append("📋 INFRASTRUCTURE CAPACITY PLAN")
        output.append("="*60)
        
        # User Input Summary
        output.append("\n📊 INPUT SUMMARY:")
        output.append(f"   Concurrent Users: {plan.user_input.concurrent_users:,}")
        output.append(f"   Registered Users: {plan.user_input.registered_users:,}")
        if plan.user_input.growth_rate:
            output.append(f"   Growth Rate: {plan.user_input.growth_rate}%")
        
        # Compute Resources
        output.append("\n💻 COMPUTE RESOURCES:")
        output.append(f"   CPU Cores: {plan.compute.cpu_cores:.2f}")
        output.append(f"   Memory: {plan.compute.memory_gb:.2f} GB")
        output.append(f"   Recommended Instance Types: {', '.join(plan.compute.recommended_instance_types)}")
        output.append(f"   Instance Range: {plan.compute.min_instances} - {plan.compute.max_instances} instances")
        
        # Storage Resources
        output.append("\n💾 STORAGE RESOURCES:")
        output.append(f"   Database Storage: {plan.storage.database_storage_gb:.2f} GB")
        output.append(f"   File Storage: {plan.storage.file_storage_gb:.2f} GB")
        output.append(f"   Backup Storage: {plan.storage.backup_storage_gb:.2f} GB")
        output.append(f"   Total Storage: {plan.storage.total_storage_gb:.2f} GB")
        
        # Network Resources
        output.append("\n🌐 NETWORK RESOURCES:")
        output.append(f"   Bandwidth: {plan.network.bandwidth_mbps:.2f} Mbps")
        output.append(f"   Connection Limit: {plan.network.connection_limit:,}")
        output.append(f"   Load Balancers: {plan.network.recommended_load_balancers}")
        
        # Scaling Recommendations
        output.append("\n📈 SCALING RECOMMENDATIONS:")
        output.append(f"   Horizontal Scaling: {plan.scaling.horizontal_scaling}")
        output.append(f"   Vertical Scaling: {plan.scaling.vertical_scaling}")
        output.append(f"   Auto-scaling: {plan.scaling.auto_scaling_min} - {plan.scaling.auto_scaling_max} instances")
        output.append(f"   CPU Threshold: {plan.scaling.auto_scaling_threshold_cpu}%")
        output.append(f"   Memory Threshold: {plan.scaling.auto_scaling_threshold_memory}%")
        
        # Cost Estimates (if available)
        if plan.cost:
            output.append("\n💰 COST ESTIMATES:")
            output.append(f"   Monthly Compute Cost: ${plan.cost.monthly_compute_cost:,.2f}")
            output.append(f"   Monthly Storage Cost: ${plan.cost.monthly_storage_cost:,.2f}")
            output.append(f"   Monthly Network Cost: ${plan.cost.monthly_network_cost:,.2f}")
            output.append(f"   Total Monthly Cost: ${plan.cost.total_monthly_cost:,.2f} {plan.cost.currency}")
        
        output.append("\n" + "="*60)
        output.append("✅ Capacity plan generated successfully!")
        output.append("="*60 + "\n")
        
        return "\n".join(output)
    
    def export_plan(self, plan: CapacityPlan, filename: str = "capacity_plan.json") -> str:
        """Export capacity plan to JSON file."""
        plan_dict = plan.to_dict()
        with open(filename, 'w') as f:
            json.dump(plan_dict, f, indent=2)
        return f"📄 Capacity plan exported to {filename}"
    
    def run(self, include_cost: bool = False):
        """Run the agent interactively."""
        try:
            # Collect input
            user_input = self.collect_input()
            
            # Generate plan
            print("\n⏳ Calculating infrastructure capacity requirements...")
            plan = self.engine.generate_capacity_plan(user_input, include_cost=include_cost)
            
            # Display results
            print(self.format_capacity_plan(plan))
            
            # Offer export
            export = input("💾 Export plan to JSON file? (y/n): ").strip().lower()
            if export == 'y':
                filename = input("   Enter filename (default: capacity_plan.json): ").strip()
                if not filename:
                    filename = "capacity_plan.json"
                if not filename.endswith('.json'):
                    filename += '.json'
                print(self.export_plan(plan, filename))
            
            return plan
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Exiting...")
            return None
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            return None
