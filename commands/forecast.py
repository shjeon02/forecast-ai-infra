#!/usr/bin/env python3
"""
Command-line entry point for infrastructure capacity forecasting.
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import CapacityForecastAgent


def main():
    """Main entry point for the forecast command."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Infrastructure Capacity Forecasting Tool"
    )
    parser.add_argument(
        '--cost',
        action='store_true',
        help='Include cost estimates in the capacity plan'
    )
    parser.add_argument(
        '--concurrent-users',
        type=int,
        help='Number of concurrent users (skip interactive mode)'
    )
    parser.add_argument(
        '--registered-users',
        type=int,
        help='Number of registered users (skip interactive mode)'
    )
    parser.add_argument(
        '--growth-rate',
        type=float,
        help='Expected growth rate percentage'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path for JSON export'
    )
    
    args = parser.parse_args()
    
    agent = CapacityForecastAgent()
    
    # Non-interactive mode if both user counts provided
    if args.concurrent_users and args.registered_users:
        from models import UserInput
        from forecast_engine import ForecastEngine
        
        user_input = UserInput(
            concurrent_users=args.concurrent_users,
            registered_users=args.registered_users,
            growth_rate=args.growth_rate,
        )
        
        engine = ForecastEngine()
        plan = engine.generate_capacity_plan(user_input, include_cost=args.cost)
        
        print(agent.format_capacity_plan(plan))
        
        if args.output:
            print(agent.export_plan(plan, args.output))
        else:
            print(agent.export_plan(plan))
    else:
        # Interactive mode
        agent.run(include_cost=args.cost)


if __name__ == "__main__":
    main()
