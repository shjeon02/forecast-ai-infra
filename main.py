#!/usr/bin/env python3
"""
Main entry point for the Infrastructure Capacity Forecasting application.
"""

from agent import CapacityForecastAgent


def main():
    """Main function to run the agent."""
    agent = CapacityForecastAgent()
    agent.run(include_cost=False)


if __name__ == "__main__":
    main()
