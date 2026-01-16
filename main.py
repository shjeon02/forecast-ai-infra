#!/usr/bin/env python3
"""
Main entry point for the LLM Infrastructure Capacity Forecasting application.

Runs the interactive agent for LLM inference and training capacity planning.
"""

from llm_agent import LLMCapacityAgent


def main():
    """Main function to run the LLM agent."""
    agent = LLMCapacityAgent()
    agent.run(include_cost=True)


if __name__ == "__main__":
    main()
