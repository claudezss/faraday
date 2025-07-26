"""
Multi-LLM configuration for comprehensive benchmarking.
Supports GPT, Claude, Gemini, and other providers.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

import dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


@dataclass
class LLMConfig:
    """Configuration for a specific LLM provider."""

    name: str
    provider: str
    model: str
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    api_key_env: str = ""
    base_url: Optional[str] = None
    additional_params: Dict[str, Any] = None


class LLMConfigManager:
    """Manages configurations for different LLM providers."""

    def __init__(self):
        self.configs = self._define_llm_configs()

    def _define_llm_configs(self) -> Dict[str, LLMConfig]:
        """Define configurations for all supported LLM providers."""
        return {
            # OpenAI GPT Models
            "gpt-4.1": LLMConfig(
                name="gpt-4.1",
                provider="openai",
                model="gpt-4.1",
                api_key_env="OPENAI_API_KEY",
            ),
            "gpt-4.1-mini": LLMConfig(
                name="gpt-4.1-mini",
                provider="openai",
                model="gpt-4.1-mini",
                api_key_env="OPENAI_API_KEY",
            ),
            "gemini-2.5-pro": LLMConfig(
                name="gemini-2.5-pro",
                provider="openai",
                model="gemini-2.5-pro",
                api_key_env="GEMINI_API_KEY",
                base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            ),
            "gemini-2.5-flash": LLMConfig(
                name="gemini-2.5-flash",
                provider="openai",
                model="gemini-2.5-flash",
                api_key_env="GEMINI_API_KEY",
                base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            ),
            # Anthropic Claude Models
            "claude-sonnet-4": LLMConfig(
                name="claude-sonnet-4",
                provider="anthropic",
                model="claude-sonnet-4-20250514",
                api_key_env="ANTHROPIC_API_KEY",
            ),
            "claude-3-7-sonnet": LLMConfig(
                name="claude-3-7-sonnet",
                provider="anthropic",
                model="claude-3-7-sonnet-latest",
                api_key_env="ANTHROPIC_API_KEY",
            ),
        }

    def get_llm_instance(self, config_name: str):
        """Create LLM instance from configuration."""
        if config_name not in self.configs:
            raise ValueError(f"Unknown LLM config: {config_name}")

        config = self.configs[config_name]

        # Get API key from environment
        api_key = os.environ.get(config.api_key_env)
        if not api_key and config.provider not in ["ollama"]:
            raise ValueError(
                f"API key not found in environment variable: {config.api_key_env}"
            )

        # Create LLM instance based on provider
        if config.provider == "openai":
            return ChatOpenAI(
                model=config.model,
                # temperature=config.temperature,
                api_key=api_key,
            )

        elif config.provider == "anthropic":
            return ChatAnthropic(
                model=config.model,
                # temperature=config.temperature,
                api_key=api_key,
            )

        else:
            raise ValueError(f"Unsupported provider: {config.provider}")

    def get_available_configs(self) -> Dict[str, LLMConfig]:
        """Get all available LLM configurations."""
        available = {}

        for name, config in self.configs.items():
            # Check if API key is available
            api_key = os.environ.get(config.api_key_env)
            if api_key or config.provider == "ollama":
                available[name] = config

        return available

    def validate_config(self, config_name: str) -> bool:
        """Validate that a configuration is properly set up."""
        try:
            llm = self.get_llm_instance(config_name)
            # Test with a simple prompt
            response = llm.invoke("Hello, this is a test.")
            return bool(response.content)
        except Exception as e:
            print(f"Config validation failed for {config_name}: {e}")
            return False


# Predefined test suites for different research scenarios
LLM_TEST_SUITES = {
    "comprehensive": [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gpt-4.1",
        "gpt-4.1-mini",
        "claude-sonnet-4",
        "claude-3-7-sonnet",
    ],
    "cost_effective": ["gemini-2.5-flash", "gpt-4.1-mini", "claude-3-7-sonnet"],
    "high_performance": ["gemini-2.5-pro", "gpt-4.1", "claude-sonnet-4"],
    "quick_test": ["gemini-2.5-flash"],
}


def get_test_suite(suite_name: str) -> list[str]:
    """Get predefined test suite."""
    if suite_name not in LLM_TEST_SUITES:
        available_suites = list(LLM_TEST_SUITES.keys())
        raise ValueError(
            f"Unknown test suite: {suite_name}. Available: {available_suites}"
        )

    return LLM_TEST_SUITES[suite_name]


def setup_environment_template():
    """Print template for environment variables setup."""
    template = """
# Environment Variables Setup Template for Multi-LLM Benchmarking
# Copy to .env file and fill in your API keys

# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Claude
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Additional Configuration
BENCHMARK_RESULTS_DIR=./benchmark_results
BENCHMARK_LOG_LEVEL=INFO
"""
    print(template)


if __name__ == "__main__":
    # Demo usage
    manager = LLMConfigManager()
    dotenv.load_dotenv()

    print("Available LLM Configurations:")
    print("-" * 40)
    available = manager.get_available_configs()

    for name, config in available.items():
        print(f"{name:20} | {config.name:25} | {config.provider}")

    print(f"\nFound {len(available)} available configurations")

    if not available:
        print("\nNo configurations available. Set up environment variables:")
        setup_environment_template()
