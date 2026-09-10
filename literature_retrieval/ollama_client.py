"""
Ollama Client Wrapper

Model-agnostic HTTP client for Ollama's local LLM inference API.
Supports both /api/generate (single-shot) and /api/chat (multi-turn) endpoints.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import requests

logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    """Configuration for Ollama client."""

    base_url: str = "http://localhost:11434"
    model: str = "qwen3.5:9b"
    thinking: Union[bool, str] = False
    temperature: float = 0.3
    top_p: float = 0.9
    timeout: int = 600
    options: Dict[str, Any] = field(default_factory=dict)


class OllamaClient:
    """
    HTTP client for Ollama local LLM API.

    Supports:
    - /api/generate: Single-shot prompt → response
    - /api/chat: Multi-turn conversation with message history
    - /api/tags: List available models
    - Health check
    """

    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()
        self.session = requests.Session()

    def is_available(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            resp = self.session.get(
                f"{self.config.base_url}/api/tags",
                timeout=5,
            )
            return resp.status_code == 200
        except requests.ConnectionError:
            return False

    def list_models(self) -> List[str]:
        """List available models on the Ollama server."""
        try:
            resp = self.session.get(
                f"{self.config.base_url}/api/tags",
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error("Failed to list models: %s", e)
            return []

    def generate_raw(self, prompt: str, system: str = None) -> Dict[str, Any]:
        """
        Single-shot generation using /api/generate.

        Returns the full Ollama JSON payload so callers can inspect
        response / thinking / done_reason when debugging failures.
        """
        url = f"{self.config.base_url}/api/generate"

        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "think": self.config.thinking,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                **self.config.options,
            },
        }
        if system:
            payload["system"] = system

        try:
            logger.info("Generating with model=%s", self.config.model)
            response = self.session.post(
                url, json=payload, timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.Timeout:
            raise RuntimeError(
                f"Ollama request timed out after {self.config.timeout}s. "
                "Try increasing timeout or using a smaller model."
            )
        except requests.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.config.base_url}. "
                "Is Ollama running? Start with: ollama serve"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")

    def generate(self, prompt: str, system: str = None) -> str:
        """Single-shot generation; returns only the final response text."""
        result = self.generate_raw(prompt, system=system)
        return result.get("response", "") or ""

    def chat(
        self,
        messages: List[Dict[str, str]],
        system: str = None,
    ) -> str:
        """
        Multi-turn conversation using /api/chat.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            system: Optional system prompt

        Returns:
            Assistant's response text
        """
        url = f"{self.config.base_url}/api/chat"

        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)

        payload = {
            "model": self.config.model,
            "messages": chat_messages,
            "stream": False,
            "think": self.config.thinking,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                **self.config.options,
            },
        }

        try:
            logger.info(
                "Chat with model=%s, %d messages",
                self.config.model,
                len(messages),
            )
            response = self.session.post(
                url, json=payload, timeout=self.config.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result["message"]["content"]
        except requests.Timeout:
            raise RuntimeError(
                f"Ollama chat timed out after {self.config.timeout}s. "
                "Try increasing timeout or using a smaller model."
            )
        except requests.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.config.base_url}. "
                "Is Ollama running? Start with: ollama serve"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama chat API error: {e}")
