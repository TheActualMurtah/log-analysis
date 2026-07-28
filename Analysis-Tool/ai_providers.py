from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


class AIProvider(ABC):
    """Common interface for AI backends used by analysis tooling."""

    # Inputs: input_text user payload, prompt system instructions, model override or None.
    # Output: provider-generated summary text.
    # Options: implementation decides how to use model override.
    # Pre: subclass implements provider call logic.
    # Post: returns non-empty summary text or raises RuntimeError on provider failure.
    @abstractmethod
    def summarize(self, input_text: str, prompt: str, model: Optional[str]) -> str:
        """Return a summary from provider output text."""


_GH_CLI_FALLBACK_PATHS = (
    "gh",
    "/opt/homebrew/bin/gh",
    "/usr/local/bin/gh",
)


# Inputs: none.
# Output: GitHub auth token string or None.
# Options: checks gh on PATH first, then common Homebrew install paths.
# Pre: GitHub CLI may or may not be installed/authenticated.
# Post: returns first valid token found without raising for missing gh binaries.
def _read_gh_auth_token() -> Optional[str]:
    """Best-effort token lookup from GitHub CLI auth state.

    Tries `gh` on PATH first, then common Homebrew install locations, since
    `gh` is not always on PATH in every shell/subprocess environment.
    """
    for gh_path in _GH_CLI_FALLBACK_PATHS:
        try:
            result = subprocess.run(
                [gh_path, "auth", "token"],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            continue

        if result.returncode != 0:
            continue

        token = result.stdout.strip()
        if token:
            return token

    return None


class BobProvider(AIProvider):
    # Inputs: input_text payload, prompt instructions, optional model name.
    # Output: summary text from Bob CLI stdout.
    # Options: passes --model only when provided.
    # Pre: bob CLI exists and is executable in PATH.
    # Post: returns non-empty summary or raises RuntimeError when execution fails.
    def summarize(self, input_text: str, prompt: str, model: Optional[str]) -> str:
        command = [
            "bob",
            "--hide-intermediary-output",
            "--output-format",
            "text",
            "-p",
            prompt,
        ]
        if model:
            command.extend(["--model", model])

        try:
            result = subprocess.run(
                command,
                input=input_text,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "The 'bob' CLI was not found in PATH. Install or configure Bob first."
            ) from exc

        if result.returncode != 0:
            stderr = result.stderr.strip() or "Bob exited with a non-zero status."
            raise RuntimeError(stderr)

        summary = result.stdout.strip()
        if not summary:
            raise RuntimeError("Bob returned an empty summary.")
        return summary


@dataclass
class CopilotConfig:
    token: str
    base_url: str = "https://api.githubcopilot.com"
    path: str = "/chat/completions"
    model: str = "gpt-4o-mini"
    timeout_seconds: int = 60


class GitHubCopilotProvider(AIProvider):
    """
    GitHub Copilot chat-completions provider.

    Endpoint and auth are configurable via environment variables so this can
    adapt to org-specific Copilot gateways without code changes.
    """

    # Inputs: CopilotConfig with token, endpoint, default model, and timeout.
    # Output: initialized provider instance.
    # Options: none.
    # Pre: config.token is set to a valid credential.
    # Post: provider stores config for future summarize calls.
    def __init__(self, config: CopilotConfig):
        self.config = config

    # Inputs: input_text payload, prompt instructions, optional model override.
    # Output: summary text parsed from Copilot chat completions response.
    # Options: uses override model when provided, else config default; fixed temperature=0.2.
    # Pre: API token and endpoint are valid and reachable.
    # Post: returns non-empty content or raises RuntimeError for HTTP/network/format errors.
    def summarize(self, input_text: str, prompt: str, model: Optional[str]) -> str:
        payload = {
            "model": model or self.config.model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_text},
            ],
            "temperature": 0.2,
        }

        url = self.config.base_url.rstrip("/") + self.config.path
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.config.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.timeout_seconds,
            ) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"GitHub Copilot request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"GitHub Copilot request failed: {exc.reason}") from exc

        try:
            parsed = json.loads(raw)
            content = parsed["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("GitHub Copilot response format was not recognized.") from exc

        if not content:
            raise RuntimeError("GitHub Copilot returned an empty summary.")
        return content


class MockProvider(AIProvider):
    """Fallback provider for demonstration or offline testing without API keys."""

    def summarize(self, input_text: str, prompt: str, model: Optional[str]) -> str:
        return (
            "### 🤖 AI Log Analysis Summary (Demo Mode)\n\n"
            "**Dominant Patterns Identified:**\n"
            "- Analyzed log events and extracted key message templates.\n"
            "- Detected recurring warnings and error stack traces in Jenkins build runs.\n\n"
            "**Key Findings:**\n"
            "1. **Permission / Security Noise:** Recurring `AccessDeniedException` warnings from anonymous access attempts.\n"
            "2. **Job Execution:** Multi-line log events properly parsed into structured templates.\n\n"
            "**Recommended Next Steps:**\n"
            "1. Configure a rule filter to silence expected noise.\n"
            "2. Inspect severe events within the Investigate tab."
        )

# Inputs: provider_name string.
# Output: concrete AIProvider implementation instance.
# Options: supports copilot aliases and bob.
# Pre: required credentials/tools for selected provider are available.
# Post: returns initialized provider or raises RuntimeError for unknown/misconfigured providers.
def build_provider(provider_name: str) -> AIProvider:
    normalized = provider_name.strip().lower()
    if normalized in {"mock", "demo"}:
        return MockProvider()

    if normalized == "bob":
        return BobProvider()

    if normalized in {"copilot", "github-copilot", "github_copilot"}:
        token = (
            os.getenv("COPILOT_API_KEY")
            or os.getenv("GITHUB_TOKEN")
            or os.getenv("GH_TOKEN")
            or _read_gh_auth_token()
        )
        if not token:
            raise RuntimeError(
                "GitHub Copilot credentials are missing. Set COPILOT_API_KEY, GITHUB_TOKEN, or GH_TOKEN, or run 'gh auth login'."
            )

        base_url = os.getenv("COPILOT_API_BASE_URL", "https://api.githubcopilot.com")
        path = os.getenv("COPILOT_API_PATH", "/chat/completions")
        default_model = os.getenv("COPILOT_MODEL", "gpt-4o-mini")
        timeout_seconds = int(os.getenv("COPILOT_TIMEOUT_SECONDS", "60"))

        return GitHubCopilotProvider(
            CopilotConfig(
                token=token,
                base_url=base_url,
                path=path,
                model=default_model,
                timeout_seconds=timeout_seconds,
            )
        )

    raise RuntimeError(
        f"Unknown AI provider '{provider_name}'. Supported values: copilot, bob, mock"
    )

