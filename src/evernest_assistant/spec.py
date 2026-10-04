"""Load the tunable agent spec from agent.yaml."""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from evernest_assistant.config import SPEC_PATH


@dataclass(frozen=True)
class AgentSpec:
    model: str
    base_url: str
    temperature: float
    max_tokens: int
    top_k: int
    system: str

    def with_overrides(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_k: int | None = None,
    ) -> "AgentSpec":
        return AgentSpec(
            model=model or self.model,
            base_url=self.base_url,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            top_k=self.top_k if top_k is None else top_k,
            system=self.system,
        )


def load_spec() -> AgentSpec:
    raw = yaml.safe_load(SPEC_PATH.read_text())
    model = raw["model"]
    return AgentSpec(
        model=str(model["name"]),
        base_url=str(model.get("base_url", "")).rstrip("/"),
        temperature=float(model.get("temperature", 0.1)),
        max_tokens=int(model.get("max_tokens", 700)),
        top_k=int(raw.get("retrieval", {}).get("top_k", 6)),
        system=str(raw["prompt"]["system"]).strip(),
    )
