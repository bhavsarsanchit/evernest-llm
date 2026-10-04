from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
DATASET = ROOT / "evernest-case-study-dataset"
CRM_DIR = DATASET / "crm"
DOCS_DIR = DATASET / "documents"
MANIFEST = DOCS_DIR / "manifest.csv"

DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "assistant.sqlite"
SPEC_PATH = ROOT / "agent.yaml"
CHROMA_DIR = DATA_DIR / "chroma"

BLOCKED_CATEGORIES = frozenset({"aml_documentation", "proof_of_identity"})


def opik_settings() -> dict[str, str] | None:
    """Where traces may be sent.

    An API key without OPIK_URL_OVERRIDE does not select the default US cloud.
    Set OPIK_ALLOW_CLOUD=1 to opt into that cloud explicitly.
    """
    url = os.environ.get("OPIK_URL_OVERRIDE", "").strip()
    key = os.environ.get("OPIK_API_KEY", "").strip()
    allow_cloud = os.environ.get("OPIK_ALLOW_CLOUD", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    project = os.environ.get("OPIK_PROJECT_NAME", "evernest-listing-assistant").strip()
    if url:
        return {"url": url, "api_key": key, "project": project}
    if key and allow_cloud:
        return {"url": "", "api_key": key, "project": project}
    return None


def api_settings(base_url: str = "") -> tuple[str, str] | None:
    """Key and host for any OpenAI-compatible chat API.

    API_KEY comes from the environment. API_BASE_URL overrides the spec
    base URL when it is set.
    """
    key = os.environ.get("API_KEY", "").strip()
    if not key:
        return None
    base = os.environ.get("API_BASE_URL", "").strip() or base_url.strip()
    if not base:
        return None
    return key, base.rstrip("/")
