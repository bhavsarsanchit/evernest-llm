"""Answer one listing question from Chroma pages and a short CRM extract."""

from __future__ import annotations

import datetime
import json
import os
import sqlite3
from typing import Any
from urllib.parse import urlparse

from evernest_assistant.acl import access
from evernest_assistant.config import opik_settings
from evernest_assistant.db import all_rows, connect, load_crm, one
from evernest_assistant.index import search
from evernest_assistant.spec import AgentSpec, load_spec

_CRM: sqlite3.Connection | None = None


def crm() -> sqlite3.Connection:
    global _CRM
    if _CRM is None:
        _CRM = connect()
        load_crm(_CRM)
    return _CRM


def _crm_text(con: sqlite3.Connection, listing_id: str) -> str:
    listing = one(con, "SELECT * FROM listings WHERE listing_id = ?", (listing_id,))
    if listing is None:
        return ""
    lines = [
        f"listings {listing['listing_id']}: {listing['street']}, {listing['zip']} {listing['city']}. "
        f"status {listing['status']}, price {listing['purchase_price']} EUR, "
        f"living area {listing['living_area_sqm']} m2, "
        f"Hausgeld {listing['condo_fee_monthly'] or 'n/a'} EUR/month, "
        f"rented {listing['is_rented']}, cold rent {listing['monthly_rent_cold'] or 'n/a'}."
    ]
    offers = all_rows(
        con,
        "SELECT * FROM offers WHERE listing_id = ? ORDER BY created_at",
        (listing_id,),
    )
    amounts = []
    for offer in offers:
        try:
            amounts.append((offer["offer_id"], float(offer["amount_eur"]), offer["contact_id"]))
        except ValueError:
            continue
    suspect = set()
    for left_id, left_amount, left_contact in amounts:
        for right_id, right_amount, right_contact in amounts:
            if left_id >= right_id or left_contact != right_contact or left_amount == 0:
                continue
            ratio = right_amount / left_amount
            if 9 <= ratio <= 11:
                suspect.add(right_id)
    for offer in offers:
        flag = " suspect 10x duplicate, ignore as the highest bid" if offer["offer_id"] in suspect else ""
        lines.append(
            f"offers {offer['offer_id']}: {offer['amount_eur']} EUR {offer['status']} "
            f"from {offer['contact_id']} at {offer['created_at']}{flag}."
        )
    calls = all_rows(
        con,
        """
        SELECT activity_id, additional_details FROM activities
        WHERE listing_id = ? AND action = 'CALL' AND additional_details != ''
        """,
        (listing_id,),
    )
    for call in calls:
        try:
            payload = json.loads(call["additional_details"])
        except json.JSONDecodeError:
            continue
        if payload.get("source") != "sipgate":
            continue
        summary = payload.get("ai_summary") or ""
        topics = ", ".join(payload.get("ai_summary_topics") or [])
        if summary or topics:
            lines.append(f"activities {call['activity_id']} sipgate: {summary} Topics: {topics}.")
    return "\n".join(lines)


def _client(base_url: str):
    from openai import OpenAI

    from evernest_assistant.config import api_settings

    settings = api_settings(base_url)
    if settings is None:
        return None
    key, base = settings
    return OpenAI(api_key=key, base_url=base), base


def ask(
    agent_id: str,
    listing_id: str,
    question: str,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    spec = load_spec().with_overrides(model, temperature, max_tokens, top_k)
    decision = access(crm(), agent_id, listing_id)
    if not decision.can_read_listing:
        return _result(
            spec,
            "I don't have access to that listing.",
            [],
            decision,
            refused=True,
            model_called=False,
        )

    pages = search(question, listing_id, spec.top_k) if decision.can_read_documents else []
    crm_text = _crm_text(crm(), listing_id)
    if not pages and not crm_text:
        return _result(spec, "I don't know. The available records don't contain an answer.", [], decision, refused=False)

    user = _user_message(question, pages, crm_text, decision.can_read_documents)
    opened = _client(spec.base_url)
    if opened is None:
        return _result(
            spec,
            "No model API is configured. Retrieval is ready; set API_KEY, and set API_BASE_URL or model.base_url.",
            pages,
            decision,
            refused=False,
            model_called=False,
        )
    client, base = opened
    messages = [
        {"role": "system", "content": spec.system},
        {"role": "user", "content": user},
    ]
    request = {
        "agent_id": agent_id,
        "listing_id": listing_id,
        "question": question,
        "model": spec.model,
        "temperature": spec.temperature,
        "max_tokens": spec.max_tokens,
        "top_k": spec.top_k,
    }
    started = datetime.datetime.now(datetime.timezone.utc)
    call = {
        "model": spec.model,
        "temperature": spec.temperature,
        "max_tokens": spec.max_tokens,
        "messages": messages,
    }
    if _provider(base) == "openrouter.ai":
        call["extra_body"] = {"usage": {"include": True}}
    completion = client.chat.completions.create(**call)
    ended = datetime.datetime.now(datetime.timezone.utc)
    answer = (completion.choices[0].message.content or "").strip()
    usage, cost = _usage_from(completion)
    tokens = {"prompt": usage["prompt_tokens"], "completion": usage["completion_tokens"]}
    result = _result(spec, answer, pages, decision, refused=False, tokens=tokens)
    _trace(request, result, messages, started, ended, usage, cost, _provider(base))
    return result


def _user_message(question: str, pages: list[dict], crm_text: str, documents: bool) -> str:
    blocks = [f"Question: {question}"]
    if documents and pages:
        blocks.append("Document pages:")
        for page in pages:
            blocks.append(
                f"[{page['document_id']} page {page['page']} | {page['sub_category']}]\n{page['text']}"
            )
    elif not documents:
        blocks.append("Document pages: not available for this agent. Do not infer them.")
    else:
        blocks.append("Document pages: none retrieved.")
    if crm_text:
        blocks.append("CRM:\n" + crm_text)
    return "\n\n".join(blocks)


def _result(spec: AgentSpec, answer: str, pages: list[dict], decision, refused: bool, tokens=None, model_called=True) -> dict[str, Any]:
    return {
        "answer": answer,
        "sources": [
            {
                "document_id": page["document_id"],
                "page": page["page"],
                "listing_id": page["listing_id"],
                "sub_category": page["sub_category"],
            }
            for page in pages
        ],
        "access": {
            "can_read_listing": decision.can_read_listing,
            "can_read_documents": decision.can_read_documents,
            "reason": decision.reason,
            "role": decision.role,
        },
        "model": spec.model if model_called else "",
        "temperature": spec.temperature,
        "max_tokens": spec.max_tokens,
        "top_k": spec.top_k,
        "refused": refused,
        "tokens": tokens or {"prompt": 0, "completion": 0},
    }


def _provider(base_url: str) -> str:
    return urlparse(base_url).netloc or base_url


def _usage_from(completion) -> tuple[dict[str, int], float | None]:
    raw: dict[str, Any] = {}
    usage = getattr(completion, "usage", None)
    if usage is not None and hasattr(usage, "model_dump"):
        raw = usage.model_dump()
    prompt = int(raw.get("prompt_tokens") or 0)
    completion_tokens = int(raw.get("completion_tokens") or 0)
    total = int(raw.get("total_tokens") or prompt + completion_tokens)
    cost = raw.get("cost")
    try:
        cost_value = None if cost is None else float(cost)
    except (TypeError, ValueError):
        cost_value = None
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion_tokens,
        "total_tokens": total,
    }, cost_value


def _trace(
    request: dict[str, Any],
    result: dict[str, Any],
    messages: list[dict[str, str]],
    started: datetime.datetime,
    ended: datetime.datetime,
    usage: dict[str, int],
    cost: float | None,
    provider: str,
) -> None:
    settings = opik_settings()
    if settings is None:
        return
    try:
        import opik
    except ImportError:
        return
    if settings["url"]:
        os.environ["OPIK_URL_OVERRIDE"] = settings["url"]
    if settings["api_key"]:
        os.environ["OPIK_API_KEY"] = settings["api_key"]
    try:
        client = opik.Opik(project_name=settings["project"])
        trace = client.trace(
            name="ask",
            start_time=started,
            end_time=ended,
            input=request,
            output={"answer": result["answer"]},
            metadata={"sources": result["sources"], "access": result["access"]},
        )
        trace.span(
            name="chat",
            type="llm",
            start_time=started,
            end_time=ended,
            model=result["model"],
            provider=provider,
            input={**request, "messages": messages},
            output={"answer": result["answer"]},
            usage=usage,
            total_cost=cost,
        )
        client.flush()
    except Exception:
        return
