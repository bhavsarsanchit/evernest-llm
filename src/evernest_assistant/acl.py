from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from evernest_assistant.db import one


@dataclass(frozen=True)
class Access:
    agent_id: str
    listing_id: str
    role: str
    can_read_listing: bool
    can_read_documents: bool
    reason: str


def access(con: sqlite3.Connection, agent_id: str, listing_id: str) -> Access:
    agent = one(con, "SELECT * FROM agents WHERE agent_id = ?", (agent_id,))
    listing = one(con, "SELECT * FROM listings WHERE listing_id = ?", (listing_id,))
    if agent is None:
        return Access(agent_id, listing_id, "", False, False, "unknown agent")
    if listing is None:
        return Access(agent_id, listing_id, agent["role"], False, False, "unknown listing")

    role = agent["role"]
    if role == "ALL_ACCESS":
        return Access(agent_id, listing_id, role, True, True, "all access")

    if role == "TEAM_LEAD":
        allowed = agent["branch_id"] == listing["branch_id"] and bool(agent["branch_id"])
        reason = "team lead of this branch" if allowed else "team lead of another branch"
        return Access(agent_id, listing_id, role, allowed, allowed, reason)

    if listing["owner_agent_id"] == agent_id:
        return Access(agent_id, listing_id, role, True, True, "owner")

    collab = one(
        con,
        """
        SELECT * FROM listing_collaborators
        WHERE listing_id = ? AND agent_id = ?
        """,
        (listing_id, agent_id),
    )
    if collab is not None and _flag(collab["read_access"]):
        docs = _flag(collab["document_read_access"])
        return Access(
            agent_id,
            listing_id,
            role,
            True,
            docs,
            "collaborator with documents" if docs else "collaborator without documents",
        )
    return Access(agent_id, listing_id, role, False, False, "not owner and no collaboration")


def _flag(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}
