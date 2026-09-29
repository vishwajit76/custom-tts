"""Data-rights manifest: which recordings we may train on, and why. JSONL, one entry per source/speaker group.

Entry schema (all required unless noted):
  rights_id              unique id, referenced by manifest rows (row.rights_id); rows without one match by speaker_id
  source                 where the audio came from (project, vendor, dataset name / URL)
  licence                licence or contract identifier (e.g. "CC-BY-4.0", "contract-2026-014")
  consent_record_id      id of the stored consent record
  speaker_authorization  true only if the speaker(s) authorized TTS/voice training
  speaker_ids            list of speaker_id values covered
  permitted_uses         list, must contain "tts_training" to be trained on
  vendor_generated       true if the audio was produced by a TTS/voice vendor API
  vendor_generation_permission  required true when vendor_generated (written permission to train on that output)
  notes                  optional
"""
import json
from pathlib import Path

REQUIRED = ("rights_id", "source", "licence", "consent_record_id", "speaker_authorization", "speaker_ids",
            "permitted_uses", "vendor_generated")
SCHEMA = {"type": "object", "required": list(REQUIRED), "properties": {
    "rights_id": {"type": "string"}, "source": {"type": "string"}, "licence": {"type": "string"},
    "consent_record_id": {"type": "string"}, "speaker_authorization": {"type": "boolean"},
    "speaker_ids": {"type": "array", "items": {"type": "string"}},
    "permitted_uses": {"type": "array", "items": {"type": "string"}},
    "vendor_generated": {"type": "boolean"}, "vendor_generation_permission": {"type": "boolean"},
    "notes": {"type": "string"}}}


class RightsError(Exception):
    pass


def load_rights(path: Path) -> list[dict]:
    entries = [json.loads(line) for line in Path(path).read_text("utf-8").splitlines() if line.strip()]
    for e in entries:
        missing = [k for k in REQUIRED if e.get(k) in (None, "", [])  and not isinstance(e.get(k), bool)]
        if missing:
            raise RightsError(f"rights entry {e.get('rights_id', '?')}: missing {missing}")
        # a string here would make `in` a substring test ("tts_training_no" contains "tts_training")
        for k in ("speaker_ids", "permitted_uses"):
            if not isinstance(e[k], list) or not all(isinstance(x, str) for x in e[k]):
                raise RightsError(f"rights entry {e['rights_id']}: {k} must be a list of strings")
        for k in ("speaker_authorization", "vendor_generated"):
            if not isinstance(e[k], bool):
                raise RightsError(f"rights entry {e['rights_id']}: {k} must be true or false")
    ids = [e["rights_id"] for e in entries]
    if dup := sorted({i for i in ids if ids.count(i) > 1}):
        raise RightsError(f"duplicate rights_id: {dup}")
    return entries


def violation(entry: dict | None) -> str | None:
    if entry is None:
        return "no data-rights entry"
    if entry["speaker_authorization"] is not True:
        return "speaker not authorized"
    if "tts_training" not in entry["permitted_uses"]:
        return "tts_training not a permitted use"
    if entry["vendor_generated"] and entry.get("vendor_generation_permission") is not True:
        return "vendor-generated audio without permission"
    return None


def check_rows(rows: list[dict], entries: list[dict]) -> list[tuple[dict, str]]:
    """-> [(row, reason)] for every row that may NOT be trained on. Empty list means all rows are cleared."""
    by_id = {e["rights_id"]: e for e in entries}
    by_speaker = {s: e for e in entries for s in e["speaker_ids"]}
    bad = []
    for r in rows:
        entry = by_id.get(r.get("rights_id")) if r.get("rights_id") else by_speaker.get(r["speaker_id"])
        why = violation(entry)
        if why is None and r["speaker_id"] not in entry["speaker_ids"]:
            why = "speaker not covered by rights entry"
        if why:
            bad.append((r, why))
    return bad


def enforce(rows: list[dict], rights_path: Path | None) -> None:
    """Raise RightsError unless every row is cleared."""
    if rights_path is None:
        raise RightsError("no data-rights manifest given (--rights); refusing to process data without rights entries")
    bad = check_rows(rows, load_rights(rights_path))
    if bad:
        detail = "; ".join(f"{r['id']}({r['speaker_id']}): {why}" for r, why in bad[:10])
        raise RightsError(f"{len(bad)} clip(s) lack training rights: {detail}")
