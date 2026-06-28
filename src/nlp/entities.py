"""
src/nlp/entities.py
Clinical named entity recognition (NER) using an LLM.
Extracts diagnoses, medications, procedures, and patient demographics
from free-text clinical notes with ICD-10 and CPT code suggestions.
"""

import json
import re
from dataclasses import dataclass, field, asdict
from src.llm.prompts import ENTITY_EXTRACTION_PROMPT


@dataclass
class Diagnosis:
    name: str
    icd10: str | None = None
    type: str = "secondary"       # "primary" or "secondary"
    confidence: float = 0.9
    evidence: str = ""


@dataclass
class Medication:
    name: str
    dose: str | None = None
    route: str | None = None
    frequency: str | None = None
    status: str = "admission"     # admission | discharge | new | continuing
    confidence: float = 0.9


@dataclass
class Procedure:
    name: str
    cpt: str | None = None
    confidence: float = 0.9


@dataclass
class PatientDemographics:
    age: str | None = None
    sex: str | None = None
    dob: str | None = None
    mrn: str | None = None


@dataclass
class PaymentIntegrity:
    diagnoses_documented: bool = True
    procedures_documented: bool = True
    medications_documented: bool = True
    flags: list[str] = field(default_factory=list)
    observations: str = ""


@dataclass
class ExtractionResult:
    clinical_summary: str = ""
    patient: PatientDemographics = field(default_factory=PatientDemographics)
    diagnoses: list[Diagnosis] = field(default_factory=list)
    medications: list[Medication] = field(default_factory=list)
    procedures: list[Procedure] = field(default_factory=list)
    payment_integrity: PaymentIntegrity = field(default_factory=PaymentIntegrity)
    overall_confidence: float = 0.9

    def to_dict(self) -> dict:
        return asdict(self)


def _clean_json(raw: str) -> str:
    """Strip markdown fences and DeepSeek-R1 reasoning tags from LLM output."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _parse_result(data: dict) -> ExtractionResult:
    """Map raw LLM JSON dict to typed ExtractionResult dataclass."""
    p = data.get("patient", {})
    pi = data.get("payment_integrity", {})

    return ExtractionResult(
        clinical_summary=data.get("clinical_summary", ""),
        overall_confidence=float(data.get("overall_confidence", 0.9)),
        patient=PatientDemographics(
            age=p.get("age"),
            sex=p.get("sex"),
            dob=p.get("dob"),
            mrn=p.get("mrn"),
        ),
        diagnoses=[
            Diagnosis(
                name=d.get("name", ""),
                icd10=d.get("icd10"),
                type=d.get("type", "secondary"),
                confidence=float(d.get("confidence", 0.9)),
                evidence=d.get("evidence", ""),
            )
            for d in data.get("diagnoses", [])
        ],
        medications=[
            Medication(
                name=m.get("name", ""),
                dose=m.get("dose"),
                route=m.get("route"),
                frequency=m.get("frequency"),
                status=m.get("status", "admission"),
                confidence=float(m.get("confidence", 0.9)),
            )
            for m in data.get("medications", [])
        ],
        procedures=[
            Procedure(
                name=pr.get("name", ""),
                cpt=pr.get("cpt"),
                confidence=float(pr.get("confidence", 0.9)),
            )
            for pr in data.get("procedures", [])
        ],
        payment_integrity=PaymentIntegrity(
            diagnoses_documented=pi.get("diagnoses_documented", True),
            procedures_documented=pi.get("procedures_documented", True),
            medications_documented=pi.get("medications_documented", True),
            flags=pi.get("flags", []),
            observations=pi.get("observations", ""),
        ),
    )


def extract_entities(text: str, llm_chat_fn) -> ExtractionResult:
    """
    Extract clinical entities from a free-text clinical note using an LLM.

    Args:
        text: Raw clinical note text (up to ~8000 chars used).
        llm_chat_fn: Callable matching signature llm_chat(system, messages, max_tokens) -> str.
                     Injected to keep this module LLM-backend agnostic.

    Returns:
        ExtractionResult with all extracted entities.

    Raises:
        ValueError: If the LLM output cannot be parsed as valid JSON.
    """
    raw = llm_chat_fn(
        ENTITY_EXTRACTION_PROMPT,
        [{"role": "user", "content": f"Analyze this clinical document:\n\n{text[:8000]}"}],
        max_tokens=2500,
    )
    cleaned = _clean_json(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON. Raw output:\n{cleaned}") from e

    return _parse_result(data)
