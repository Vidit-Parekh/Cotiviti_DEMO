"""
src/nlp/summarizer.py
Clinical document summarization using an LLM.
Produces concise plain-language summaries suitable for payment integrity analysts.
"""

import re
from src.llm.prompts import SUMMARIZATION_PROMPT


def _strip_reasoning(text: str) -> str:
    """Remove DeepSeek-R1 <think>...</think> reasoning blocks."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def summarize(text: str, llm_chat_fn, style: str = "clinical") -> str:
    """
    Generate a plain-language summary of a clinical document.

    Args:
        text: Raw clinical note text.
        llm_chat_fn: Callable matching llm_chat(system, messages, max_tokens) -> str.
        style: Summary style.
                "clinical"   - 2 to 3 sentence summary for clinicians.
                "executive"  - 1 sentence business-level summary.
                "coding"     - Summary focused on documentation for coding purposes.

    Returns:
        Summary string.
    """
    style_instructions = {
        "clinical": (
            "Write a 2 to 3 sentence plain-language clinical summary of this document. "
            "Focus on the primary diagnosis, key findings, and disposition."
        ),
        "executive": (
            "Write a single sentence executive summary of this clinical document "
            "suitable for a healthcare analytics dashboard."
        ),
        "coding": (
            "Write a 2 to 3 sentence summary of this clinical document focused on "
            "diagnoses, procedures, and medications relevant to medical coding and billing."
        ),
    }

    instruction = style_instructions.get(style, style_instructions["clinical"])
    prompt = f"{SUMMARIZATION_PROMPT}\n\n{instruction}"

    raw = llm_chat_fn(
        prompt,
        [{"role": "user", "content": f"Summarize this clinical document:\n\n{text[:6000]}"}],
        max_tokens=300,
    )
    return _strip_reasoning(raw)


def summarize_section(section_text: str, section_name: str, llm_chat_fn) -> str:
    """
    Summarize a specific section of a clinical document (e.g. Assessment and Plan).

    Args:
        section_text: The text of the section.
        section_name: Name of the section (e.g. "Assessment and Plan").
        llm_chat_fn: LLM callable.

    Returns:
        One to two sentence summary of the section.
    """
    raw = llm_chat_fn(
        "You are a clinical AI assistant. Summarize the provided clinical document section "
        "in one to two sentences. Be concise and factual. Return only the summary.",
        [{
            "role": "user",
            "content": f"Section: {section_name}\n\n{section_text[:3000]}"
        }],
        max_tokens=200,
    )
    return _strip_reasoning(raw)
