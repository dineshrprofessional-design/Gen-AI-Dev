"""Grounded answer generation with forced refusal.

Uses xAI's Grok through its OpenAI-compatible endpoint. The key is read from
`XAI_API_KEY`; without it, retrieval still works and generation reports that it
is unconfigured rather than failing obscurely.

## The refusal must be forced, not suggested

The single most damaging line you can put in a grounding prompt is
*"if the context is insufficient, use your best judgement"* — that sentence is
how a plausible, non-existent parameter ends up in someone's production code.

So the contract here is mechanical: the model answers **only** from the
supplied chunks, and when they do not contain the answer it must emit exactly
`INSUFFICIENT_CONTEXT` and nothing else. There is a score gate in front of the
model as well, so a query that retrieves nothing relevant is refused before a
token is ever generated.
"""

import os
import re
from dataclasses import dataclass, field

from app.rag.store import SearchHit

XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = os.environ.get("XAI_MODEL", "grok-4")

REFUSAL_TOKEN = "INSUFFICIENT_CONTEXT"

# A coarse backstop, NOT the refusal mechanism.
#
# Measured on this corpus with bge-m3, top-1 cosine similarity:
#   in-corpus  questions: 0.547 - 0.734
#   out-of-corpus        : 0.422 - 0.529
#
# The distributions separate by 0.018. That is far too narrow to threshold
# reliably — everything in the corpus is SDK documentation, so an unrelated SDK
# question still looks similar. A floor tuned to catch all five out-of-corpus
# samples (0.54) would wrongly refuse a real question phrased slightly off.
#
# So the floor sits below the lowest real question instead, catching only
# obviously unrelated queries. The grounding prompt is the primary gate; a
# false refusal costs more than a question the model then declines itself.
SCORE_FLOOR = 0.45

SYSTEM_PROMPT = f"""\
You answer questions about an SDK using ONLY the numbered context chunks given \
to you. You have no other knowledge of this SDK.

Rules, in order of priority:

1. If the context chunks do not contain the answer, reply with exactly this \
and nothing else:
{REFUSAL_TOKEN}

2. Never use knowledge from outside the context chunks. Never guess a parameter \
name, a default value, or a type. If a value is not written in the context, it \
is not available to you.

3. Every factual claim must end with a citation naming the chunk it came from, \
in square brackets, exactly as the chunk id is written. For example: \
The default is 250 ms [v3/client-send#parameters::1].

4. Do not cite a chunk id that was not given to you.

5. Be brief. Answer the question asked, then stop.
"""

_CITATION = re.compile(r"\[([^\]\s]+::\d+)\]")


@dataclass
class Answer:
    """The result of one grounded generation."""

    question: str
    text: str
    refused: bool
    reason: str = ""
    citations: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    context_ids: list[str] = field(default_factory=list)
    model: str = ""
    top_score: float = 0.0

    @property
    def citations_valid(self) -> bool:
        """Every cited chunk id was actually in the context."""
        return not self.invalid_citations


class GenerationUnavailable(RuntimeError):
    """No API key configured. Retrieval still works; generation does not."""


def is_configured() -> bool:
    return bool(os.environ.get("XAI_API_KEY"))


def _client():
    key = os.environ.get("XAI_API_KEY")
    if not key:
        raise GenerationUnavailable(
            "XAI_API_KEY is not set. Retrieval works without it; "
            "generation needs a key from https://console.x.ai"
        )
    from openai import OpenAI

    return OpenAI(api_key=key, base_url=XAI_BASE_URL)


def list_models() -> list[str]:
    """Ask xAI which models this key can use, rather than guessing a name."""
    return sorted(m.id for m in _client().models.list().data)


def build_context(hits: list[SearchHit]) -> str:
    """Render retrieved chunks with their ids, so the model can cite them."""
    blocks = []
    for hit in hits:
        blocks.append(f"[{hit.chunk_id}]\n{hit.text}")
    return "\n\n---\n\n".join(blocks)


def answer(
    question: str,
    hits: list[SearchHit],
    model: str = DEFAULT_MODEL,
    score_floor: float = SCORE_FLOOR,
) -> Answer:
    """Answer strictly from `hits`, or refuse."""
    context_ids = [h.chunk_id for h in hits]
    top_score = hits[0].score if hits else 0.0

    # Gate one: nothing retrieved, or nothing retrieved well enough. Refuse
    # before the model is involved — no prompt can argue its way past this.
    if not hits or top_score < score_floor:
        return Answer(
            question=question,
            text=REFUSAL_TOKEN,
            refused=True,
            reason=(
                "no chunks retrieved"
                if not hits
                else f"top score {top_score:.3f} below floor {score_floor:.2f}"
            ),
            context_ids=context_ids,
            top_score=top_score,
        )

    # Gate two: the model itself, under a prompt that forbids outside knowledge
    # and mandates the exact refusal token.
    completion = _client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context chunks:\n\n{build_context(hits)}\n\n"
                f"Question: {question}",
            },
        ],
    )
    text = (completion.choices[0].message.content or "").strip()

    refused = text.strip().upper().startswith(REFUSAL_TOKEN)
    cited = list(dict.fromkeys(_CITATION.findall(text)))

    return Answer(
        question=question,
        text=text,
        refused=refused,
        reason="model judged the context insufficient" if refused else "",
        citations=cited,
        # A citation the model invented is worse than no citation: it looks
        # verifiable and isn't. Flag them rather than trusting the output.
        invalid_citations=[c for c in cited if c not in context_ids],
        context_ids=context_ids,
        model=model,
        top_score=top_score,
    )
