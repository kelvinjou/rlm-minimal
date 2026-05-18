from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import DEFAULT_ONTOLOGY_PROMPT, load_ttl_context
from rlm.utils.llm import create_llm_client


FLAT_ONTOLOGY_SYSTEM_PROMPT = """You are answering a user query using a Turtle/TTL ontology.

Treat the ontology as a knowledge graph. Ground your answer in explicit ontology
evidence such as classes, instances, labels, comments, subclass relations,
object properties, datatype properties, domains, ranges, and assertions.

For XR design queries, return a concise structured answer with:
- interpreted scenario requirements
- high-priority recommendations grouped by interaction techniques, UI
  components, hardware considerations, design principles, and evaluation methods
- evidence from the ontology for each recommendation
- tradeoffs or conflicts
- gaps where the ontology lacks enough information

Clearly distinguish explicit ontology evidence from your own inference. Do not
use tools, code, REPL syntax, or FINAL(...) wrappers. Answer directly."""


def build_flat_messages(context: str, query: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": FLAT_ONTOLOGY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Ontology TTL:\n"
                "```turtle\n"
                f"{context}\n"
                "```\n\n"
                f"User query:\n{query.strip()}"
            ),
        },
    ]


def main() -> None:
    print("Example of using a flat LLM with an ontology-backed XR knowledge graph.")

    ttl_path = "outputs/enhanced_xr.ttl"
    query = DEFAULT_ONTOLOGY_PROMPT

    client_backend = "nvidia"
    model = "moonshotai/kimi-k2.6"
    request_timeout = 500.0
    max_retries = 0
    max_tokens = 3000

    context = load_ttl_context(ttl_path)
    llm = create_llm_client(
        provider=client_backend,
        model=model,
        timeout=request_timeout,
        max_retries=max_retries,
    )

    result = llm.completion(
        build_flat_messages(context=context, query=query),
        max_tokens=max_tokens,
    )

    print(f"Result: {result}")
    print("Last usage:", llm.last_usage)
    print("Prompt tokens:", llm.total_prompt_tokens)
    print("Completion tokens:", llm.total_completion_tokens)
    print("Total tokens:", llm.total_tokens)


if __name__ == "__main__":
    main()
