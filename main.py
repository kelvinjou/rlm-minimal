from pathlib import Path

from rlm.rlm_repl import RLM_REPL


# Previous flat long-context demo. Kept here as a reference while the entrypoint
# moves to ontology-backed retrieval over a TTL file.
#
# import random
#
# def generate_massive_context(num_lines: int = 1_000_000, answer: str = "1298418") -> str:
#     print("Generating massive context with 1M lines...")
#     random_words = ["blah", "random", "text", "data", "content", "information", "sample"]
#     lines = []
#     for _ in range(num_lines):
#         num_words = random.randint(3, 8)
#         line_words = [random.choice(random_words) for _ in range(num_words)]
#         lines.append(" ".join(line_words))
#     magic_position = random.randint(400000, 600000)
#     lines[magic_position] = f"The magic number is {answer}"
#     print(f"Magic number inserted at position {magic_position}")
#     return "\n".join(lines)
#
# def main():
#     print("Example of using RLM (REPL) with NVIDIA DeepSeek V4 Pro on a needle-in-haystack problem.")
#     answer = str(random.randint(1000000, 9999999))
#     context = generate_massive_context(num_lines=1_000_000, answer=answer)
#     rlm = RLM_REPL(
#         client_backend="nvidia",
#         recursive_client_backend="nvidia",
#         model="moonshotai/kimi-k2.6",
#         recursive_model="moonshotai/kimi-k2.6",
#         enable_logging=True,
#         log_to_file=False,
#         log_dir="logs",
#         max_iterations=3,
#         depth=2
#     )
#     query = "I'm looking for a magic number in this context. What is it?"
#     result = rlm.completion(context=context, query=query)
#     print(f"Result: {result}. Expected: {answer}")


DEFAULT_ONTOLOGY_PROMPT = """
Design an XR training scenario for surgeons practicing precise manipulation in a constrained physical room, with low sickness risk.
"""


def load_ttl_context(ttl_path: str) -> str:
    path = Path(ttl_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"TTL file not found: {path}")
    if not path.is_file():
        raise ValueError(f"TTL path is not a file: {path}")
    return path.read_text(encoding="utf-8")


def main() -> None:
    print("Example of using RLM (REPL) with an ontology-backed XR knowledge graph.")

    ttl_path = "outputs/enhanced_xr.ttl"
    query = DEFAULT_ONTOLOGY_PROMPT
    smoke_test = False

    client_backend = "nvidia"
    recursive_client_backend = "nvidia"
    model = "moonshotai/kimi-k2.6"
    recursive_model = "moonshotai/kimi-k2.6"
    max_iterations = 3
    enable_logging = True
    log_to_file = True
    request_timeout = 500.0
    max_retries = 0
    max_tokens = 3000
    max_format_retries = 1
    force_final_answer_on_max_iterations = True

    context = load_ttl_context(ttl_path)
    prompt_mode = "ontology"
    if smoke_test:
        context = load_ttl_context(ttl_path)
        query = DEFAULT_ONTOLOGY_PROMPT
        prompt_mode = "ontology"

    rlm = RLM_REPL(
        client_backend=client_backend,
        recursive_client_backend=recursive_client_backend,
        model=model,
        recursive_model=recursive_model,
        enable_logging=enable_logging,
        log_to_file=log_to_file,
        log_dir="logs",
        max_iterations=max_iterations,
        prompt_mode=prompt_mode,
        request_timeout=request_timeout,
        max_retries=max_retries,
        max_tokens=max_tokens,
        max_format_retries=max_format_retries,
        force_final_answer_on_max_iterations=force_final_answer_on_max_iterations,
    )

    result = rlm.completion(context=context, query=query)
    print(f"Result: {result}")

    print("Last usage:", rlm.llm.last_usage)
    print("Prompt tokens:", rlm.llm.total_prompt_tokens)
    print("Completion tokens:", rlm.llm.total_completion_tokens)
    print("Total tokens:", rlm.llm.total_tokens)


if __name__ == "__main__":
    main()
