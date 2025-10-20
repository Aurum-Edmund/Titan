# Titan80M Agentic Copilot Schema (S3)

Agentic traces extend the conversational schema with structured tool calls. Every record remains a single `{"text": "<payload>"}` line.

## Structure
1. `<|bos|><|system|>` system primer (one or two lines).
2. `<|user|>` problem statement.
3. Repeated tool loop consisting of:
   - `<|agent|>` high-level plan or reflection.
   - `<|action|>` JSON payload describing the tool invocation.
   - `<|observation|>` tool feedback (plain text).
   - Optional `<|tool_error|>` when a call fails (retry one time with scoped action).
4. `<|result|>` concise wrap-up.
5. `<|eos|>` terminator.

## Tool Payload Contract
- `<|action|>` content must be valid minified JSON with keys `tool` and `args`.
- Supported tool names: `run_tests`, `execute`, `edit_file`, `search_docs`, `math_solve`, `http_get`, `fs_glob`, `lint_code`.
- Arguments object should contain only serialisable literals; prefer `cmd`, `path`, `query`, `url`, `pattern`, etc.

## Masking
- Apply loss to `<|agent|>`, `<|action|>`, `<|result|>`, and `<|tool_error|>` bodies.
- Mask `<|observation|>` spans along with system, user, and `<|eos|>`.

## Validation
- Ensure JSON in action blocks parses.
- Disallow consecutive `<|action|>` without an `<|observation|>` in between.
- Enforce that the final block before `<|eos|>` is `<|result|>`.

## Sharding
- Target shard size: 5k–20k rows.
- Maintain 96 / 2 / 2 split across train/val/test with no near-duplicate prompts between splits (SimHash Hamming ≤ 3).
