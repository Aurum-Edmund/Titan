# Titan80M Conversational Schema (S1, S2, S4, S5, S6, S7)

Each JSONL record stores a single example as `{"text": "<payload>"}` where `<payload>` is a newline-joined string containing Titan chat control tokens.

## Token Order
- Required prefix: `<|bos|><|system|>\n<system message>`
- Followed by alternating user and assistant (or agent) blocks.
- Each block begins with its role token on its own line (`<|user|>`, `<|assistant|>`, `<|agent|>`, `<|action|>`, `<|observation|>`, `<|result|>`, `<|tool_error|>`).
- Close every record with a trailing `<|eos|>` on its own line.

## Masking Rules
- Loss is applied only to assistant-facing spans: `<|assistant|>`, `<|agent|>`, `<|action|>`, `<|result|>`, and `<|tool_error|>` content.
- Mask system, user, observation, and `<|eos|>` tokens. Pad tokens are masked as well.

## Validation Checks
- Exactly one `<|bos|>` and one `<|eos|>` per sample.
- Role ordering: system block first, then user/assistant alternation (agentic blocks may interleave actions and observations).
- For multistep conversations ensure final speaker is the assistant or agent result.

## Length Buckets
- Short: 20–60 tokens (45 %)
- Medium: 60–180 tokens (40 %)
- Long: 180–400 tokens (15 %)

## Safety Notes
- Conversations must remain culturally neutral, respectful, and avoid real-person impersonations.
- Instruction, safety, and creative slices reuse the same container format with slice-specific content rules described in `sampling_config_v1.yaml`.
