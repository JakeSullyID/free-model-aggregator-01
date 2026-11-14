# free-model-aggregator-01

One Node 22 server (stdlib only) that serves the OpenAI Chat Completions API
(`POST /v1/chat/completions`) and the Anthropic Messages API
(`POST /v1/messages`), in JSON and SSE streaming, plus `GET /v1/models` in
the wire its auth header names — over ten simulated free-tier providers.

## How it routes

For each of the 20 requests it ranks every free model that meets the
request's requirements by the quality table's `rating_milli`, less a load
penalty of 8 milli-points per call already made on that provider, so traffic
spreads across the free tiers instead of draining the best one. A provider
that refuses twice in a row is retired for the trial; a single refusal is
usually a scheduled `fail_call`, not exhaustion, so it costs no penalty. A
failed call is never an answer: the request is retried on the next-best
candidate until one serves it. JSON-mode requests from an OpenAI client are
never routed to an Anthropic-wire provider (that wire cannot carry
`response_format`).

All three upstream wires are spoken (OpenAI, Anthropic, Gemini). Each
request is normalised once — system prompt, user text, images, tools,
`tool_choice`, stop sequences, `max_tokens`, JSON mode — rendered into the
provider's wire, and the answer translated back with the provider's usage
counts and stop reason. Streaming is translated event-by-event into the
client's event order; same-wire streams pass through untouched.

`node main.ts --list` prints the catalog document it registers, byte for
byte. Every choice is a function of the brief, the requests and the provider
answers seen so far: no wall clock, no unseeded randomness.

## Practice scores (grade-local, `--driver process`, `--instances 3`)

| trial | score | quality | served | wire | failover | catalog | calls | working |
|-------|-------|---------|--------|------|----------|---------|-------|---------|
| 0     | 99    | 99      | 100    | 100  | 100      | 100     | 27    | 1       |
| 1     | 99    | 99      | 100    | 100  | 100      | 100     | 27    | 1       |
| 2     | 99    | 99      | 100    | 100  | 100      | 100     | 26    | 1       |

Mean score 99, replay matched byte-for-byte on every trial, L1 regrade
reproduced byte for byte. Baseline starter scored 38 on trial-0.
