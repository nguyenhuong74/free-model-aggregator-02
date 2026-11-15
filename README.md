# free-model-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/nguyenhuong74/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/nguyenhuong74/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

A live free-model aggregator: one server that answers the OpenAI Chat Completions
API and the Anthropic Messages API — in JSON and in server-sent events — with real
completions from the best-rated free OpenRouter model that meets each request,
dialled through the trial gateway. It ranks free models by the brief's published
quality rating, fails over across models when one refuses or answers without the
token the prompt asked for, and reports every recovered refusal on its responses.

## Run it

```sh
node main.ts
```

Node 22 or later, standard library only — nothing to install. The trial sets
`MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE`, `OPENAI_BASE_URL` and the trial key in
`OPENAI_API_KEY`; the server picks a free port, registers it with the env, serves
the two wires, and exits 0 when the env says the work is done.

```sh
node main.ts --list   # print the catalog document and exit 0
```

## How it works

- **Routing.** For each request, the candidates are the brief's free models that
  meet the prompt's requirements (JSON mode, tools, vision, context), ranked by
  the house's published `rating_milli`. The best-rated model serves first.
- **Failover.** A refused gateway call (4xx/5xx, or a connection that never
  answered) retries on the next-ranked model, up to five attempts per prompt,
  and every refusal is reported in the `x-makefun-upstream-failures` response
  header. A model that refused, or answered without the token the prompt asked
  for, is demoted 100 milli-points per failure for the rest of the trial, so a
  rate-limited model stops taking later prompts down with it.
- **Both wires, both shapes.** `POST /v1/chat/completions` and
  `POST /v1/messages`, JSON or SSE. Anthropic requests are translated to Chat
  Completions upstream and the answers translated back, with the stop reason and
  usage carried whole. `GET /v1/models` answers in the shape its auth header
  names: `Authorization: Bearer` gets the OpenAI list, `x-api-key` the
  Anthropic one.
- **The model rule.** Every response's `model` field names the model actually
  dialled — the provider's spelling (without the gateway's `openrouter/` prefix)
  is replaced with the dialled id on every answer and every streamed chunk.
- **Determinism.** Every choice derives from the brief, the requests and the
  gateway's answers: no wall clock and no unseeded randomness, so a replay
  makes the same calls and leaves the same state.

## Layout

```text
main.ts                  the server: routes, routing, failover, register, poll
lib/brief.ts             the brief's model list and prompts, and the catalog
lib/wire.ts              the two client wires, the upstream wire, SSE rendering
vendor/makefun-agent.ts  the kit's agent SDK, vendored byte for byte
```

## License

MIT — see [LICENSE](LICENSE).

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
