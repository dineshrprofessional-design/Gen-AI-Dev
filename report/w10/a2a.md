# Week 10 bonus: AgentCard and the A2A task lifecycle

## The AgentCard

`report/w10/agentcard.json` is the card the orchestrator would advertise:

- **one skill**, `answer-sdk-question`: a question in, an answer out in the
  ANSWER / VERSION / CITATIONS contract
- **input/output modes**: `text/plain`
- **auth**: a bearer token

It is a design. The build in `app/rag/week10_squad.py` serves its workers over
plain local HTTP, not A2A, and has no authentication. A deployment would need
both before exposing the squad.

## Mapping the failed case onto the A2A task lifecycle

A2A tasks move through `submitted → working`, then end in a terminal state
(`completed`, `failed`, `canceled`, `rejected`) or pause for the caller
(`input-required`, `auth-required`).

| | |
|---|---|
| what our orchestrator did on `w6-019` | it caught the HTTP 500 and returned `INSUFFICIENT_CONTEXT` as a normal answer. In A2A terms that is a **`completed`** task whose artifact says "no answer", the wrong state, because a caller cannot tell "the docs do not say" from "a worker was down" |
| what it should have been | **`failed`**, with the worker's 500 in the status message |
| why not `input-required` | `input-required` fits when the developer can supply what is missing. Here the question already says what is needed ("between v2 and v3"), so asking "which SDK version are you on?" adds nothing; the missing piece is a broken dependency, not user input. |
| when `input-required` *is* right | for a question that names no version while the version worker is down. Pausing to ask the developer for their SDK version would replace exactly what the failed worker was meant to provide, and is better than guessing v3. |

## What A2A buys over a plain REST call (two lines)

A2A turns a worker call into a task with an id and a standard lifecycle, so a 500
surfaces as a `failed` (or `input-required`) task the orchestrator has to handle,
rather than an HTTP error it can quietly paper over with "insufficient context".
And the AgentCard makes each agent's skills, modes and auth discoverable, the way
`tools/list` did for MCP tools, instead of hard-coding every worker's endpoint
contract.
