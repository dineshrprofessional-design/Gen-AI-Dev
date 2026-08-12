---
page_id: client-send
sdk_version: v3
page_type: reference
title: Client.send()
---

# Client.send()

Sends a single request to the configured endpoint and blocks until a response is
received or the retry budget is exhausted. The method is safe to call from
multiple threads; each call takes its own connection from the pool and returns
it on completion. Callers issuing many requests at once should prefer
`Client.send_batch()`, which amortises connection setup across the whole batch
and applies a shared rate limiter. When the endpoint returns a 5xx status the
client retries automatically according to the backoff parameters below; 4xx
statuses are surfaced to the caller immediately and are never retried, because a
malformed request does not become well-formed by being sent again.

## Parameters

| name             | type | default   | required | description                                          |
|------------------|------|-----------|----------|------------------------------------------------------|
| payload          | dict | —         | yes      | The request body, serialised as JSON before sending. |
| timeout_ms       | int  | 30000     | no       | Total budget for the call, including every retry.    |
| retry_backoff_ms | int  | 250       | no       | Base delay between retries, before jitter is applied.|
| max_retries      | int  | 3         | no       | Attempts after the first before the call gives up.   |
| idempotency_key  | str  | None      | no       | Deduplicates retries server-side within 24 hours.    |
| compress         | bool | False     | no       | Gzip the request body before transmission.           |
| headers          | dict | None      | no       | Extra headers merged over the client defaults.       |
| base_url         | str  | None      | no       | Overrides the client base URL for this call only.    |
| stream           | bool | False     | no       | Return a generator instead of a materialised body.   |
| verify_tls       | bool | True      | no       | Validate the server certificate chain.               |
| proxy            | str  | None      | no       | HTTP proxy URL applied to this call only.            |
| user_agent       | str  | None      | no       | Overrides the default SDK user agent string.         |
| retry_on         | list | [500,502] | no       | Status codes that trigger the automatic retry path.  |
| max_redirects    | int  | 5         | no       | Redirects followed before RedirectLoopError is set.  |
| pool_timeout_ms  | int  | 5000      | no       | Time spent waiting for a free pooled connection.     |
| trace_id         | str  | None      | no       | Propagated to the server for distributed tracing.    |

## Example

```python
from sdk import Client

client = Client(api_key="sk-...")

response = client.send(
    payload={"query": "hello"},
    retry_backoff_ms=500,
    max_retries=5,
    idempotency_key="req-0001",
    retry_on=[500, 502, 503],
)

print(response.status_code)
print(response.json())
```

## Notes

The `retry_backoff_ms` value is the base delay, not the total delay. The client
applies full jitter, so the actual wait before attempt *n* is drawn at random
from `[0, retry_backoff_ms * 2 ** n)`. A base of 250 with five retries can
therefore wait up to eight seconds in the worst case. Set `timeout_ms` with that
ceiling in mind, since the timeout covers the entire call including every retry
attempt rather than each attempt individually.
