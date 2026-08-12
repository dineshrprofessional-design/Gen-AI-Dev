---
page_id: client-batch
sdk_version: v3
page_type: reference
title: Client.send_batch()
---

# Client.send_batch()

Sends many requests over a shared connection pool and returns their responses
in submission order. Connection setup is amortised across the whole batch and a
single rate limiter governs the group, which makes this substantially cheaper
than calling `Client.send()` in a loop. Individual failures do not abort the
batch: each response carries its own status, and the caller decides what to
retry.

## Parameters

| name            | type | default | required | description                                          |
|-----------------|------|---------|----------|------------------------------------------------------|
| payloads        | list | —       | yes      | Request bodies, sent in the order given.             |
| concurrency     | int  | 8       | no       | Requests in flight at once across the batch.         |
| timeout_ms      | int  | 60000   | no       | Budget for the whole batch, not per request.         |
| fail_fast       | bool | False   | no       | Abort remaining requests after the first failure.    |
| ordered         | bool | True    | no       | Return responses in submission order.                |
| rate_limit_rps  | int  | 20      | no       | Requests per second ceiling for this batch.          |
| chunk_bytes     | int  | 1048576 | no       | Split oversized payload groups at this boundary.     |
| progress_every  | int  | 0       | no       | Emit a progress callback every N responses; 0 is off.|

## Example

```python
responses = client.send_batch(
    payloads=[{"query": q} for q in queries],
    concurrency=16,
    rate_limit_rps=50,
    fail_fast=False,
)

failed = [r for r in responses if r.status_code >= 500]
print(f"{len(failed)} of {len(responses)} need a retry")
```

## Notes

`concurrency` and `rate_limit_rps` interact. The client honours whichever is
more restrictive at any moment, so setting concurrency to 64 against a limit of
20 requests per second gives you 20 requests per second, not 64. Raising
`concurrency` alone will not increase throughput past the rate limit.

`timeout_ms` covers the entire batch. A batch of 500 payloads at the default
60000 ms gives each request roughly 120 ms of budget on average, which is
usually too tight — scale the timeout with the batch size.
