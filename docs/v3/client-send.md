---
page_id: client-send
sdk_version: v3
page_type: reference
title: Client.send()
---

# Client.send()

Sends a single request to the configured endpoint and blocks until a response is
received or the retry budget is exhausted. When the endpoint returns a 5xx
status the client retries automatically; 4xx statuses are surfaced to the caller
immediately and are never retried.

## Parameters

| name             | type | default | required |
|------------------|------|---------|----------|
| payload          | dict | —       | yes      |
| timeout_ms       | int  | 30000   | no       |
| retry_backoff_ms | int  | 250     | no       |
| max_retries      | int  | 3       | no       |

## Example

```python
response = client.send(payload={"query": "hello"}, retry_backoff_ms=500)
```

## Notes

The `retry_backoff_ms` value is the base delay, not the total delay. Full jitter
is applied, so the wait before attempt *n* is drawn from
`[0, retry_backoff_ms * 2 ** n)`.
