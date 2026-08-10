---
page_id: client-configure
sdk_version: v3
page_type: reference
title: Client.configure()
---

# Client.configure()

Sets client-wide defaults that every subsequent call inherits. Options passed
to an individual `send()` or `stream()` call override these. Calling
`configure()` does not affect requests already in flight.

## Parameters

| name              | type | default        | required | description                                        |
|-------------------|------|----------------|----------|-----------------------------------------------------|
| api_key           | str  | —              | yes      | Credential used for every request.                 |
| base_url          | str  | https://api.example.com | no | Endpoint root for all calls.                |
| pool_size         | int  | 10             | no       | Maximum simultaneous pooled connections.           |
| pool_reap_seconds | int  | 90             | no       | Idle time before a pooled connection is collected. |
| default_timeout_ms| int  | 30000          | no       | Timeout applied when a call does not set one.      |
| retries_enabled   | bool | True           | no       | Master switch for the automatic retry path.        |
| log_level         | str  | warning        | no       | One of debug, info, warning, error.                |
| telemetry         | bool | True           | no       | Send anonymous usage counters to the vendor.       |

## Example

```python
client = Client()
client.configure(
    api_key="sk-...",
    pool_size=32,
    default_timeout_ms=45000,
    telemetry=False,
)
```

## Notes

`pool_size` is a ceiling, not a reservation — connections are created lazily
and reaped after `pool_reap_seconds` of idleness. Setting it far above your
real concurrency costs nothing but does not help either.

Setting `retries_enabled` to false disables the automatic retry path entirely.
The per-call `retry_backoff_ms` and `max_retries` parameters are then accepted
but ignored, which mirrors how v2 behaved.
