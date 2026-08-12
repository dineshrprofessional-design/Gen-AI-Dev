---
page_id: client-close
sdk_version: v3
page_type: reference
title: Client.close()
---

# Client.close()

Releases every pooled connection and stops the background reaper thread. Safe
to call more than once; subsequent calls are no-ops. A client that is not
closed leaks its pool until the interpreter exits, which matters in long-lived
processes and in tests that construct many clients.

## Parameters

| name           | type | default | required | description                                       |
|----------------|------|---------|----------|---------------------------------------------------|
| timeout_ms     | int  | 5000    | no       | Time allowed for in-flight requests to finish.    |
| force          | bool | False   | no       | Cancel in-flight requests instead of draining.    |
| drain          | bool | True    | no       | Wait for in-flight requests before closing.       |
| close_sessions | bool | True    | no       | Also close sessions created by this client.       |

## Example

```python
client = Client(api_key="sk-...")
try:
    client.send(payload={"query": "hello"})
finally:
    client.close(timeout_ms=2000)

# Or as a context manager, which closes on exit:
with Client(api_key="sk-...") as client:
    client.send(payload={"query": "hello"})
```

## Notes

With `force=True` in-flight requests are cancelled rather than drained, and
their futures raise `CancelledError`. Prefer the default unless the process is
shutting down and latency matters more than completeness.

If `drain` is true and `timeout_ms` elapses before in-flight requests finish,
the remaining connections are closed anyway and a warning is logged. Close is
never allowed to block indefinitely.
