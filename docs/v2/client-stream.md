---
page_id: client-stream
sdk_version: v2
page_type: reference
title: Client.stream()
---

# Client.stream()

Opens a streaming connection and yields events as they arrive. In v2 the
generator does not reconnect: a dropped connection raises `StreamClosed` and the
caller must start again from the beginning. Resumable streams arrived in v3.

## Parameters

| name         | type | default | required | description                                     |
|--------------|------|---------|----------|--------------------------------------------------|
| payload      | dict | —       | yes      | The request body.                               |
| chunk_size   | int  | 512     | no       | Bytes read from the socket per iteration.       |
| heartbeat_ms | int  | 30000   | no       | Interval between server keep-alive frames.      |

## Example

```python
events = client.stream(payload={"query": "hello"})
for event in events:
    print(event.text, end="")
events.close()
```

## Notes

The default `chunk_size` in v2 is 512 bytes. It was raised to 1024 in v3 after
profiling showed the smaller reads dominating CPU time on large responses.

There is no context-manager form in v2 — you must call `close()` yourself, and
forgetting to do so leaks the connection until the process exits.
