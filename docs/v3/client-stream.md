---
page_id: client-stream
sdk_version: v3
page_type: reference
title: Client.stream()
---

# Client.stream()

Opens a streaming connection and yields events as they arrive. The generator
holds its connection open for the lifetime of the iteration, so callers must
either exhaust it or close it explicitly.

## Parameters

| name         | type | default | required |
|--------------|------|---------|----------|
| payload      | dict | —       | yes      |
| chunk_size   | int  | 1024    | no       |
| heartbeat_ms | int  | 15000   | no       |

## Example

```python
with client.stream(payload={"query": "hello"}) as events:
    for event in events:
        print(event.text, end="")
```
