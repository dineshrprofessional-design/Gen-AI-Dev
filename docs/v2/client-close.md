---
page_id: client-close
sdk_version: v2
page_type: reference
title: Client.close()
---

# Client.close()

Closes every pooled connection. In v2 this is unconditional: in-flight requests
are cancelled rather than drained, and there is no timeout to wait on. Graceful
draining arrived in v3.

## Parameters

| name  | type | default | required | description                              |
|-------|------|---------|----------|-------------------------------------------|
| force | bool | True    | no       | Cancel in-flight requests. Always true.  |

## Example

```python
client = Client(api_key="sk-...")
client.send(payload={"query": "hello"})
client.close()
```

## Notes

`force` defaults to true in v2 and setting it to false has no effect — the
parameter exists only so that code written against the v3 preview does not fail
to import. In v3 the default is false and requests are drained.

There is no context-manager form in v2.
