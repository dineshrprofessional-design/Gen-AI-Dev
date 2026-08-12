---
page_id: client-configure
sdk_version: v2
page_type: reference
title: Client.configure()
---

# Client.configure()

Sets client-wide defaults inherited by every subsequent call. Options passed to
an individual call override these.

## Parameters

| name               | type | default        | required | description                                    |
|--------------------|------|----------------|----------|-------------------------------------------------|
| api_key            | str  | —              | yes      | Credential used for every request.             |
| base_url           | str  | https://api.example.com | no | Endpoint root for all calls.            |
| pool_size          | int  | 4              | no       | Maximum simultaneous pooled connections.       |
| default_timeout_ms | int  | 10000          | no       | Timeout applied when a call does not set one.  |
| log_level          | str  | error          | no       | One of debug, info, warning, error.            |

## Example

```python
client = Client()
client.configure(api_key="sk-...", pool_size=8, default_timeout_ms=20000)
```

## Notes

There is no `retries_enabled` switch in v2 because there is no automatic retry
path to switch off. There is no connection reaper either: pooled connections
live until the client is garbage collected.

The v2 default `pool_size` is 4, raised to 10 in v3.
