---
page_id: client-send
sdk_version: v2
page_type: reference
title: Client.send()
---

# Client.send()

Sends a single request to the configured endpoint and blocks until a response
is received. Retries are not performed automatically in v2; callers implement
their own retry loop around the call.

## Parameters

| name             | type | default | required |
|------------------|------|---------|----------|
| payload          | dict | —       | yes      |
| timeout_ms       | int  | 10000   | no       |
| retry_backoff_ms | int  | 100     | no       |

## Notes

The `retry_backoff_ms` parameter is accepted in v2 but ignored. Automatic
retries arrived in v3.
