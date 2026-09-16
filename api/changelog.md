# Changelog — v2 to v3

GENERATED — do not edit. Reproduce with:

    python -m app.rag.week7_corpus

Every entry below is computed from the parameter tables in `docs/`. The HTTP
operation bindings and deprecations referenced here are authored — see
`api/README.md`.

## Changed (default value)

| method | parameter | v2 | v3 |
| --- | --- | --- | --- |
| `Client.close()` | `force` | `True` | `False` |
| `Client.configure()` | `default_timeout_ms` | `10000` | `30000` |
| `Client.configure()` | `log_level` | `error` | `warning` |
| `Client.configure()` | `pool_size` | `4` | `10` |
| `Client.send()` | `retry_backoff_ms` | `100` | `250` |
| `Client.send()` | `timeout_ms` | `10000` | `30000` |
| `Client.stream()` | `chunk_size` | `512` | `1024` |
| `Client.stream()` | `heartbeat_ms` | `30000` | `15000` |

## Added (parameter)

- `Client.close()` — `close_sessions`, `drain`, `timeout_ms`
- `Client.configure()` — `pool_reap_seconds`, `retries_enabled`, `telemetry`
- `Client.send()` — `base_url`, `compress`, `headers`, `idempotency_key`, `max_redirects`, `max_retries`, `pool_timeout_ms`, `proxy`, `retry_on`, `stream`, `trace_id`, `user_agent`, `verify_tls`

## Added (method)

- `Client.send_batch()` — new in v3, 8 parameters
- `Client.upload()` — new in v3, 7 parameters

## Removed

**Removed: none.** v3 adds to v2 and takes nothing away; every v2 parameter is present in v3 with the same type, and no v2 method was dropped. That is computed, not asserted — and it is why the deprecations in `api/deprecations.json` are all at the HTTP layer and all authored. See `api/README.md`.

