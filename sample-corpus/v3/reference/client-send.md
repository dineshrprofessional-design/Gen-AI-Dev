# Client.send()

No front matter at all. Everything below must be derived: `sdk_version` from
the `v3` folder, `page_type` from the `reference` folder, `page_id` from the
filename, and the title from this heading.

| name             | type | default | required |
|------------------|------|---------|----------|
| payload          | dict | —       | yes      |
| retry_backoff_ms | int  | 250     | no       |
