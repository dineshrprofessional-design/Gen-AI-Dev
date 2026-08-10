---
page_id: client-upload
sdk_version: v3
page_type: reference
title: Client.upload()
---

# Client.upload()

Uploads a file and returns a handle that can be referenced by later requests.
Files above `chunk_bytes` are sent as a multipart upload and reassembled
server-side; below it, a single request is used. The handle is valid for 24
hours unless `retain_hours` says otherwise.

## Parameters

| name          | type | default  | required | description                                        |
|---------------|------|----------|----------|-----------------------------------------------------|
| path          | str  | —        | yes      | Local path of the file to upload.                  |
| content_type  | str  | None     | no       | Overrides the type guessed from the extension.     |
| chunk_bytes   | int  | 8388608  | no       | Multipart threshold and part size, in bytes.       |
| retain_hours  | int  | 24       | no       | How long the handle stays valid.                   |
| checksum      | bool | True     | no       | Verify a SHA-256 checksum after upload.            |
| resume        | bool | True     | no       | Resume an interrupted multipart upload.            |
| max_bytes     | int  | 524288000| no       | Reject files larger than this before sending.      |

## Example

```python
handle = client.upload(
    path="./corpus/manual.pdf",
    chunk_bytes=16777216,
    retain_hours=72,
)

client.send(payload={"query": "summarise", "file": handle.id})
```

## Notes

When `resume` is true the client records part offsets in a local state file
next to the upload. Deleting that file forces a fresh upload from byte zero.

`checksum` costs one extra pass over the file. Disabling it is only sensible
for files you can cheaply re-upload, because a silent corruption will otherwise
surface much later as an unexplained parsing error.
