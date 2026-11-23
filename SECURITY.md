# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability (such as path traversal, unsafe file handling, command injection, or arbitrary file read/write in the local ComfyUI server routes), please **do not** open a public issue with exploit details.

Instead, report it privately via GitHub Security Advisories at:
- https://github.com/AharaOoO/ComfyUI-Acheng-ShotVault/security/advisories/new

Please include:
1. Affected version (`v5.8.0` or commit SHA) and ComfyUI environment details.
2. Minimal reproduction steps or proof-of-concept.
3. Expected vs. observed impact.

We will acknowledge receipt and coordinate a fix before public disclosure.

## Local-First Security Design

- All ShotVault disk operations enforce root-directory confinement (`ensure_inside_root`) to prevent path traversal outside the ComfyUI output/vault workspace.
- Trash/cleanup actions route through `send2trash` (OS recycle bin) rather than irreversible raw recursive deletes.
- No telemetry, analytics, or external network uploads are performed by any node in this repository.
