# Contributing to ComfyUI-Acheng-ShotVault

Thank you for your interest in improving **ComfyUI-Acheng-ShotVault**!

## Code of Conduct & Open-Source Ethos

- Keep discussions technical, constructive, and evidence-based.
- Respect all third-party licenses and never submit copyrighted code, proprietary weights, or private user media/prompts in Pull Requests.

## Development Workflow

1. Fork [https://github.com/AharaOoO/ComfyUI-Acheng-ShotVault](https://github.com/AharaOoO/ComfyUI-Acheng-ShotVault) and create a feature branch from `main`.
2. Ensure all Python files compile cleanly and existing unit tests pass:
   ```bash
   python -m unittest discover -s tests -p "test_phase*.py"
   ```
3. Verify frontend JS modules (`web/*.js`, `web/panel/*.js`) have zero syntax errors and preserve bilingual (`zh` / `en`) labels in `web/i18n.js`.
4. Never break existing `INPUT_TYPES` widget order or defaults in `ShotVaultPlan` / `ShotVaultSave` / `ShotVaultAssemble` without an explicit migration path.

## License Agreement (Inbound = Outbound)

By submitting a Pull Request or patch to this repository, you certify that you have the legal right to contribute the work and agree that your contribution is licensed to the project and all downstream users under **GNU General Public License v3.0 (`GPL-3.0-only`)**.
