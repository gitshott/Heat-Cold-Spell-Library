# Contributing

Thank you for contributing to the Heat / Cold Spell Prediction and Analysis library.

## Before contributing

Please:

- Read the `README.md`.
- Check existing issues and pull requests.
- Do not upload private datasets, credentials, API keys, or institutional information.
- Confirm that contributed code and documentation may be redistributed under the project license.
- Clearly identify whether your change affects algorithms, thresholds, APIs, outputs, or documentation.

## Development setup

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt