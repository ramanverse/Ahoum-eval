# Contributing to Ahoum Conversation Evaluation System

We welcome contributions to improve the Ahoum evaluation pipeline! Please review these guidelines before submitting a pull request.

## Code Style & Standards

1. **Python Formatting**: Follow PEP-8. We use `black` for formatting and `flake8` / `ruff` for linting.
2. **Type Hints**: Annotate all public methods and functions.
3. **Docstrings**: Provide Google-style docstrings for all modules, classes, and helper functions.
4. **Reproducibility**: If you introduce random processes, ensure you use the project-wide random seed configured in `config.yaml` and set using `utils.set_seed()`.

## Testing

All code changes must pass the test suite:
```bash
pytest tests/
```
Please add unit tests under `tests/` for any new feature extraction methods or scorer updates.

## Development Workflow

1. Fork the repository and create your branch from `main`.
2. Install dev dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Implement your changes and verify with `pytest`.
4. Submit a Pull Request detailing the changes, the motivations, and verification metrics.
