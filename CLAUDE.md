# CLAUDE.md - Repository Guide

## Commands
- Install: `uv pip install -r requirements.txt && uv pip install -e .` (or `requirements_gpu.txt` for GPU)
- Run script: `python script_name.py`
- Test single file: `python -m pytest path/to/test_file.py`
- Batch OCR: `python project_scripts/send_processed_issues_to_pixtral_as_batch.py`
- Convert PDF: `python project_scripts/convert_all_ncse.py`

## Code Style Guidelines
- Type hints: Use proper type annotations (functions, params, returns)
- Docstrings: Include detailed documentation with Args/Returns/Raises sections
- Imports: Group standard library, third-party, and local imports
- Error handling: Use try/except with specific exceptions, log errors
- File structure: Organize code in function_modules/, project_scripts/, lightning_scripts/
- Naming: snake_case for functions/variables, PascalCase for classes
- Line length: Keep under 100 characters
- Retry logic: Use tenacity for API calls and network operations
- Paths: Use pathlib.Path for file system operations