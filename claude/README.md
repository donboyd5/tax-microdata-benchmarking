# Claude Diagnostic Files

This directory contains diagnostic scripts created during development to troubleshoot specific issues. These files are **not** part of the main TMD codebase and should not be run by pytest.

## Files

- **`cps_comprehensive_test.py`** - Comprehensive CPS URL testing with SSL handling  
- **`cps_step_by_step_diagnostic.py`** - Step-by-step CPS URL diagnostic script
- **`cps_url_diagnostic.py`** - Initial CPS URL accessibility tests

## Purpose

These scripts were created to diagnose and fix SSL certificate verification issues with Census Bureau CPS data downloads. They can be run individually for troubleshooting but are not part of the automated test suite.

## Usage

Run individually:
```bash
cd claude/
python cps_comprehensive_test.py
python cps_step_by_step_diagnostic.py
python cps_url_diagnostic.py
```

**Note**: Files were renamed (no longer start with `test_`) to prevent pytest from attempting to run them as part of the main test suite.