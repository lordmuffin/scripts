#!/bin/bash
# Claude Launcher Wrapper Script for WSL/Linux environments
# This script ensures the launcher works correctly in all environments

# Force threading mode for maximum compatibility
export CLAUDE_LAUNCHER_FORCE_THREADING=1

# Run the Python launcher with all passed arguments
python "$(dirname "$0")/claude_launcher.py" "$@"