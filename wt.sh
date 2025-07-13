#!/bin/bash

# wt - Git Worktree Manager Script
# Usage: wt <feature-name> OR wt <repo-name> <feature-name>
# Creates a new git worktree with the specified feature name
#
# INSTALLATION:
# To make this script available globally in your bash shell:
#
# Option 1 - Add to PATH:
#   1. Make script executable: chmod +x /path/to/wt.sh
#   2. Add script directory to PATH in ~/.bashrc:
#      export PATH="$PATH:/path/to/scripts"
#   3. Reload shell: source ~/.bashrc
#
# Option 2 - Create symlink in PATH directory:
#   1. Make script executable: chmod +x /path/to/wt.sh
#   2. Create symlink: sudo ln -s /path/to/wt.sh /usr/local/bin/wt
#
# Option 3 - Copy to system directory:
#   1. Make script executable: chmod +x /path/to/wt.sh
#   2. Copy to system bin: sudo cp /path/to/wt.sh /usr/local/bin/wt

set -e

# Set git discovery across filesystem for WSL compatibility
export GIT_DISCOVERY_ACROSS_FILESYSTEM=1

# Handle destroy-all command
if [ "$1" = "--destroy-all" ]; then
    # Find git repository root
    GIT_ROOT=$(git -c safe.directory='*' rev-parse --show-toplevel 2>/dev/null)
    if [ -z "$GIT_ROOT" ]; then
        echo "Error: Not in a git repository"
        exit 1
    fi
    
    # Get project directory name from git root
    CURRENT_DIR=$(basename "$GIT_ROOT")
    PARENT_DIR=$(dirname "$GIT_ROOT")
    WORKTREES_DIR="${PARENT_DIR}/${CURRENT_DIR}-worktrees"
    
    # Change to git root for worktree operations
    cd "$GIT_ROOT"
    
    echo "Destroying all worktrees for project: $CURRENT_DIR"
    echo "Worktrees directory: $WORKTREES_DIR"
    
    # Check if worktrees directory exists
    if [ ! -d "$WORKTREES_DIR" ]; then
        echo "No worktrees directory found at: $WORKTREES_DIR"
        exit 0
    fi
    
    # List and remove all worktrees
    echo "Found worktrees to remove:"
    for worktree_path in "$WORKTREES_DIR"/*; do
        if [ -d "$worktree_path" ]; then
            worktree_name=$(basename "$worktree_path")
            echo "  - $worktree_name"
            
            # Remove the worktree from git
            git -c safe.directory='*' worktree remove "$worktree_path" --force 2>/dev/null || true
            
            # Remove the directory if it still exists
            if [ -d "$worktree_path" ]; then
                rm -rf "$worktree_path"
            fi
        fi
    done
    
    # Remove the worktrees directory if empty
    if [ -d "$WORKTREES_DIR" ]; then
        rmdir "$WORKTREES_DIR" 2>/dev/null || true
    fi
    
    echo "All worktrees have been destroyed."
    exit 0
fi

# Check arguments and determine usage pattern
if [ -z "$1" ]; then
    echo "Usage: wt <feature-name>"
    echo "   OR: wt <repo-name> <feature-name>"
    echo "   OR: wt --destroy-all"
    echo "   OR: wt <repo-name> --destroy-all"
    echo "Creates a new git worktree with the specified feature name"
    echo "If repo-name is provided, changes to that directory first"
    echo "Use --destroy-all to remove all worktrees for this project"
    exit 1
fi

# Determine if we have repo-name and feature-name or just feature-name
if [ -n "$2" ]; then
    # Two arguments: wt <repo-name> <feature-name> OR wt <repo-name> --destroy-all
    REPO_NAME="$1"
    SECOND_ARG="$2"
    
    # Change to the specified repository directory
    # First try current directory, then parent directory
    if [ -d "$REPO_NAME" ]; then
        REPO_PATH="$REPO_NAME"
    elif [ -d "../$REPO_NAME" ]; then
        REPO_PATH="../$REPO_NAME"
    else
        echo "Error: Repository directory '$REPO_NAME' not found in current or parent directory"
        exit 1
    fi
    
    echo "Changing to repository: $REPO_NAME"
    cd "$REPO_PATH"
    
    # Handle destroy-all for specific repository
    if [ "$SECOND_ARG" = "--destroy-all" ]; then
        # Find git repository root
        GIT_ROOT=$(git -c safe.directory='*' rev-parse --show-toplevel 2>/dev/null)
        if [ -z "$GIT_ROOT" ]; then
            echo "Error: Not in a git repository"
            exit 1
        fi
        
        # Get project directory name from git root
        CURRENT_DIR=$(basename "$GIT_ROOT")
        PARENT_DIR=$(dirname "$GIT_ROOT")
        WORKTREES_DIR="${PARENT_DIR}/${CURRENT_DIR}-worktrees"
        
        # Change to git root for worktree operations
        cd "$GIT_ROOT"
        
        echo "Destroying all worktrees for project: $CURRENT_DIR"
        echo "Worktrees directory: $WORKTREES_DIR"
        
        # Check if worktrees directory exists
        if [ ! -d "$WORKTREES_DIR" ]; then
            echo "No worktrees directory found at: $WORKTREES_DIR"
            exit 0
        fi
        
        # List and remove all worktrees
        echo "Found worktrees to remove:"
        for worktree_path in "$WORKTREES_DIR"/*; do
            if [ -d "$worktree_path" ]; then
                worktree_name=$(basename "$worktree_path")
                echo "  - $worktree_name"
                
                # Remove the worktree from git
                git -c safe.directory='*' worktree remove "$worktree_path" --force 2>/dev/null || true
                
                # Remove the directory if it still exists
                if [ -d "$worktree_path" ]; then
                    rm -rf "$worktree_path"
                fi
            fi
        done
        
        # Remove the worktrees directory if empty
        if [ -d "$WORKTREES_DIR" ]; then
            rmdir "$WORKTREES_DIR" 2>/dev/null || true
        fi
        
        echo "All worktrees have been destroyed."
        exit 0
    fi
    
    FEATURE_NAME="$SECOND_ARG"
else
    # One argument: wt <feature-name>
    FEATURE_NAME="$1"
fi

# Find git repository root
GIT_ROOT=$(git -c safe.directory='*' rev-parse --show-toplevel 2>/dev/null)
if [ -z "$GIT_ROOT" ]; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Get project directory name from git root
CURRENT_DIR=$(basename "$GIT_ROOT")
PARENT_DIR=$(dirname "$GIT_ROOT")
WORKTREES_DIR="${PARENT_DIR}/${CURRENT_DIR}-worktrees"

# Change to git root for worktree operations
cd "$GIT_ROOT"

echo "Creating worktree setup for feature: $FEATURE_NAME"
echo "Current project: $CURRENT_DIR"
echo "Worktrees directory: $WORKTREES_DIR"

# Create worktrees directory if it doesn't exist
if [ ! -d "$WORKTREES_DIR" ]; then
    echo "Creating worktrees directory: $WORKTREES_DIR"
    mkdir -p "$WORKTREES_DIR"
fi

# Create git worktree
WORKTREE_PATH="${WORKTREES_DIR}/${FEATURE_NAME}"

echo "Creating git worktree at: $WORKTREE_PATH"

# Check if worktree already exists
if [ -d "$WORKTREE_PATH" ]; then
    echo "Error: Worktree '$FEATURE_NAME' already exists at $WORKTREE_PATH"
    exit 1
fi

# Create the worktree with a new branch
git -c safe.directory='*' worktree add -b "$FEATURE_NAME" "$WORKTREE_PATH"

echo "Successfully created worktree and branch: $FEATURE_NAME"
echo "Worktree location: $WORKTREE_PATH"

# Open the new worktree in Claude
echo "Opening worktree in Claude..."
cd "$WORKTREE_PATH"
claude --model claude-sonnet-4-20250514

echo "Done! New Claude window should open with the worktree."