#!/bin/bash

# wt - Git Worktree Manager Script
# Usage: wt <feature-name>
# Creates a new git worktree with the specified feature name

set -e

# Check if feature name is provided
if [ -z "$1" ]; then
    echo "Usage: wt <feature-name>"
    echo "Creates a new git worktree with the specified feature name"
    exit 1
fi

FEATURE_NAME="$1"

# Get current project directory name
CURRENT_DIR=$(basename "$(pwd)")
PARENT_DIR=$(dirname "$(pwd)")
WORKTREES_DIR="${PARENT_DIR}/${CURRENT_DIR}-worktrees"

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
git worktree add -b "$FEATURE_NAME" "$WORKTREE_PATH"

echo "Successfully created worktree and branch: $FEATURE_NAME"
echo "Worktree location: $WORKTREE_PATH"

# Open the new worktree in Claude
echo "Opening worktree in Claude..."
cd "$WORKTREE_PATH"
claude --model claude-sonnet-4-20250514

echo "Done! New Claude window should open with the worktree."