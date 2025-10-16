#!/bin/bash

# Toggle Pre-commit Hook Script
# This script allows you to easily enable/disable the pre-commit hook

HOOK_PATH=".git/hooks/pre-commit"
DISABLED_PATH=".git/hooks/pre-commit.disabled"

if [ -f "$HOOK_PATH" ]; then
    echo "🔧 Disabling pre-commit hook..."
    mv "$HOOK_PATH" "$DISABLED_PATH"
    echo "✅ Pre-commit hook disabled"
    echo "   You can now commit without linting checks"
elif [ -f "$DISABLED_PATH" ]; then
    echo "🔧 Enabling pre-commit hook..."
    mv "$DISABLED_PATH" "$HOOK_PATH"
    echo "✅ Pre-commit hook enabled"
    echo "   Linting checks will run on commit"
else
    echo "❌ No pre-commit hook found"
    echo "   Available options:"
    echo "   - Create new hook: ./scripts/setup_pre_commit.sh"
    echo "   - Check status: ls -la .git/hooks/pre-commit*"
fi

echo ""
echo "📋 Current status:"
if [ -f "$HOOK_PATH" ]; then
    echo "   Pre-commit hook: ✅ ENABLED"
elif [ -f "$DISABLED_PATH" ]; then
    echo "   Pre-commit hook: ❌ DISABLED"
else
    echo "   Pre-commit hook: ❓ NOT FOUND"
fi
