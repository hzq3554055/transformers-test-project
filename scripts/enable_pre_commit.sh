#!/bin/bash

# Enable Pre-commit Hook Script
# This script re-enables the pre-commit hook

HOOK_PATH=".git/hooks/pre-commit"
DISABLED_PATH=".git/hooks/pre-commit.disabled"

if [ -f "$DISABLED_PATH" ]; then
    echo "🔧 Re-enabling pre-commit hook..."
    mv "$DISABLED_PATH" "$HOOK_PATH"
    echo "✅ Pre-commit hook enabled"
    echo "   Linting checks will run on commit"
else
    echo "❌ No disabled pre-commit hook found"
    echo "   Current status:"
    if [ -f "$HOOK_PATH" ]; then
        echo "   Pre-commit hook: ✅ ENABLED"
    else
        echo "   Pre-commit hook: ❓ NOT FOUND"
    fi
fi
