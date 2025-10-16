#!/bin/bash

# Make Proxy Configuration Permanent
# This script adds proxy settings to your shell configuration

set -e  # Exit on any error

echo "🔧 Making Proxy Configuration Permanent"
echo "====================================="

PROXY_URL="http://127.0.0.1:7897"
SHELL_CONFIG=""

# Detect shell and set config file
if [ -n "$ZSH_VERSION" ]; then
    SHELL_CONFIG="$HOME/.zshrc"
    echo "🐚 Detected Zsh shell"
elif [ -n "$BASH_VERSION" ]; then
    SHELL_CONFIG="$HOME/.bashrc"
    echo "🐚 Detected Bash shell"
else
    echo "⚠️  Unknown shell, using ~/.bashrc as default"
    SHELL_CONFIG="$HOME/.bashrc"
fi

echo "📝 Adding proxy configuration to $SHELL_CONFIG"

# Check if proxy settings already exist
if grep -q "HTTP_PROXY.*127.0.0.1:7897" "$SHELL_CONFIG" 2>/dev/null; then
    echo "✅ Proxy settings already exist in $SHELL_CONFIG"
else
    # Add proxy configuration
    cat >> "$SHELL_CONFIG" << EOF

# Hugging Face Proxy Configuration (added by transformers-test framework)
export HTTP_PROXY="$PROXY_URL"
export HTTPS_PROXY="$PROXY_URL"
export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"

EOF
    echo "✅ Proxy settings added to $SHELL_CONFIG"
fi

echo ""
echo "🎉 Proxy configuration made permanent!"
echo ""
echo "📋 Configuration added:"
echo "  export HTTP_PROXY=\"$PROXY_URL\""
echo "  export HTTPS_PROXY=\"$PROXY_URL\""
echo "  export http_proxy=\"$PROXY_URL\""
echo "  export https_proxy=\"$PROXY_URL\""
echo ""
echo "🔄 To apply changes immediately, run:"
echo "  source $SHELL_CONFIG"
echo ""
echo "🚀 Your proxy settings will now be available in new terminal sessions!"
