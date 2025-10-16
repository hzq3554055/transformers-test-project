#!/bin/bash

# Setup Proxy Script
# This script configures proxy settings for Hugging Face Hub access

set -e  # Exit on any error

echo "🌐 Setting up Proxy Configuration for Hugging Face"
echo "================================================="

# Proxy configuration
PROXY_URL="http://127.0.0.1:7897"

echo "📋 Proxy Configuration:"
echo "  HTTP_PROXY: $PROXY_URL"
echo "  HTTPS_PROXY: $PROXY_URL"

# Set environment variables
export HTTP_PROXY="$PROXY_URL"
export HTTPS_PROXY="$PROXY_URL"
export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"

# Also set for Hugging Face Hub
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0

echo "✅ Environment variables set"

# Test proxy connection
echo "🔍 Testing proxy connection..."
if curl -I --proxy "$PROXY_URL" https://huggingface.co > /dev/null 2>&1; then
    echo "✅ Proxy connection to Hugging Face successful"
else
    echo "❌ Proxy connection failed"
    echo "Please check your proxy settings and try again"
    exit 1
fi

# Test Hugging Face Hub connection
echo "🧪 Testing Hugging Face Hub connection..."
python3 -c "
import os
os.environ['HTTP_PROXY'] = '$PROXY_URL'
os.environ['HTTPS_PROXY'] = '$PROXY_URL'

try:
    from huggingface_hub import HfApi
    api = HfApi()
    # Test connection by getting model info
    model_info = api.model_info('bert-base-uncased')
    print('✅ Hugging Face Hub connection successful')
    print(f'   Model: {model_info.modelId}')
    print(f'   Downloads: {model_info.downloads:,}')
except Exception as e:
    print(f'❌ Hugging Face Hub connection failed: {e}')
    exit(1)
"

echo ""
echo "🎉 Proxy configuration completed successfully!"
echo ""
echo "📋 Configuration Summary:"
echo "  ✅ HTTP_PROXY: $HTTP_PROXY"
echo "  ✅ HTTPS_PROXY: $HTTPS_PROXY"
echo "  ✅ Hugging Face Hub: Connected"
echo ""
echo "🚀 You can now use the transformers testing framework with real models!"
echo ""
echo "To make this permanent, add these lines to your ~/.bashrc or ~/.zshrc:"
echo "export HTTP_PROXY=\"$PROXY_URL\""
echo "export HTTPS_PROXY=\"$PROXY_URL\""
echo "export http_proxy=\"$PROXY_URL\""
echo "export https_proxy=\"$PROXY_URL\""
