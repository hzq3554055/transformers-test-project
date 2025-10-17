"""
Proxy utilities for Hugging Face Hub access.
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


def setup_proxy(
    http_proxy: Optional[str] = None,
    https_proxy: Optional[str] = None,
    config_file: Optional[str] = None
) -> Dict[str, str]:
    """
    Setup proxy configuration for Hugging Face Hub access.
    
    Args:
        http_proxy: HTTP proxy URL
        https_proxy: HTTPS proxy URL
        config_file: Path to proxy configuration file
        
    Returns:
        Dictionary of proxy settings
    """
    proxy_settings = {}
    
    # Load from config file if provided
    if config_file and Path(config_file).exists():
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            if config.get('proxy', {}).get('enabled', False):
                http_proxy = config['proxy'].get('http_proxy', http_proxy)
                https_proxy = config['proxy'].get('https_proxy', https_proxy)
                
                # Set Hugging Face Hub settings
                if 'huggingface' in config:
                    hf_config = config['huggingface']
                    os.environ['HF_HUB_OFFLINE'] = str(not hf_config.get('hub_offline', False))
                    os.environ['HF_DATASETS_OFFLINE'] = str(not hf_config.get('datasets_offline', False))
                    
                    if 'cache_dir' in hf_config:
                        os.environ['HF_HOME'] = hf_config['cache_dir']
                
                logger.info(f"Proxy configuration loaded from {config_file}")
                
        except Exception as e:
            logger.warning(f"Failed to load proxy config from {config_file}: {e}")
    
    # Use provided values or defaults
    if not http_proxy:
        http_proxy = os.environ.get('HTTP_PROXY', os.environ.get('http_proxy'))
    if not https_proxy:
        https_proxy = os.environ.get('HTTPS_PROXY', os.environ.get('https_proxy'))
    
    # Set proxy environment variables
    if http_proxy:
        os.environ['HTTP_PROXY'] = http_proxy
        os.environ['http_proxy'] = http_proxy
        proxy_settings['HTTP_PROXY'] = http_proxy
        logger.info(f"HTTP proxy set to: {http_proxy}")
    
    if https_proxy:
        os.environ['HTTPS_PROXY'] = https_proxy
        os.environ['https_proxy'] = https_proxy
        proxy_settings['HTTPS_PROXY'] = https_proxy
        logger.info(f"HTTPS proxy set to: {https_proxy}")
    
    return proxy_settings


def test_proxy_connection(proxy_url: Optional[str] = None) -> bool:
    """
    Test proxy connection to Hugging Face.
    
    Args:
        proxy_url: Proxy URL to test
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        import requests
        
        if proxy_url:
            proxies = {
                'http': proxy_url,
                'https': proxy_url
            }
        else:
            proxies = {
                'http': os.environ.get('HTTP_PROXY'),
                'https': os.environ.get('HTTPS_PROXY')
            }
        
        # Test connection to Hugging Face
        response = requests.get(
            'https://huggingface.co',
            proxies=proxies,
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info("✅ Proxy connection to Hugging Face successful")
            return True
        else:
            logger.warning(f"Proxy connection returned status {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"Proxy connection test failed: {e}")
        return False


def get_proxy_info() -> Dict[str, Any]:
    """
    Get current proxy configuration information.
    
    Returns:
        Dictionary with proxy information
    """
    return {
        'HTTP_PROXY': os.environ.get('HTTP_PROXY'),
        'HTTPS_PROXY': os.environ.get('HTTPS_PROXY'),
        'http_proxy': os.environ.get('http_proxy'),
        'https_proxy': os.environ.get('https_proxy'),
        'HF_HUB_OFFLINE': os.environ.get('HF_HUB_OFFLINE', '0'),
        'HF_DATASETS_OFFLINE': os.environ.get('HF_DATASETS_OFFLINE', '0'),
        'HF_HOME': os.environ.get('HF_HOME', '~/.cache/huggingface')
    }


def setup_huggingface_proxy(config_file: str = "configs/proxy_config.yaml"):
    """
    Setup proxy for Hugging Face Hub with automatic configuration.
    
    Args:
        config_file: Path to proxy configuration file
    """
    # Setup proxy from config
    proxy_settings = setup_proxy(config_file=config_file)
    
    # Test connection
    if proxy_settings:
        connection_ok = test_proxy_connection()
        if connection_ok:
            logger.info("🎉 Hugging Face proxy configuration successful!")
        else:
            logger.warning("⚠️  Proxy connection test failed, but settings are configured")
    else:
        logger.info("ℹ️  No proxy configuration found, using direct connection")
    
    return proxy_settings


if __name__ == "__main__":
    # Test proxy setup
    logging.basicConfig(level=logging.INFO)
    
    print("🌐 Testing Proxy Configuration")
    print("=" * 40)
    
    # Setup proxy
    proxy_settings = setup_huggingface_proxy()
    
    # Show proxy info
    proxy_info = get_proxy_info()
    print("\n📋 Current Proxy Settings:")
    for key, value in proxy_info.items():
        print(f"  {key}: {value}")
    
    # Test connection
    print("\n🔍 Testing Connection...")
    if test_proxy_connection():
        print("✅ Proxy connection successful!")
    else:
        print("❌ Proxy connection failed!")
