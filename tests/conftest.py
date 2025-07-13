"""Test fixtures for Claude launcher tests."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from claude_launcher import ClaudeModel, LauncherConfig


@pytest.fixture
def mock_api_response():
    """Mock API response data."""
    return {
        "data": [
            {
                "id": "claude-sonnet-4-20250514",
                "display_name": "Claude Sonnet 4",
                "created": "2025-01-01T00:00:00Z",
                "type": "model"
            },
            {
                "id": "claude-opus-4-20250514", 
                "display_name": "Claude Opus 4",
                "created": "2025-01-01T00:00:00Z",
                "type": "model"
            },
            {
                "id": "claude-3-5-haiku-20241022",
                "display_name": "Claude 3.5 Haiku",
                "created": "2024-10-22T00:00:00Z",
                "type": "model"
            }
        ]
    }


@pytest.fixture
def sample_models():
    """Sample ClaudeModel instances for testing."""
    return [
        ClaudeModel(
            id="claude-sonnet-4-20250514",
            display_name="Claude Sonnet 4",
            created_at=datetime(2025, 1, 1),
            type="model"
        ),
        ClaudeModel(
            id="claude-opus-4-20250514",
            display_name="Claude Opus 4", 
            created_at=datetime(2025, 1, 1),
            type="model"
        )
    ]


@pytest.fixture
def sample_config():
    """Sample LauncherConfig for testing."""
    return LauncherConfig(
        last_selected_model="claude-sonnet-4-20250514",
        api_key_env_var="ANTHROPIC_API_KEY",
        cache_duration_hours=24,
        default_args=["--verbose"]
    )


@pytest.fixture
def temp_config_file():
    """Temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config_data = {
            "last_selected_model": "claude-sonnet-4-20250514",
            "api_key_env_var": "ANTHROPIC_API_KEY",
            "cache_duration_hours": 24,
            "default_args": ["--verbose"]
        }
        json.dump(config_data, f)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_cache_file():
    """Temporary cache file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        # Use current time to ensure cache is valid
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "models": [
                {
                    "id": "claude-sonnet-4-20250514",
                    "display_name": "Claude Sonnet 4",
                    "created_at": "2025-01-01T00:00:00",
                    "type": "model"
                }
            ]
        }
        json.dump(cache_data, f)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for API testing."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_client.get.return_value = mock_response
    return mock_client, mock_response


@pytest.fixture
def mock_subprocess():
    """Mock subprocess for CLI testing."""
    with patch('claude_launcher.subprocess') as mock_sub:
        mock_sub.run.return_value = Mock(returncode=0)
        mock_sub.CalledProcessError = Exception
        yield mock_sub


@pytest.fixture
def mock_environment():
    """Mock environment variables."""
    with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'sk-test-key-1234567890'}, clear=False):
        yield