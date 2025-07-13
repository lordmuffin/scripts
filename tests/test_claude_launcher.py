"""Tests for Claude CLI Launcher."""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import pytest

from claude_launcher import (
    ClaudeModel, LauncherConfig, AnthropicAPIClient, 
    validate_api_key, get_api_key, load_cached_models, 
    save_cached_models, launch_claude_cli, select_model_interactive
)


def test_claude_model_from_api_response():
    """Test ClaudeModel creation from API response."""
    api_data = {
        "id": "claude-sonnet-4-20250514",
        "display_name": "Claude Sonnet 4",
        "created": "2025-01-01T00:00:00Z",
        "type": "model"
    }
    
    model = ClaudeModel.from_api_response(api_data)
    
    assert model.id == "claude-sonnet-4-20250514"
    assert model.display_name == "Claude Sonnet 4"
    assert model.type == "model"
    assert isinstance(model.created_at, datetime)


def test_launcher_config_defaults():
    """Test LauncherConfig default values."""
    config = LauncherConfig()
    
    assert config.last_selected_model is None
    assert config.api_key_env_var == "ANTHROPIC_API_KEY"
    assert config.cache_duration_hours == 24
    assert config.default_args == []


def test_launcher_config_post_init():
    """Test LauncherConfig __post_init__ handling."""
    config = LauncherConfig(default_args=None)
    assert config.default_args == []


def test_configuration_loading(temp_config_file):
    """Test configuration file loading and validation."""
    config = LauncherConfig.load_from_file(temp_config_file)
    
    assert isinstance(config, LauncherConfig)
    assert config.api_key_env_var == "ANTHROPIC_API_KEY"
    assert config.last_selected_model == "claude-sonnet-4-20250514"
    assert config.cache_duration_hours == 24
    assert config.default_args == ["--verbose"]


def test_configuration_loading_nonexistent_file():
    """Test configuration loading with nonexistent file."""
    nonexistent_path = Path("/tmp/nonexistent_config.json")
    config = LauncherConfig.load_from_file(nonexistent_path)
    
    # Should return default config
    assert isinstance(config, LauncherConfig)
    assert config.last_selected_model is None


def test_configuration_saving():
    """Test configuration saving to file."""
    config = LauncherConfig(
        last_selected_model="claude-sonnet-4-20250514",
        cache_duration_hours=48,
        default_args=["--verbose", "--help"]
    )
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = Path(f.name)
    
    try:
        config.save_to_file(temp_path)
        
        # Verify file was created and contains correct data
        assert temp_path.exists()
        
        with open(temp_path, 'r') as f:
            data = json.load(f)
        
        assert data["last_selected_model"] == "claude-sonnet-4-20250514"
        assert data["cache_duration_hours"] == 48
        assert data["default_args"] == ["--verbose", "--help"]
        
    finally:
        if temp_path.exists():
            temp_path.unlink()


def test_api_key_validation():
    """Test API key presence and format validation."""
    # Valid API key
    assert validate_api_key("sk-test-key-1234567890") is True
    
    # Invalid format
    with pytest.raises(ValueError, match="Invalid API key format"):
        validate_api_key("invalid-key")
    
    # Empty key
    with pytest.raises(ValueError, match="API key not found"):
        validate_api_key("")
    
    # Too short
    with pytest.raises(ValueError, match="API key too short"):
        validate_api_key("sk-123")


def test_get_api_key_from_environment(mock_environment):
    """Test API key retrieval from environment."""
    config = LauncherConfig()
    api_key = get_api_key(config)
    assert api_key == "sk-test-key-1234567890"


def test_get_api_key_missing():
    """Test API key retrieval when missing."""
    config = LauncherConfig()
    
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(SystemExit):
            get_api_key(config)


@pytest.mark.asyncio
async def test_model_discovery():
    """Test API model fetching with mocked response."""
    mock_response_data = {
        "data": [
            {
                "id": "claude-sonnet-4-20250514",
                "display_name": "Claude Sonnet 4",
                "created": "2025-01-01T00:00:00Z",
                "type": "model"
            }
        ]
    }
    
    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        
        client = AnthropicAPIClient("sk-test-key")
        models = await client.fetch_models()
        
        assert len(models) == 1
        assert models[0].id == "claude-sonnet-4-20250514"
        assert models[0].display_name == "Claude Sonnet 4"


@pytest.mark.asyncio 
async def test_api_client_rate_limiting():
    """Test API client rate limiting handling."""
    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        
        # First call returns 429, second succeeds
        rate_limit_response = Mock()
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {"retry-after": "1"}
        
        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = {"data": []}
        success_response.raise_for_status.return_value = None
        
        mock_client.get.side_effect = [rate_limit_response, success_response]
        
        with patch('asyncio.sleep') as mock_sleep:
            client = AnthropicAPIClient("sk-test-key")
            models = await client.fetch_models()
            
            assert isinstance(models, list)
            mock_sleep.assert_called_once()


@pytest.mark.asyncio
async def test_api_client_invalid_key():
    """Test API client with invalid API key."""
    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        
        mock_response = Mock()
        mock_response.status_code = 401
        mock_client.get.return_value = mock_response
        
        # Configure the side effect to raise HTTPStatusError
        import httpx
        error = httpx.HTTPStatusError("Unauthorized", request=Mock(), response=mock_response)
        mock_client.get.side_effect = error
        
        client = AnthropicAPIClient("invalid-key")
        
        with pytest.raises(ValueError, match="Invalid API key"):
            await client.fetch_models()


def test_cache_loading(temp_cache_file):
    """Test loading models from cache."""
    models = load_cached_models(temp_cache_file, 24)
    
    assert models is not None
    assert len(models) == 1
    assert models[0].id == "claude-sonnet-4-20250514"


def test_cache_loading_expired():
    """Test cache loading with expired cache."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        # Create cache with old timestamp
        old_time = datetime.now() - timedelta(hours=25)
        cache_data = {
            "timestamp": old_time.isoformat(),
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
    
    try:
        models = load_cached_models(temp_path, 24)
        assert models is None  # Should be None due to expiry
        
    finally:
        if temp_path.exists():
            temp_path.unlink()


def test_cache_saving(sample_models):
    """Test saving models to cache."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = Path(f.name)
    
    try:
        save_cached_models(temp_path, sample_models)
        
        # Verify cache file was created
        assert temp_path.exists()
        
        with open(temp_path, 'r') as f:
            cache_data = json.load(f)
        
        assert "timestamp" in cache_data
        assert "models" in cache_data
        assert len(cache_data["models"]) == 2
        
    finally:
        if temp_path.exists():
            temp_path.unlink()


def test_claude_cli_integration(mock_subprocess, sample_models):
    """Test subprocess execution with proper arguments."""
    model = sample_models[0]
    
    launch_claude_cli(model, ["--verbose"], dry_run=False)
    
    mock_subprocess.run.assert_called_once()
    args = mock_subprocess.run.call_args[0][0]
    
    assert "claude" in args
    assert "--model" in args
    assert "claude-sonnet-4-20250514" in args
    assert "--verbose" in args


def test_claude_cli_dry_run(sample_models):
    """Test dry run mode."""
    model = sample_models[0]
    
    with patch('claude_launcher.console') as mock_console:
        launch_claude_cli(model, ["--verbose"], dry_run=True)
        
        # Should print the command but not execute
        mock_console.print.assert_called()
        call_args = mock_console.print.call_args[0][0]
        assert "Would execute:" in call_args


def test_claude_cli_file_not_found(sample_models):
    """Test Claude CLI not found error."""
    model = sample_models[0]
    
    with patch('claude_launcher.subprocess') as mock_subprocess:
        mock_subprocess.run.side_effect = FileNotFoundError()
        mock_subprocess.CalledProcessError = Exception
        
        with pytest.raises(SystemExit):
            launch_claude_cli(model, [])


def test_claude_cli_process_error(sample_models):
    """Test Claude CLI process error."""
    model = sample_models[0]
    
    with patch('claude_launcher.subprocess') as mock_subprocess:
        import subprocess
        error = subprocess.CalledProcessError(1, ['claude'])
        mock_subprocess.run.side_effect = error
        mock_subprocess.CalledProcessError = subprocess.CalledProcessError
        
        with pytest.raises(SystemExit):
            launch_claude_cli(model, [])


def test_interactive_selection_fallback(sample_models, sample_config):
    """Test interactive model selection fallback when InquirerPy unavailable."""
    with patch('claude_launcher.has_inquirer', False):
        with patch('builtins.input', return_value='1'):
            with patch('claude_launcher.console') as mock_console:
                selected = select_model_interactive(sample_models, sample_config)
                
                assert selected.id == "claude-sonnet-4-20250514"
                mock_console.print.assert_called()


def test_interactive_selection_invalid_input(sample_models, sample_config):
    """Test interactive selection with invalid input."""
    with patch('claude_launcher.has_inquirer', False):
        # Simulate invalid input then valid input
        with patch('builtins.input', side_effect=['invalid', '99', '1']):
            with patch('claude_launcher.console'):
                # Since invalid input leads to sys.exit(), expect SystemExit
                with pytest.raises(SystemExit):
                    select_model_interactive(sample_models, sample_config)


def test_interactive_selection_keyboard_interrupt(sample_models, sample_config):
    """Test handling of keyboard interrupt during selection."""
    with patch('claude_launcher.has_inquirer', False):
        with patch('builtins.input', side_effect=KeyboardInterrupt()):
            with pytest.raises(SystemExit):
                select_model_interactive(sample_models, sample_config)


class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_full_workflow_mock(self, mock_environment, mock_api_response):
        """Test complete workflow with mocked dependencies."""
        from claude_launcher import fetch_available_models
        
        config = LauncherConfig()
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_api_response
            mock_response.raise_for_status.return_value = None
            mock_client.get.return_value = mock_response
            
            models = await fetch_available_models(config, force_refresh=True)
            
            assert len(models) == 3
            assert any(model.id == "claude-sonnet-4-20250514" for model in models)
            assert any(model.id == "claude-opus-4-20250514" for model in models)
            assert any(model.id == "claude-3-5-haiku-20241022" for model in models)
    
    def test_argument_parsing(self):
        """Test command line argument parsing."""
        from claude_launcher import parse_args
        
        # Test with various arguments
        test_args = ["--list-models", "--verbose", "--dry-run", "--cache-bust"]
        
        with patch('sys.argv', ['claude_launcher.py'] + test_args):
            args = parse_args()
            
            assert args.list_models is True
            assert args.verbose is True
            assert args.dry_run is True
            assert args.cache_bust is True