#!/usr/bin/env python3
"""
Claude CLI Launcher with Model Selection

A Python-based interactive launcher script for Claude CLI that provides model selection
capabilities with persistent configuration management.
"""

import asyncio
import os
import sys
import argparse
import json
import subprocess
import stat
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

# Check for required dependencies
missing_deps = []
try:
    import httpx
except ImportError:
    missing_deps.append("httpx")

try:
    import importlib.util
    has_anthropic = importlib.util.find_spec("anthropic") is not None
except ImportError:
    has_anthropic = False

try:
    from dotenv import load_dotenv
    has_dotenv = True
except ImportError:
    has_dotenv = False

# Graceful import for optional dependencies
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    has_rich = True
except ImportError:
    has_rich = False

try:
    from InquirerPy import inquirer
    has_inquirer = True
except ImportError:
    has_inquirer = False

try:
    import nest_asyncio
    has_nest_asyncio = True
except ImportError:
    has_nest_asyncio = False

# Handle missing critical dependencies
if missing_deps:
    print("Error: Missing critical dependencies:")
    for dep in missing_deps:
        print(f"  - {dep}")
    print("\nPlease install the required dependencies:")
    print("pip install -r requirements.txt")
    sys.exit(1)

# Load environment variables if available
if has_dotenv:
    load_dotenv()

# Create console (with graceful degradation)
if has_rich:
    console = Console()
else:
    # Simple console fallback (MIRROR pattern from claude_token_tracker.py)
    class SimpleConsole:
        def print(self, text):
            # Remove rich formatting
            text = str(text).replace("[bold]", "").replace("[/bold]", "")
            text = text.replace("[bold red]", "").replace("[/bold red]", "")
            text = text.replace("[bold blue]", "").replace("[/bold blue]", "")
            text = text.replace("[bold green]", "").replace("[/bold green]", "")
            text = text.replace("[bold yellow]", "").replace("[/bold yellow]", "")
            text = text.replace("[yellow]", "").replace("[/yellow]", "")
            text = text.replace("[green]", "").replace("[/green]", "")
            text = text.replace("[cyan]", "").replace("[/cyan]", "")
            print(text)
        
        def status(self, text):
            class DummyContextManager:
                def __enter__(self): 
                    print(text.replace("[bold green]", "").replace("[/bold green]", ""))
                    return self
                def __exit__(self, *args): 
                    pass
            return DummyContextManager()
    
    console = SimpleConsole()  # type: ignore[assignment]


@dataclass
class ClaudeModel:
    """Represents a Claude model from the API."""
    id: str
    display_name: str
    created_at: datetime
    type: str = "model"
    
    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "ClaudeModel":
        """Create ClaudeModel from API response data."""
        created_at = datetime.fromisoformat(data["created"].replace("Z", "+00:00"))
        return cls(
            id=data["id"],
            display_name=data.get("display_name", data["id"]),
            created_at=created_at,
            type=data.get("type", "model")
        )


@dataclass
class ClaudeCLICredentials:
    """Claude CLI stored credential information."""
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_at: Optional[datetime] = None
    subscription_type: Optional[str] = None
    scopes: List[str] = field(default_factory=list)
    api_key: Optional[str] = None
    source: str = "unknown"  # "oauth", "api_key", "environment"
    
    def __post_init__(self):
        if self.scopes is None:
            self.scopes = []


@dataclass
class AuthProfile:
    """Authentication profile for different subscription tiers."""
    name: str
    api_key_env_var: str
    display_name: str
    tier: Optional[str] = None  # "free", "pro", "max", or None for auto-detect
    rate_limit_rpm: Optional[int] = None  # Requests per minute
    priority_models: List[str] = field(default_factory=list)  # Models exclusive to this tier
    use_claude_cli_auth: bool = False  # Enable Claude CLI authentication integration
    claude_cli_fallback: bool = True  # Enable fallback to Claude CLI when env var missing
    
    def __post_init__(self):
        if self.priority_models is None:
            self.priority_models = []
    
    def get_api_key(self) -> Optional[str]:
        """Get API key with Claude CLI fallback support."""
        # Try environment variable first (existing behavior)
        api_key = os.environ.get(self.api_key_env_var)
        if api_key:
            return api_key
        
        # Try Claude CLI stored credentials if enabled
        if self.claude_cli_fallback:
            cli_provider = ClaudeCLIAuthProvider()
            if credentials := cli_provider.detect_claude_cli_auth():
                if cli_provider.validate_credentials(credentials):
                    return cli_provider.get_api_key_from_credentials(credentials)
        
        return None


@dataclass
class LauncherConfig:
    """Configuration for the Claude launcher."""
    last_selected_model: Optional[str] = None
    active_profile: str = "default"
    cache_duration_hours: int = 24
    default_args: List[str] = field(default_factory=list)
    auth_profiles: Dict[str, AuthProfile] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.default_args is None:
            self.default_args = []
        if self.auth_profiles is None or not self.auth_profiles:
            # Create default profile
            self.auth_profiles = {
                "default": AuthProfile(
                    name="default",
                    api_key_env_var="ANTHROPIC_API_KEY",
                    display_name="Default API Key"
                )
            }
    
    def get_active_profile(self) -> AuthProfile:
        """Get the currently active authentication profile."""
        return self.auth_profiles.get(self.active_profile, self.auth_profiles["default"])
    
    def add_profile(self, profile: AuthProfile) -> None:
        """Add a new authentication profile."""
        self.auth_profiles[profile.name] = profile
    
    def list_profiles(self) -> List[str]:
        """List all available profile names."""
        return list(self.auth_profiles.keys())
    
    @classmethod
    def load_from_file(cls, config_path: Path) -> "LauncherConfig":
        """Load configuration from JSON file."""
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    data = json.load(f)
                
                # Handle auth_profiles conversion
                if "auth_profiles" in data:
                    profiles = {}
                    for name, profile_data in data["auth_profiles"].items():
                        profiles[name] = AuthProfile(**profile_data)
                    data["auth_profiles"] = profiles
                
                return cls(**data)
            except (json.JSONDecodeError, TypeError) as e:
                console.print(f"[bold yellow]Warning:[/bold yellow] Could not load config: {e}")
                console.print("Using default configuration.")
        return cls()
    
    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to JSON file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict with proper serialization
        data = asdict(self)
        
        # Ensure auth_profiles are properly serialized
        if "auth_profiles" in data:
            profiles = {}
            for name, profile in self.auth_profiles.items():
                profiles[name] = asdict(profile)
            data["auth_profiles"] = profiles
        
        with open(config_path, 'w') as f:
            json.dump(data, f, indent=2)


@dataclass
class SubscriptionInfo:
    """Information about API subscription tier."""
    tier: str  # "free", "pro", "max", or "unknown"
    rate_limit_rpm: Optional[int] = None
    rate_limit_tokens_per_minute: Optional[int] = None
    available_models: List[str] = field(default_factory=list)
    priority_access: bool = False


class AnthropicAPIClient:
    """Client for Anthropic API with retry logic and tier detection."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1"
        self.headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        self.detected_tier: Optional[SubscriptionInfo] = None
    
    async def fetch_models(self, max_retries: int = 5) -> List[ClaudeModel]:
        """Fetch available Claude models with retry logic."""
        async with httpx.AsyncClient() as client:
            for attempt in range(max_retries):
                try:
                    response = await client.get(
                        f"{self.base_url}/models",
                        headers=self.headers,
                        timeout=30.0
                    )
                    
                    if response.status_code == 429:
                        # Rate limiting - implement exponential backoff
                        retry_after = int(response.headers.get("retry-after", 60))
                        wait_time = min(retry_after, 2 ** attempt)
                        console.print(f"[yellow]Rate limited. Waiting {wait_time}s...[/yellow]")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    response.raise_for_status()
                    data = response.json()
                    
                    # Parse and validate response
                    models = []
                    for model_data in data.get("data", []):
                        if model_data.get("type") == "model":
                            models.append(ClaudeModel.from_api_response(model_data))
                    
                    return models
                    
                except httpx.TimeoutException:
                    console.print(f"[yellow]Request timeout (attempt {attempt + 1}/{max_retries})[/yellow]")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        raise
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 401:
                        raise ValueError("Invalid API key")
                    elif e.response.status_code == 403:
                        raise ValueError("API access forbidden")
                    else:
                        console.print(f"[yellow]HTTP error {e.response.status_code} (attempt {attempt + 1}/{max_retries})[/yellow]")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                        else:
                            raise
                except Exception as e:
                    console.print(f"[yellow]Request failed: {e} (attempt {attempt + 1}/{max_retries})[/yellow]")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        raise
        
        return []
    
    async def detect_subscription_tier(self) -> SubscriptionInfo:
        """Detect subscription tier based on API responses and rate limits."""
        if self.detected_tier:
            return self.detected_tier
        
        try:
            # Fetch models to analyze available tiers
            models = await self.fetch_models()
            model_ids = [model.id for model in models]
            
            # Analyze rate limit headers from API calls
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=self.headers,
                    timeout=30.0
                )
                
                # Extract rate limit information from headers
                rpm_limit = self._extract_rate_limit(response.headers, "requests-limit")
                tokens_limit = self._extract_rate_limit(response.headers, "tokens-limit")
                
                # Tier detection logic based on model availability and rate limits
                tier = self._infer_tier_from_models_and_limits(model_ids, rpm_limit, tokens_limit)
                
                self.detected_tier = SubscriptionInfo(
                    tier=tier,
                    rate_limit_rpm=rpm_limit,
                    rate_limit_tokens_per_minute=tokens_limit,
                    available_models=model_ids,
                    priority_access=tier in ["pro", "max"]
                )
                
                return self.detected_tier
                
        except Exception as e:
            console.print(f"[yellow]Warning: Could not detect subscription tier: {e}[/yellow]")
            return SubscriptionInfo(tier="unknown")
    
    def _extract_rate_limit(self, headers: Dict[str, str], limit_type: str) -> Optional[int]:
        """Extract rate limit from response headers."""
        # Common rate limit header patterns
        possible_headers = [
            f"x-ratelimit-limit-{limit_type}",
            f"x-ratelimit-{limit_type}",
            f"ratelimit-limit-{limit_type}",
            f"anthropic-ratelimit-{limit_type}",
        ]
        
        for header in possible_headers:
            if header in headers:
                try:
                    return int(headers[header])
                except ValueError:
                    continue
        return None
    
    def _infer_tier_from_models_and_limits(self, model_ids: List[str], rpm_limit: Optional[int], tokens_limit: Optional[int]) -> str:
        """Infer subscription tier from available models and rate limits."""
        # Model availability patterns (these are hypothetical - adjust based on actual Anthropic tiers)
        premium_models = [
            "claude-3-opus",
            "claude-sonnet-4", 
            "claude-opus-4"
        ]
        
        max_exclusive_models = [
            "claude-4-max",
            "claude-experimental"
        ]
        
        # Check for Max tier indicators
        if any(model_id for model_id in model_ids if any(exclusive in model_id for exclusive in max_exclusive_models)):
            return "max"
        
        # Check for Pro tier indicators
        if any(model_id for model_id in model_ids if any(premium in model_id for premium in premium_models)):
            # Rate limit analysis for Pro vs Max
            if rpm_limit and rpm_limit >= 5000:  # Hypothetical Max tier limit
                return "max"
            elif rpm_limit and rpm_limit >= 1000:  # Hypothetical Pro tier limit  
                return "pro"
            else:
                return "pro"  # Has premium models but lower rate limit
        
        # Fallback to rate limit analysis
        if rpm_limit:
            if rpm_limit >= 5000:
                return "max"
            elif rpm_limit >= 1000:
                return "pro"
            else:
                return "free"
        
        # Default to unknown if we can't determine
        return "unknown"


class ClaudeCLIAuthProvider:
    """Provider for Claude CLI authentication integration."""
    
    def __init__(self):
        self.credentials_path = Path.home() / ".claude" / ".credentials.json"
        self.config_path = Path.home() / ".claude" / "launcher_config.json"
    
    def detect_claude_cli_auth(self) -> Optional[ClaudeCLICredentials]:
        """Detect and validate Claude CLI stored authentication."""
        # Try OAuth credentials first
        if oauth_creds := self._load_oauth_credentials():
            return oauth_creds
        
        # Fallback to API key detection
        if api_key_creds := self._detect_api_key_credentials():
            return api_key_creds
        
        return None
    
    def _load_oauth_credentials(self) -> Optional[ClaudeCLICredentials]:
        """Load and validate OAuth credentials from Claude CLI."""
        if not self.credentials_path.exists():
            return None
        
        # Security check: verify file permissions
        if not self._verify_file_security(self.credentials_path):
            console.print("[yellow]Warning: Claude CLI credentials file has insecure permissions[/yellow]")
            return None
        
        try:
            with open(self.credentials_path, 'r') as f:
                data = json.load(f)
            
            oauth_data = data.get("claudeAiOauth", {})
            if not oauth_data:
                return None
            
            credentials = ClaudeCLICredentials(
                access_token=oauth_data.get("accessToken"),
                refresh_token=oauth_data.get("refreshToken"),
                expires_at=self._parse_expires_at(oauth_data.get("expiresAt")),
                subscription_type=oauth_data.get("subscriptionType"),
                scopes=oauth_data.get("scopes", []),
                source="oauth"
            )
            
            # Validate token is not expired
            if self._is_token_expired(credentials):
                console.print("[yellow]Claude CLI OAuth token is expired[/yellow]")
                return None
            
            return credentials
            
        except (json.JSONDecodeError, KeyError, PermissionError) as e:
            console.print(f"[yellow]Warning: Could not read Claude CLI credentials: {e}[/yellow]")
            return None
    
    def _detect_api_key_credentials(self) -> Optional[ClaudeCLICredentials]:
        """Detect API key credentials from environment or config."""
        # Check common environment variables
        env_vars = ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "ANTHROPIC_PRO_KEY", "ANTHROPIC_MAX_KEY"]
        
        for env_var in env_vars:
            api_key = os.environ.get(env_var)
            if api_key and self._is_valid_api_key_format(api_key):
                return ClaudeCLICredentials(
                    api_key=api_key,
                    source="api_key"
                )
        
        return None
    
    def _verify_file_security(self, file_path: Path) -> bool:
        """Verify file has secure permissions (600 or more restrictive)."""
        try:
            stat_info = file_path.stat()
            permissions = stat_info.st_mode & 0o777
            # Allow 600 (owner rw) or more restrictive
            return permissions <= 0o600
        except OSError:
            return False
    
    def _parse_expires_at(self, expires_at_data: Optional[Any]) -> Optional[datetime]:
        """Parse expiration timestamp from OAuth data."""
        if not expires_at_data:
            return None
        
        try:
            # Handle string format
            if isinstance(expires_at_data, str):
                # Try ISO format first
                if expires_at_data.endswith('Z'):
                    expires_at_data = expires_at_data[:-1] + '+00:00'
                return datetime.fromisoformat(expires_at_data)
            else:
                # Handle numeric timestamp (milliseconds)
                timestamp = float(expires_at_data) / 1000  # Convert from milliseconds
                return datetime.fromtimestamp(timestamp)
        except (ValueError, TypeError):
            try:
                # Fallback: try as timestamp
                timestamp = float(expires_at_data) / 1000  # Convert from milliseconds
                return datetime.fromtimestamp(timestamp)
            except (ValueError, TypeError):
                return None
    
    def _is_token_expired(self, credentials: ClaudeCLICredentials) -> bool:
        """Check if OAuth token is expired or expiring soon."""
        if not credentials.expires_at:
            return False  # No expiration info, assume valid
        
        # Consider expired if expires within 5 minutes
        threshold = datetime.now() + timedelta(minutes=5)
        return credentials.expires_at <= threshold
    
    def _is_valid_api_key_format(self, api_key: str) -> bool:
        """Check if API key has valid format."""
        return api_key.startswith("sk-") and len(api_key) >= 10
    
    def validate_credentials(self, credentials: ClaudeCLICredentials) -> bool:
        """Validate credentials are usable for API access."""
        if credentials.source == "oauth" and credentials.access_token:
            return self._validate_oauth_token(credentials.access_token)
        elif credentials.source == "api_key" and credentials.api_key:
            try:
                validate_api_key(credentials.api_key)
                return True
            except ValueError:
                return False
        return False
    
    def _validate_oauth_token(self, access_token: str) -> bool:
        """Validate OAuth access token format."""
        # OAuth tokens start with sk-ant-oat01-
        return access_token.startswith("sk-ant-oat01-") and len(access_token) > 20
    
    def get_api_key_from_credentials(self, credentials: ClaudeCLICredentials) -> Optional[str]:
        """Extract usable API key from Claude CLI credentials."""
        if credentials.source == "oauth" and credentials.access_token:
            # OAuth access tokens can be used directly as API keys
            return credentials.access_token
        elif credentials.source == "api_key" and credentials.api_key:
            return credentials.api_key
        return None


class AuthenticationAuditor:
    """Audit logging for authentication events."""
    
    def __init__(self):
        self.log_path = Path.home() / ".claude" / "auth.log"
        # Ensure log directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log_auth_attempt(self, source: str, success: bool, profile: str, details: str = ""):
        """Log authentication attempt with timestamp."""
        timestamp = datetime.now().isoformat()
        status = "SUCCESS" if success else "FAILURE"
        log_entry = f"{timestamp} | {status} | {source} | {profile} | {details}\\n"
        
        try:
            with open(self.log_path, 'a') as f:
                f.write(log_entry)
        except PermissionError:
            # Fail silently for audit logging
            pass
    
    def log_cli_credential_access(self, credentials: ClaudeCLICredentials, profile: str):
        """Log Claude CLI credential access."""
        details = f"source={credentials.source}, subscription={credentials.subscription_type}"
        self.log_auth_attempt("claude_cli", True, profile, details)
    
    def log_profile_creation(self, profile_name: str, env_var: str):
        """Log profile creation event."""
        details = f"env_var={env_var}"
        self.log_auth_attempt("profile_creation", True, profile_name, details)
    
    def log_profile_switch(self, from_profile: str, to_profile: str):
        """Log profile switching event."""
        details = f"from={from_profile}, to={to_profile}"
        self.log_auth_attempt("profile_switch", True, to_profile, details)


def validate_claude_cli_integration_security() -> bool:
    """Validate security of Claude CLI integration."""
    cli_provider = ClaudeCLIAuthProvider()
    security_issues = []
    
    # Check file permissions
    if cli_provider.credentials_path.exists():
        if not cli_provider._verify_file_security(cli_provider.credentials_path):
            security_issues.append(f"Claude CLI credentials file has insecure permissions: {cli_provider.credentials_path}")
    
    # Check log file permissions
    auditor = AuthenticationAuditor()
    if auditor.log_path.exists():
        try:
            stat_info = auditor.log_path.stat()
            permissions = stat_info.st_mode & 0o777
            if permissions > 0o600:
                security_issues.append(f"Auth log file has insecure permissions: {auditor.log_path}")
        except OSError:
            pass
    
    # Report security issues
    if security_issues:
        console.print("[bold red]Security Warning:[/bold red] Found security issues:")
        for issue in security_issues:
            console.print(f"  - {issue}")
        console.print("\\n[bold]Recommended fixes:[/bold]")
        if cli_provider.credentials_path.exists():
            console.print(f"  chmod 600 {cli_provider.credentials_path}")
        if auditor.log_path.exists():
            console.print(f"  chmod 600 {auditor.log_path}")
        return False
    
    return True


def parse_args():
    """Parse command line arguments (MIRROR pattern from claude_token_tracker.py)."""
    parser = argparse.ArgumentParser(
        description="Launch Claude CLI with interactive model selection"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available Claude models and exit"
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        help="Path to configuration file (default: ~/.claude/launcher_config.json)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show selected model without launching Claude CLI"
    )
    parser.add_argument(
        "--cache-bust",
        action="store_true",
        help="Force refresh of model cache"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output"
    )
    parser.add_argument(
        "--profile",
        type=str,
        help="Use specific authentication profile"
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available authentication profiles"
    )
    parser.add_argument(
        "--add-profile",
        type=str,
        help="Add new authentication profile (format: name:env_var:display_name:tier)"
    )
    parser.add_argument(
        "--detect-tier",
        action="store_true",
        help="Detect and display current subscription tier information"
    )
    parser.add_argument(
        "--enable-cli-auth",
        action="store_true",
        help="Enable Claude CLI authentication fallback for current profile"
    )
    parser.add_argument(
        "--cli-auth-status",
        action="store_true",
        help="Check Claude CLI authentication status and compatibility"
    )
    parser.add_argument(
        "--sync-from-cli",
        action="store_true",
        help="Sync authentication from Claude CLI to current profile"
    )
    # Additional arguments are passed through to Claude CLI
    parser.add_argument(
        "claude_args",
        nargs="*",
        help="Additional arguments to pass to Claude CLI"
    )
    
    return parser.parse_args()


def get_api_key(config: LauncherConfig) -> str:
    """Enhanced API key retrieval with Claude CLI authentication fallback."""
    profile = config.get_active_profile()
    
    # Try profile's primary method (includes Claude CLI fallback if enabled)
    if api_key := profile.get_api_key():
        return api_key
    
    # Detailed error with Claude CLI guidance
    cli_provider = ClaudeCLIAuthProvider()
    cli_credentials = cli_provider.detect_claude_cli_auth()
    
    console.print("[bold red]Error:[/bold red] No API key found.")
    console.print(f"Profile: {profile.display_name} ({profile.name})")
    console.print(f"Environment variable {profile.api_key_env_var} not set.")
    
    if cli_credentials:
        console.print(f"[yellow]Found Claude CLI credentials ({cli_credentials.source})[/yellow]")
        if cli_credentials.source == "oauth" and cli_provider._is_token_expired(cli_credentials):
            console.print("[yellow]However, OAuth token is expired. Please run 'claude doctor' to refresh.[/yellow]")
        else:
            console.print("[yellow]Enable Claude CLI fallback with: --enable-cli-auth[/yellow]")
    else:
        console.print("[yellow]No Claude CLI credentials found. Try 'claude doctor' to authenticate.[/yellow]")
    
    console.print(f"\\nTo use a different profile: --profile <name>")
    console.print(f"Available profiles: {', '.join(config.list_profiles())}")
    sys.exit(1)


def validate_api_key(api_key: str) -> bool:
    """Validate API key format and basic structure."""
    if not api_key:
        raise ValueError("API key not found")
    
    if not api_key.startswith("sk-"):
        raise ValueError("Invalid API key format")
    
    if len(api_key) < 10:
        raise ValueError("API key too short")
    
    return True


def get_config_path(args) -> Path:
    """Get configuration file path."""
    if args.config_path:
        return args.config_path
    
    # Default to user's .claude directory
    claude_dir = Path.home() / ".claude"
    return claude_dir / "launcher_config.json"


def get_cache_path() -> Path:
    """Get cache file path."""
    cache_dir = Path.home() / ".cache" / "claude_launcher"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "models.json"


def load_cached_models(cache_path: Path, cache_duration_hours: int) -> Optional[List[ClaudeModel]]:
    """Load models from cache if valid."""
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        cache_time = datetime.fromisoformat(cache_data["timestamp"])
        if datetime.now() - cache_time < timedelta(hours=cache_duration_hours):
            models = []
            for model_data in cache_data["models"]:
                # Reconstruct datetime object
                model_data["created_at"] = datetime.fromisoformat(model_data["created_at"])
                models.append(ClaudeModel(**model_data))
            return models
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        console.print(f"[yellow]Warning: Invalid cache file: {e}[/yellow]")
    
    return None


def save_cached_models(cache_path: Path, models: List[ClaudeModel]) -> None:
    """Save models to cache."""
    cache_data = {
        "timestamp": datetime.now().isoformat(),
        "models": [asdict(model) for model in models]
    }
    
    # Convert datetime objects to strings for JSON serialization
    for model_data in cache_data["models"]:
        if isinstance(model_data, dict) and isinstance(model_data.get("created_at"), datetime):
            model_data["created_at"] = model_data["created_at"].isoformat()
    
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f, indent=2)


async def fetch_available_models(config: LauncherConfig, force_refresh: bool = False) -> List[ClaudeModel]:
    """Fetch available models with caching."""
    cache_path = get_cache_path()
    
    # Try to load from cache first
    if not force_refresh:
        cached_models = load_cached_models(cache_path, config.cache_duration_hours)
        if cached_models:
            return cached_models
    
    # Fetch from API
    api_key = get_api_key(config)
    validate_api_key(api_key)
    
    client = AnthropicAPIClient(api_key)
    models = await client.fetch_models()
    
    # Save to cache
    save_cached_models(cache_path, models)
    
    return models


def select_model_interactive(models: List[ClaudeModel], config: LauncherConfig) -> ClaudeModel:
    """Interactive model selection with fallback."""
    if has_inquirer:
        # Use InquirerPy for interactive selection
        choices = []
        default_choice = None
        
        for i, model in enumerate(models):
            choice_name = f"{model.display_name} ({model.id})"
            choices.append(choice_name)
            
            # Set default to last selected model
            if model.id == config.last_selected_model:
                default_choice = choice_name
        
        if not default_choice and choices:
            default_choice = choices[0]
        
        try:
            selected_name = inquirer.select(
                message="Select Claude model:",
                choices=choices,
                default=default_choice
            ).execute()
            
            # Find the corresponding model
            for model in models:
                if f"{model.display_name} ({model.id})" == selected_name:
                    return model
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Selection cancelled.[/bold yellow]")
            sys.exit(0)
    
    # Fallback to simple numbered selection
    console.print("\n[bold]Available Claude models:[/bold]")
    for i, model in enumerate(models, 1):
        marker = " (last used)" if model.id == config.last_selected_model else ""
        console.print(f"{i}. {model.display_name} ({model.id}){marker}")
    
    while True:
        try:
            choice = input("\nSelect model (number): ").strip()
            index = int(choice) - 1
            if 0 <= index < len(models):
                return models[index]
            else:
                console.print("[bold red]Invalid selection. Please try again.[/bold red]")
        except (ValueError, KeyboardInterrupt):
            console.print("\n[bold yellow]Selection cancelled.[/bold yellow]")
            sys.exit(0)


def launch_claude_cli(model: ClaudeModel, args: List[str], dry_run: bool = False) -> None:
    """Launch Claude CLI with selected model (MIRROR pattern from prp_runner.py)."""
    # Build command
    cmd = ["claude", "--model", model.id] + args
    
    if dry_run:
        console.print(f"[bold]Would execute:[/bold] {' '.join(cmd)}")
        return
    
    # Environment variable inheritance
    env = os.environ.copy()
    
    console.print(f"[bold green]Launching Claude CLI with model:[/bold green] {model.display_name}")
    console.print(f"[bold]Command:[/bold] {' '.join(cmd)}")
    
    try:
        # PATTERN: Subprocess execution with streaming output
        subprocess.run(cmd, env=env, check=True)
    except FileNotFoundError:
        console.print("[bold red]Error:[/bold red] Claude CLI not found.")
        console.print("Please ensure Claude CLI is installed and in your PATH.")
        console.print("Installation: https://docs.anthropic.com/en/docs/claude-code")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Error launching Claude CLI:[/bold red] {e}")
        console.print(f"Return code: {e.returncode}")
        sys.exit(1)


def list_profiles_command(config: LauncherConfig) -> None:
    """List available authentication profiles."""
    if has_rich:
        table = Table(title="Authentication Profiles", box=box.MINIMAL_HEAVY_HEAD)
        table.add_column("Name", style="cyan")
        table.add_column("Display Name", style="green")
        table.add_column("Environment Variable", style="yellow")
        table.add_column("Tier", style="magenta")
        table.add_column("Active", style="bold green")
        
        for name, profile in config.auth_profiles.items():
            is_active = "✓" if name == config.active_profile else ""
            tier = profile.tier or "auto-detect"
            table.add_row(
                name,
                profile.display_name,
                profile.api_key_env_var,
                tier,
                is_active
            )
        
        console.print(table)
    else:
        console.print("\\nAuthentication Profiles:")
        console.print("-" * 50)
        for name, profile in config.auth_profiles.items():
            active_marker = " (ACTIVE)" if name == config.active_profile else ""
            tier = profile.tier or "auto-detect"
            console.print(f"Name: {name}{active_marker}")
            console.print(f"Display: {profile.display_name}")
            console.print(f"Env Var: {profile.api_key_env_var}")
            console.print(f"Tier: {tier}")
            console.print("-" * 50)


def add_profile_command(config: LauncherConfig, profile_spec: str, config_path: Path) -> None:
    """Add a new authentication profile."""
    try:
        parts = profile_spec.split(":")
        if len(parts) < 3:
            raise ValueError("Invalid format. Use: name:env_var:display_name[:tier]")
        
        name = parts[0].strip()
        env_var = parts[1].strip()
        display_name = parts[2].strip()
        tier = parts[3].strip() if len(parts) > 3 else None
        
        if name in config.auth_profiles:
            console.print(f"[bold yellow]Profile '{name}' already exists. Updating...[/bold yellow]")
        
        profile = AuthProfile(
            name=name,
            api_key_env_var=env_var,
            display_name=display_name,
            tier=tier
        )
        
        config.add_profile(profile)
        config.save_to_file(config_path)
        
        # Log profile creation
        auditor = AuthenticationAuditor()
        auditor.log_profile_creation(name, env_var)
        
        console.print(f"[bold green]Profile '{name}' added successfully![/bold green]")
        console.print(f"Display Name: {display_name}")
        console.print(f"Environment Variable: {env_var}")
        if tier:
            console.print(f"Tier: {tier}")
        
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        console.print("Format: name:env_var:display_name[:tier]")
        console.print("Example: pro:ANTHROPIC_PRO_KEY:Claude Pro Account:pro")
        sys.exit(1)


async def cli_auth_status_command(config: LauncherConfig) -> None:
    """Check and display Claude CLI authentication status."""
    cli_provider = ClaudeCLIAuthProvider()
    credentials = cli_provider.detect_claude_cli_auth()
    
    if has_rich:
        table = Table(title="Claude CLI Authentication Status", box=box.MINIMAL_HEAVY_HEAD)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        if credentials:
            table.add_row("Authentication Found", "✓")
            table.add_row("Source", credentials.source.upper())
            
            if credentials.source == "oauth":
                table.add_row("Subscription Type", credentials.subscription_type or "Unknown")
                table.add_row("Scopes", ", ".join(credentials.scopes))
                if credentials.expires_at:
                    expired = cli_provider._is_token_expired(credentials)
                    status = "EXPIRED" if expired else "Valid"
                    table.add_row("Token Status", status)
                    table.add_row("Expires At", credentials.expires_at.strftime("%Y-%m-%d %H:%M:%S"))
            
            # Test API access
            if api_key := cli_provider.get_api_key_from_credentials(credentials):
                try:
                    validate_api_key(api_key)
                    table.add_row("API Access", "✓ Valid")
                except ValueError as e:
                    table.add_row("API Access", f"✗ {e}")
        else:
            table.add_row("Authentication Found", "✗")
            table.add_row("Recommendation", "Run 'claude doctor' to authenticate")
        
        console.print(table)
    else:
        console.print("\\nClaude CLI Authentication Status:")
        console.print("-" * 50)
        if credentials:
            console.print(f"Authentication Found: ✓")
            console.print(f"Source: {credentials.source.upper()}")
            
            if credentials.source == "oauth":
                console.print(f"Subscription Type: {credentials.subscription_type or 'Unknown'}")
                console.print(f"Scopes: {', '.join(credentials.scopes)}")
                if credentials.expires_at:
                    expired = cli_provider._is_token_expired(credentials)
                    status = "EXPIRED" if expired else "Valid"
                    console.print(f"Token Status: {status}")
                    console.print(f"Expires At: {credentials.expires_at.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Test API access
            if api_key := cli_provider.get_api_key_from_credentials(credentials):
                try:
                    validate_api_key(api_key)
                    console.print("API Access: ✓ Valid")
                except ValueError as e:
                    console.print(f"API Access: ✗ {e}")
        else:
            console.print("Authentication Found: ✗")
            console.print("Recommendation: Run 'claude doctor' to authenticate")
    
    # Show integration recommendations
    if credentials and not config.get_active_profile().claude_cli_fallback:
        console.print("\\n[bold]Recommendation:[/bold] Enable Claude CLI fallback with --enable-cli-auth")


def enable_cli_auth_command(config: LauncherConfig, config_path: Path) -> None:
    """Enable Claude CLI authentication fallback for current profile."""
    profile = config.get_active_profile()
    
    # Check if Claude CLI credentials are available
    cli_provider = ClaudeCLIAuthProvider()
    credentials = cli_provider.detect_claude_cli_auth()
    
    if not credentials:
        console.print("[bold red]Error:[/bold red] No Claude CLI credentials found.")
        console.print("Please run 'claude doctor' to authenticate with Claude CLI first.")
        sys.exit(1)
    
    if not cli_provider.validate_credentials(credentials):
        console.print("[bold red]Error:[/bold red] Claude CLI credentials are invalid or expired.")
        console.print("Please run 'claude doctor' to refresh your authentication.")
        sys.exit(1)
    
    # Enable CLI fallback for the profile
    profile.claude_cli_fallback = True
    config.save_to_file(config_path)
    
    console.print(f"[bold green]✓ Enabled Claude CLI authentication fallback for profile '{profile.name}'[/bold green]")
    console.print(f"Found {credentials.source} credentials: {credentials.subscription_type or 'API Key'}")


def sync_from_cli_command(config: LauncherConfig, config_path: Path) -> None:
    """Sync authentication information from Claude CLI to current profile."""
    profile = config.get_active_profile()
    cli_provider = ClaudeCLIAuthProvider()
    credentials = cli_provider.detect_claude_cli_auth()
    
    if not credentials:
        console.print("[bold red]Error:[/bold red] No Claude CLI credentials found.")
        console.print("Please run 'claude doctor' to authenticate with Claude CLI first.")
        sys.exit(1)
    
    # Extract and set API key in environment if it's available
    if api_key := cli_provider.get_api_key_from_credentials(credentials):
        console.print(f"[bold]Found Claude CLI credentials ({credentials.source})[/bold]")
        
        if credentials.source == "oauth":
            console.print(f"Subscription: {credentials.subscription_type or 'Unknown'}")
            console.print(f"Scopes: {', '.join(credentials.scopes)}")
            if credentials.expires_at:
                console.print(f"Expires: {credentials.expires_at.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Update profile configuration
        profile.claude_cli_fallback = True
        if credentials.subscription_type:
            profile.tier = credentials.subscription_type.lower()
        
        config.save_to_file(config_path)
        
        console.print(f"[bold green]✓ Synced authentication from Claude CLI to profile '{profile.name}'[/bold green]")
        console.print(f"Profile tier updated to: {profile.tier or 'auto-detect'}")
        console.print("\\n[bold]Note:[/bold] The OAuth token will be used automatically when environment variable is not set.")
    else:
        console.print("[bold red]Error:[/bold red] Could not extract usable credentials from Claude CLI.")
        sys.exit(1)


async def detect_tier_command(config: LauncherConfig) -> None:
    """Detect and display subscription tier information."""
    profile = config.get_active_profile()
    
    with console.status(f"[bold green]Detecting subscription tier for {profile.display_name}...[/bold green]"):
        try:
            api_key = get_api_key(config)
            validate_api_key(api_key)
            
            client = AnthropicAPIClient(api_key)
            subscription_info = await client.detect_subscription_tier()
            
        except SystemExit:
            return
        except Exception as e:
            console.print(f"[bold red]Error detecting tier:[/bold red] {e}")
            return
    
    # Display results
    if has_rich:
        table = Table(title=f"Subscription Information - {profile.display_name}", box=box.MINIMAL_HEAVY_HEAD)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Profile", f"{profile.name} ({profile.display_name})")
        table.add_row("Environment Variable", profile.api_key_env_var)
        table.add_row("Detected Tier", subscription_info.tier.upper())
        
        if subscription_info.rate_limit_rpm:
            table.add_row("Rate Limit (RPM)", str(subscription_info.rate_limit_rpm))
        
        if subscription_info.rate_limit_tokens_per_minute:
            table.add_row("Token Limit (TPM)", str(subscription_info.rate_limit_tokens_per_minute))
        
        table.add_row("Priority Access", "✓" if subscription_info.priority_access else "✗")
        table.add_row("Available Models", str(len(subscription_info.available_models)))
        
        console.print(table)
        
        # Show available models if verbose
        if subscription_info.available_models:
            models_table = Table(title="Available Models", box=box.MINIMAL_HEAVY_HEAD)
            models_table.add_column("Model ID", style="cyan")
            
            for model_id in sorted(subscription_info.available_models):
                models_table.add_row(model_id)
            
            console.print(models_table)
    else:
        console.print(f"\\nSubscription Information - {profile.display_name}")
        console.print("-" * 50)
        console.print(f"Profile: {profile.name} ({profile.display_name})")
        console.print(f"Environment Variable: {profile.api_key_env_var}")
        console.print(f"Detected Tier: {subscription_info.tier.upper()}")
        
        if subscription_info.rate_limit_rpm:
            console.print(f"Rate Limit (RPM): {subscription_info.rate_limit_rpm}")
        
        if subscription_info.rate_limit_tokens_per_minute:
            console.print(f"Token Limit (TPM): {subscription_info.rate_limit_tokens_per_minute}")
        
        console.print(f"Priority Access: {'✓' if subscription_info.priority_access else '✗'}")
        console.print(f"Available Models: {len(subscription_info.available_models)}")
        
        if subscription_info.available_models:
            console.print("\\nAvailable Models:")
            for model_id in sorted(subscription_info.available_models):
                console.print(f"  - {model_id}")


async def list_models_command(config: LauncherConfig, force_refresh: bool = False) -> None:
    """List available models command."""
    with console.status("[bold green]Fetching available models...[/bold green]"):
        models = await fetch_available_models(config, force_refresh)
    
    if not models:
        console.print("[bold red]No models found.[/bold red]")
        return
    
    # Display models in a table
    if has_rich:
        table = Table(title="Available Claude Models", box=box.MINIMAL_HEAVY_HEAD)
        table.add_column("ID", style="cyan")
        table.add_column("Display Name", style="green")
        table.add_column("Created", style="yellow")
        
        for model in models:
            table.add_row(
                model.id,
                model.display_name,
                model.created_at.strftime("%Y-%m-%d")
            )
        
        console.print(table)
    else:
        console.print("\nAvailable Claude Models:")
        console.print("-" * 50)
        for model in models:
            console.print(f"ID: {model.id}")
            console.print(f"Name: {model.display_name}")
            console.print(f"Created: {model.created_at.strftime('%Y-%m-%d')}")
            console.print("-" * 50)


async def main():
    """Main function."""
    args = parse_args()
    
    # Load configuration
    config_path = get_config_path(args)
    config = LauncherConfig.load_from_file(config_path)
    
    # Handle profile selection
    if args.profile:
        if args.profile not in config.auth_profiles:
            console.print(f"[bold red]Error:[/bold red] Profile '{args.profile}' not found.")
            console.print(f"Available profiles: {', '.join(config.list_profiles())}")
            sys.exit(1)
        
        # Log profile switch
        auditor = AuthenticationAuditor()
        old_profile = config.active_profile
        config.active_profile = args.profile
        auditor.log_profile_switch(old_profile, args.profile)
        config.save_to_file(config_path)
    
    if args.verbose:
        profile = config.get_active_profile()
        console.print(f"[bold]Config path:[/bold] {config_path}")
        console.print(f"[bold]Active profile:[/bold] {profile.display_name} ({profile.name})")
        console.print(f"[bold]Cache duration:[/bold] {config.cache_duration_hours} hours")
    
    # Handle profile management commands
    if args.list_profiles:
        list_profiles_command(config)
        return
    
    if args.add_profile:
        add_profile_command(config, args.add_profile, config_path)
        return
    
    if args.detect_tier:
        await detect_tier_command(config)
        return
    
    # Handle Claude CLI authentication commands
    if args.cli_auth_status:
        await cli_auth_status_command(config)
        return
    
    if args.enable_cli_auth:
        enable_cli_auth_command(config, config_path)
        return
    
    if args.sync_from_cli:
        sync_from_cli_command(config, config_path)
        return
    
    # Handle list models command
    if args.list_models:
        await list_models_command(config, args.cache_bust)
        return
    
    # Fetch available models
    with console.status("[bold green]Fetching available models...[/bold green]"):
        models = await fetch_available_models(config, args.cache_bust)
    
    if not models:
        console.print("[bold red]Error:[/bold red] No models found.")
        console.print("Please check your API key and network connection.")
        sys.exit(1)
    
    # Select model
    selected_model = select_model_interactive(models, config)
    
    # Update configuration with last selected model
    config.last_selected_model = selected_model.id
    config.save_to_file(config_path)
    
    # Launch Claude CLI
    launch_claude_cli(selected_model, args.claude_args, args.dry_run)


async def run_main_async():
    """Async wrapper for main function to handle await context."""
    return await main()


def run_main_sync():
    """Synchronous wrapper with nest_asyncio support (Option 3)."""
    # Option 3: Use nest_asyncio for nested event loop support
    if has_nest_asyncio:
        nest_asyncio.apply()
        # Now asyncio.run() should work even in nested contexts
        return asyncio.run(main())
    else:
        # Fallback to Option 2 approach if nest_asyncio not available
        try:
            loop = asyncio.get_running_loop()
            # If we're in async context, create task and return it for awaiting
            task = loop.create_task(main())
            return task
        except RuntimeError:
            # No loop running, safe to use asyncio.run()
            return asyncio.run(main())


def run_main_safely():
    """Run main function with comprehensive event loop handling for all environments."""
    import threading
    import platform
    
    # Option 3: Try nest_asyncio first for nested event loop support
    if has_nest_asyncio:
        nest_asyncio.apply()
        # Now asyncio.run() should work even in nested contexts
        return asyncio.run(main())
    else:
        # nest_asyncio not available, use Option 2 approach
        try:
            loop = asyncio.get_running_loop()
            # Use create_task or ensure_future for existing loop
            task = loop.create_task(main())
            # For testing environments, we need to await this
            return task
        except RuntimeError:
            # No loop running, safe to use asyncio.run()
            try:
                return asyncio.run(main())
            except RuntimeError as e:
                if "cannot be called from a running event loop" in str(e):
                    # Fallback to threading approach
                    pass
                else:
                    raise
    
    # Always use threading in problematic environments
    force_threading = (
        os.environ.get("CLAUDE_LAUNCHER_FORCE_THREADING", "").lower() in ("1", "true", "yes") or
        platform.system() == "Linux" or  # All Linux environments (including WSL)
        "microsoft" in platform.uname().release.lower() or  # Explicit WSL check
        "WSL" in os.environ.get("WSL_DISTRO_NAME", "") or  # WSL2 environment variable
        "/mnt/c" in os.getcwd()  # WSL filesystem check
    )
    
    # Debug output for troubleshooting (can be removed later)
    if "--verbose" in sys.argv:
        console.print(f"[dim]Platform: {platform.system()}, Threading: {force_threading}[/dim]")
    
    if force_threading:
        # Use threading approach for maximum compatibility
        result = {"exception": None}
        
        def run_in_thread():
            try:
                # Completely isolated event loop
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    new_loop.run_until_complete(main())
                finally:
                    new_loop.close()
                    asyncio.set_event_loop(None)
            except Exception as e:
                result["exception"] = e
        
        thread = threading.Thread(target=run_in_thread, daemon=False)
        thread.start()
        thread.join()
        
        if result["exception"]:
            raise result["exception"]
        return
    
    # Standard approach for other environments
    try:
        # Try to detect running event loop
        try:
            current_loop = asyncio.get_running_loop()
            is_running = True
        except RuntimeError:
            is_running = False
        
        if is_running:
            # Try nest_asyncio first
            try:
                import nest_asyncio
                nest_asyncio.apply()
                asyncio.run(main())
                return
            except ImportError:
                pass
            
            # Fallback to threading
            result = {"exception": None}
            
            def run_in_thread():
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        new_loop.run_until_complete(main())
                    finally:
                        new_loop.close()
                        asyncio.set_event_loop(None)
                except Exception as e:
                    result["exception"] = e
            
            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()
            
            if result["exception"]:
                raise result["exception"]
        else:
            # No event loop detected, use standard approach
            asyncio.run(main())
            
    except Exception as e:
        # Last resort: always use threading if anything fails
        if "cannot be called from a running event loop" in str(e):
            result = {"exception": None}
            
            def run_in_thread():
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        new_loop.run_until_complete(main())
                    finally:
                        new_loop.close()
                        asyncio.set_event_loop(None)
                except Exception as e:
                    result["exception"] = e
            
            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()
            
            if result["exception"]:
                raise result["exception"]
        else:
            raise


if __name__ == "__main__":
    try:
        # Option 2: Use the sync wrapper that handles event loop detection
        result = run_main_sync()
        # If result is a task (from async context), we can't await it in __main__
        # The calling async context should handle awaiting
        if asyncio.iscoroutine(result) or hasattr(result, '__await__'):
            # This means we're being called from an async context
            # The caller should await the returned task
            pass
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Operation cancelled by user.[/bold yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        if "--verbose" in sys.argv:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)