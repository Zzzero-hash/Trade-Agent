"""
Authentication repository for database operations.
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID
import asyncpg
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.models.auth import (
    User, UserProfile, MFADevice, Session, AuthToken, APIKey,
    LoginAttempt, PasswordReset, EmailVerification
)
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class AuthReposit