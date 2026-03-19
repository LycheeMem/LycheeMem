"""
JWT 认证工具与 FastAPI 依赖。

使用 PyJWT (HS256) 签发和验证 token。
"""

from __future__ import annotations

import datetime
from typing import Any

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.auth.user_store import User, UserStore

_bearer_scheme = HTTPBearer(auto_error=False)

# 默认值仅用于开发，生产环境必须通过 config 覆盖
_DEFAULT_SECRET = "lychee-dev-secret-change-me"
_DEFAULT_EXPIRE_HOURS = 168  # 7 天

# 运行时由 server 初始化赋值
_jwt_secret: str = _DEFAULT_SECRET
_jwt_expire_hours: int = _DEFAULT_EXPIRE_HOURS


def configure_jwt(secret: str, expire_hours: int = _DEFAULT_EXPIRE_HOURS) -> None:
    """由 server.py 启动时调用，注入实际的 secret。"""
    global _jwt_secret, _jwt_expire_hours
    _jwt_secret = secret
    _jwt_expire_hours = expire_hours


def create_access_token(user_id: str, username: str) -> str:
    """签发 JWT access token。"""
    now = datetime.datetime.now(datetime.timezone.utc)
    payload: dict[str, Any] = {
        "sub": user_id,
        "username": username,
        "iat": now,
        "exp": now + datetime.timedelta(hours=_jwt_expire_hours),
    }
    return jwt.encode(payload, _jwt_secret, algorithm="HS256")


def decode_access_token(token: str) -> dict[str, Any]:
    """解码并验证 JWT，返回 payload。失败抛出 HTTPException 401。"""
    try:
        return jwt.decode(token, _jwt_secret, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 已过期",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的 Token",
        )


def _get_user_store(request: Request) -> UserStore:
    store = getattr(request.app.state, "user_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="UserStore not initialized")
    return store


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> User:
    """FastAPI 依赖：从 Bearer token 解析当前用户。

    未提供 token 或 token 无效时返回 401。
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="需要登录",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_access_token(credentials.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="无效的 Token")

    user_store = _get_user_store(request)
    user = user_store.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户不存在")
    return user


async def get_optional_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> User | None:
    """FastAPI 依赖：可选认证。未提供 token 时返回 None（兼容无认证模式）。"""
    if credentials is None:
        return None
    try:
        return await get_current_user(request, credentials)
    except HTTPException:
        return None
