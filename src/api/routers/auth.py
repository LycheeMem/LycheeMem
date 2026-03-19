"""用户认证端点：注册与登录。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from src.api.models import LoginRequest, LoginResponse, RegisterRequest, RegisterResponse
from src.auth.auth import create_access_token
from src.auth.user_store import UserStore

router = APIRouter(prefix="/auth", tags=["auth"])


def _get_user_store(request: Request) -> UserStore:
    store = getattr(request.app.state, "user_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="UserStore not initialized")
    return store


@router.post("/register", response_model=RegisterResponse)
async def register(req: RegisterRequest, request: Request):
    """用户注册。"""
    user_store = _get_user_store(request)
    try:
        user = user_store.create_user(
            username=req.username,
            password=req.password,
            display_name=req.display_name or req.username,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    token = create_access_token(user.user_id, user.username)
    return RegisterResponse(
        user_id=user.user_id,
        username=user.username,
        display_name=user.display_name,
        token=token,
    )


@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest, request: Request):
    """用户登录。"""
    user_store = _get_user_store(request)
    user = user_store.authenticate(req.username, req.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )
    token = create_access_token(user.user_id, user.username)
    return LoginResponse(
        user_id=user.user_id,
        username=user.username,
        display_name=user.display_name,
        token=token,
    )
