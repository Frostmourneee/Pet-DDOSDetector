from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

api_router: APIRouter = APIRouter(tags=["Ping"])


@api_router.get("/ping", status_code=status.HTTP_200_OK)
async def ping() -> JSONResponse:
    """
    Health check for API.
    Returns 'pong' if succeeded.
    """
    return {"message": "pong"}
