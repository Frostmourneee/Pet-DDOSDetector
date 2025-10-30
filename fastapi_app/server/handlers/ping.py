from fastapi import APIRouter, status

api_router = APIRouter(tags=["Ping"])


@api_router.get("/ping", status_code=status.HTTP_200_OK)
async def ping():
    """
    Health check for API.
    Returns 'pong' if succeeded.
    """
    return {"message": "pong"}
