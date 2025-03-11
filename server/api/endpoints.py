from fastapi import APIRouter

router = APIRouter()


@router.get("/latency")
async def check_latency():
    return {"message": "pong"}
