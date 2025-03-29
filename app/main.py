import os

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.user_router import router as user_router
from app.api.vectorstore_router import router as vectorstore_router

app = FastAPI(
    title="Vector Store API",
    description="API для работы с векторными хранилищами текстовых данных",
    version="0.1.0",
)

app.include_router(user_router, prefix="/api/v1")
app.include_router(vectorstore_router, prefix="/api/v1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health Check"])
async def health_check():
    return {"status": "ok"}


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )
    return JSONResponse(
        status_code=500,
        content={"detail": "Внутренняя ошибка сервера"},
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
