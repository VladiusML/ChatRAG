from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os

from api.user_router import router as user_router
from api.vectorstore_router import router as vectorstore_router

# Создаем экземпляр FastAPI
app = FastAPI(
    title="Vector Store API",
    description="API для работы с векторными хранилищами текстовых данных",
    version="0.1.0"
)

app.include_router(user_router, prefix="/api/v1")
app.include_router(vectorstore_router, prefix="/api/v1")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В production установите конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Корневой маршрут для проверки работоспособности
@app.get("/", tags=["Health Check"])
async def health_check():
    return {"status": "ok"}

# Обработчик ошибок
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )
    # Для всех других исключений
    return JSONResponse(
        status_code=500,
        content={"detail": "Внутренняя ошибка сервера"},
    )

# Запуск сервера (если скрипт запущен напрямую)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)