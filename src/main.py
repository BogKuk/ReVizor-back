from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.authorization.auth_router import router as auth_router
from src.upload.upload_router import router as upload_router
from src.analysis.analysis_router import router as analysis_router

app = FastAPI()
app.include_router(auth_router)
app.include_router(upload_router)
app.include_router(analysis_router)

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Authorization", "X-Refresh-Token"]
)
