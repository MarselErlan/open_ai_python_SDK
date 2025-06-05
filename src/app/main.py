from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.endpoints import companies, jobs
# from db.session import engine, Base  # Removed since not needed
# from models import models  # Removed since not used directly
# Base.metadata.create_all(bind=engine)  # Removed

app = FastAPI(title="Job Board API", description="A modern job board API with AI-powered job descriptions")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(companies.router, prefix="/companies", tags=["companies"])
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])


