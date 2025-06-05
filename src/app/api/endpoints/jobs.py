from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
import json
from db.session import get_db
from models.models import JobPosting as JobPostingModel, Company
from schemas.schemas import JobPosting as JobPostingSchema, JobPostingCreate, JobPostingUpdate, JobDescriptionRequest, JobDescriptionResponse
from openai import OpenAI
from core.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize OpenAI client
openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

def get_db_session():
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()

# Job CRUD endpoints
@router.post("/", response_model=JobPostingSchema)
def create_job_posting(job: JobPostingCreate, db: Session = Depends(get_db_session)):
    # Verify company exists
    company = db.query(Company).filter(Company.id == job.company_id).first()
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    
    db_job = JobPostingModel(**job.model_dump())
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    return db_job

@router.get("/", response_model=List[JobPostingSchema])
def read_job_postings(skip: int = 0, limit: int = 100, db: Session = Depends(get_db_session)):
    jobs = db.query(JobPostingModel).offset(skip).limit(limit).all()
    return jobs

@router.get("/{job_id}", response_model=JobPostingSchema)
def read_job_posting(job_id: int, db: Session = Depends(get_db_session)):
    db_job = db.query(JobPostingModel).filter(JobPostingModel.id == job_id).first()
    if db_job is None:
        raise HTTPException(status_code=404, detail="Job posting not found")
    return db_job

@router.put("/{job_id}", response_model=JobPostingSchema)
def update_job_posting(job_id: int, job: JobPostingUpdate, db: Session = Depends(get_db_session)):
    db_job = db.query(JobPostingModel).filter(JobPostingModel.id == job_id).first()
    if db_job is None:
        raise HTTPException(status_code=404, detail="Job posting not found")
    
    # If company_id is being updated, verify the new company exists
    if job.company_id is not None:
        company = db.query(Company).filter(Company.id == job.company_id).first()
        if not company:
            raise HTTPException(status_code=404, detail="Company not found")
    
    update_data = job.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_job, field, value)
    
    db.commit()
    db.refresh(db_job)
    return db_job

@router.delete("/{job_id}")
def delete_job_posting(job_id: int, db: Session = Depends(get_db_session)):
    db_job = db.query(JobPostingModel).filter(JobPostingModel.id == job_id).first()
    if db_job is None:
        raise HTTPException(status_code=404, detail="Job posting not found")
    
    db.delete(db_job)
    db.commit()
    return {"message": "Job posting deleted successfully"}

# Job Description Generation
@router.post("/{id}/description", response_model=JobDescriptionResponse)
async def generate_job_description(id: int, request: JobDescriptionRequest, db: Session = Depends(get_db_session)):
    # Retrieve the job posting and associated company
    job = db.query(JobPostingModel).filter(JobPostingModel.id == id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job posting not found")

    if not job.company:
        raise HTTPException(status_code=404, detail="Associated company not found")

    # Prepare the prompt for OpenAI
    tools = ", ".join(request.required_tools)
    prompt = (
        f"Generate a professional job description for the position '{job.title}' at {job.company.name}, "
        f"a company in the {job.company.industry} industry"
        f"The role requires expertise in the following tools: {tools}. "
        f"Structure the response as a single paragraph."
    )

    # Define the function for OpenAI function calling
    functions = [
        {
            "name": "generate_job_description",
            "description": "Generate a job description based on provided details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "The generated job description."
                    }
                },
                "required": ["description"]
            }
        }
    ]

    try:
        # Use OpenAI chat completion with streaming
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional HR assistant specializing in writing job descriptions."},
                {"role": "user", "content": prompt}
            ],
            functions=functions,
            function_call={"name": "generate_job_description"},
            stream=True
        )

        # Collect the streamed response
        full_description = ""
        for chunk in response:
            if chunk.choices[0].delta.function_call:
                if chunk.choices[0].delta.function_call.arguments:
                    # Parse the arguments (comes as JSON chunks)
                    args_chunk = chunk.choices[0].delta.function_call.arguments
                    full_description += args_chunk
                    

        # Parse the accumulated description (it's a JSON string)
        description_data = json.loads(full_description)
        description = description_data.get("description", "")

        # Save the description to the database
        job.description = description
        db.commit()
        db.refresh(job)

        # Prepare the response
        generated_at = datetime.utcnow()
        return JobDescriptionResponse(
            job_id=job.id,
            description=description,
            generated_at=generated_at
        )

    except Exception as e:
        logger.error(f"Error generating job description: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate job description: {str(e)}")