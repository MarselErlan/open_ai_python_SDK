from fastapi import FastAPI, Query, Path, Depends
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy import create_engine, Column, Integer, String, text
from sqlalchemy.orm import sessionmaker, Session


# SQLAlchemy setup 
DATABASE_URL = "postgresql://postgres:DBlZpVocCkftaAnY@db.jnstlqijxqgmcepwnemt.supabase.co:5432/postgres"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)



app = FastAPI()

# This is our data model - what an application looks like
class Candidate(BaseModel):
    candidate_id: str 
    name: str 
    email: str 
    job_id: str | None = None

# This is our "database" - just a list in memory - cache memory
applications: List[Candidate] = []

#creating a db connection session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()    
    

@app.get("/jobs")
def get_all_job_postings(db: Session = Depends(get_db)):
    result = db.execute(text('SELECT * FROM "JobPosting"'))
    
    rows = result.fetchall()

    #format each row as a String
    output = []
    for row in rows:
        output.append(str(dict(row._mapping)))

    return output    
    


    
    






@app.post("/applications")
def postApplications(candidate: Candidate):
    #input sanitization --> if the email fits the format or no? 
    #name is at least 2 words
    #does this jobId already in the cache? if yes, update it. if no insert
    applications.append(candidate)
    return {
        "status": "success",
        "message": f"Application submitted for {candidate.name}"
    }

@app.get("/applications")
def getApplication(
    company_name: str = Query(None, description="optional query param for company name"),
    candidate_email: str = Query(None, description="optional query param for candidate email")
):
    if company_name:
        return {
            "status": "success",
            "message": f"Here is your application for {company_name}"
        }
    elif candidate_email:
        return {
            "status": "success",
            "message": f"Here is your application for {candidate_email}"
        }
    else:
        return {
            "status": "success",
            "message": "Here are all of your applications"
        }

@app.get("/applications/{candidate_id}")
def getApplicationById(candidate_id: str):
    for app in applications:
        if app.candidate_id == candidate_id:
            return {
                "status": "success",
                "message": f"Application found for candidate ID: {candidate_id}"
            }
    return {
        "status": "success",
        "message": "Application not found"
    }

@app.put("/applications/{candidate_id}")
def putApplications(
    candidate_id: str = Path(..., description="The ID of the candidate to update"),
    email: str = Query(None, description="New email address"),
    job_id: str = Query(None, description="New job ID")
):
    for app in applications:
        if app.candidate_id == candidate_id:
            if email:
                app.email = email
                return {
                    "status": "success",
                    "message": f"Email updated to {email}"
                }
            if job_id:
                app.job_id = job_id
                return {
                    "status": "success",
                    "message": f"Job ID updated to {job_id}"
                }
    return {
        "status": "success",
        "message": "Application not found"
    }

@app.delete("/applications/{candidate_id}")
def deleteApplication(candidate_id: str):
    for i, app in enumerate(applications):
        if app.candidate_id == candidate_id:
            applications.pop(i)
            return {
                "status": "success",
                "message": f"Application for {candidate_id} has been deleted"
            }
    return {
        "status": "success",
        "message": "Application not found"
    }






