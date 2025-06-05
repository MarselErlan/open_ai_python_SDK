from pydantic import BaseModel, HttpUrl
from typing import Optional, List
from datetime import datetime

# Company Schemas
class CompanyBase(BaseModel):
    name: str  # required
    industry: Optional[str] = None
    url: Optional[str] = None  # Changed from HttpUrl to str, made optional
    headcount: Optional[int] = None
    country: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None
    isPublic: Optional[bool] = False

class CompanyCreate(CompanyBase):
    pass

class CompanyUpdate(CompanyBase):
    name: Optional[str] = None  # optional

class Company(CompanyBase):
    id: int

    class Config:
        from_attributes = True

# JobPosting Schemas
class JobPostingBase(BaseModel):
    company_id: int  # required
    title: str  # required
    compensation_min: Optional[float] = None
    compensation_max: Optional[float] = None
    location_type: Optional[str] = None
    employment_type: Optional[str] = None
    description: Optional[str] = None

class JobPostingCreate(JobPostingBase):
    pass

class JobPostingUpdate(JobPostingBase):
    company_id: Optional[int] = None
    title: Optional[str] = None

class JobPosting(JobPostingBase):
    id: int
    company_id: int
    description: Optional[str] = None
    company: Optional[Company] = None

    class Config:
        from_attributes = True


class JobDescriptionRequest(BaseModel):
    required_tools: List[str]

class JobDescriptionResponse(BaseModel):
    job_id: int
    description: str
    generated_at: datetime