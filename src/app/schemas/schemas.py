from pydantic import BaseModel, HttpUrl
from typing import Optional, List

# Company Schemas
class CompanyBase(BaseModel):
    name: str #required
    industry: Optional[str] = None
    url: Optional[HttpUrl] = None
    headcount: Optional[int] = None
    country: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None
    isPublic: Optional[bool] = False

class CompanyCreate(CompanyBase):
    pass

class CompanyUpdate(CompanyBase):
    name: Optional[str] = None #optional

class Company(CompanyBase):
    id: int

    class Config:
        from_attributes = True

# JobPosting Schemas
class JobPostingBase(BaseModel):
    company_id: int
    title: str
    compensation_min: Optional[float] = None
    compensation_max: Optional[float] = None
    location_type: Optional[str] = None
    employment_type: Optional[str] = None

class JobPostingCreate(JobPostingBase):
    pass

class JobPostingUpdate(JobPostingBase):
    company_id: Optional[int] = None
    title: Optional[str] = None

class JobPosting(JobPostingBase):
    id: int

    class Config:
        from_attributes = True 