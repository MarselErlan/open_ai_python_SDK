from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

# Define a single Base instance
Base = declarative_base()

class Company(Base):
    __tablename__ = "Company"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    industry = Column(String)
    headcount = Column(Integer)
    country = Column(String)
    state = Column(String)
    city = Column(String)
    isPublic = Column(Boolean, default=False)
    url = Column(String, nullable=True)  # Ensure this matches the database

    job_postings = relationship("JobPosting", back_populates="company")

class JobPosting(Base):
    __tablename__ = "JobPosting"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    company_id = Column(Integer, ForeignKey("Company.id"), nullable=False)
    compensation_min = Column(Float)
    compensation_max = Column(Float)
    location_type = Column(String)
    employment_type = Column(String)
    description = Column(String, nullable=True)  # Already added as per task

    # Add relationship to Company
    company = relationship("Company", back_populates="job_postings")