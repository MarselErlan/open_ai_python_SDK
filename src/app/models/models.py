from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey
from sqlalchemy.orm import relationship

from app.db.session import Base

class Company(Base):
    __tablename__ = "Company"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    industry = Column(String)
    url = Column(String)
    headcount = Column(Integer)
    country = Column(String)
    state = Column(String)
    city = Column(String)
    isPublic = Column(Boolean, default=False)

    job_postings = relationship("JobPosting", back_populates="company")

class JobPosting(Base):
    __tablename__ = "JobPosting"

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("Company.id"))
    title = Column(String, nullable=False)
    compensation_min = Column(Float)
    compensation_max = Column(Float)
    location_type = Column(String)
    employment_type = Column(String)

    company = relationship("Company", back_populates="job_postings") 