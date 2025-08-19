"""
Question-Answer API endpoints
"""
from fastapi import APIRouter

router = APIRouter(prefix="/api/qa", tags=["qa"])