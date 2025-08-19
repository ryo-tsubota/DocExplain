"""
Document management API endpoints
"""
from typing import List
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from app.models.document import DocumentResponse, DocumentSummary, ProcessingStatus
from app.services.document_service import document_service

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload")
) -> DocumentResponse:
    """
    Upload a new document for processing
    
    Accepts PDF, DOCX, TXT, and MD files up to the configured size limit.
    The document will be processed asynchronously for question answering.
    
    Args:
        file: The document file to upload
        
    Returns:
        DocumentResponse with upload confirmation and document metadata
        
    Raises:
        HTTPException: If file validation fails or upload errors occur
    """
    try:
        return await document_service.upload_document(file)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("", response_model=List[DocumentSummary])
async def get_documents() -> List[DocumentSummary]:
    """
    Get list of all uploaded documents
    
    Returns:
        List of document summaries with metadata
    """
    return await document_service.get_documents()


@router.delete("/{document_id}")
async def delete_document(document_id: str) -> dict:
    """
    Delete a document and its associated data
    
    Args:
        document_id: The ID of the document to delete
        
    Returns:
        Confirmation message
        
    Raises:
        HTTPException: If document not found
    """
    success = await document_service.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"message": "Document deleted successfully"}


@router.get("/{document_id}/status")
async def get_document_status(document_id: str) -> dict:
    """
    Get the processing status of a document
    
    Args:
        document_id: The ID of the document
        
    Returns:
        Document processing status
        
    Raises:
        HTTPException: If document not found
    """
    status = await document_service.get_processing_status(document_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "document_id": document_id,
        "processing_status": status,
        "message": f"Document is {status.value}"
    }