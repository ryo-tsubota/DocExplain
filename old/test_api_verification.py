#!/usr/bin/env python3
"""
Quick verification script for document management API endpoints
"""
import requests
import json
import io
import time

# Test the API endpoints
def test_api_endpoints():
    base_url = "http://127.0.0.1:8000"
    
    print("Testing Document Management API Endpoints...")
    
    # Test 1: GET /api/documents (empty)
    print("\n1. Testing GET /api/documents (empty)...")
    try:
        response = requests.get(f"{base_url}/api/documents")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
        assert response.json() == []
        print("✓ PASS")
    except Exception as e:
        print(f"✗ FAIL: {e}")
    
    # Test 2: Upload a document
    print("\n2. Testing POST /api/documents/upload...")
    try:
        files = {
            'file': ('test.txt', io.BytesIO(b'This is a test document for API verification.'), 'text/plain')
        }
        response = requests.post(f"{base_url}/api/documents/upload", files=files)
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        assert response.status_code == 200
        document_id = data['document']['id']
        print(f"✓ PASS - Document ID: {document_id}")
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return
    
    # Test 3: GET /api/documents (with data)
    print("\n3. Testing GET /api/documents (with data)...")
    try:
        response = requests.get(f"{base_url}/api/documents")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        assert response.status_code == 200
        assert len(data) == 1
        assert data[0]['id'] == document_id
        print("✓ PASS")
    except Exception as e:
        print(f"✗ FAIL: {e}")
    
    # Test 4: GET /api/documents/{document_id}/status
    print(f"\n4. Testing GET /api/documents/{document_id}/status...")
    try:
        response = requests.get(f"{base_url}/api/documents/{document_id}/status")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        assert response.status_code == 200
        assert data['document_id'] == document_id
        assert 'processing_status' in data
        print("✓ PASS")
    except Exception as e:
        print(f"✗ FAIL: {e}")
    
    # Test 5: DELETE /api/documents/{document_id}
    print(f"\n5. Testing DELETE /api/documents/{document_id}...")
    try:
        response = requests.delete(f"{base_url}/api/documents/{document_id}")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        assert response.status_code == 200
        assert data['message'] == 'Document deleted successfully'
        print("✓ PASS")
    except Exception as e:
        print(f"✗ FAIL: {e}")
    
    # Test 6: GET /api/documents (empty again)
    print("\n6. Testing GET /api/documents (empty after deletion)...")
    try:
        response = requests.get(f"{base_url}/api/documents")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
        assert response.json() == []
        print("✓ PASS")
    except Exception as e:
        print(f"✗ FAIL: {e}")
    
    # Test 7: Error cases
    print("\n7. Testing error cases...")
    
    # Test DELETE non-existent document
    try:
        response = requests.delete(f"{base_url}/api/documents/nonexistent-id")
        print(f"DELETE non-existent - Status: {response.status_code}")
        assert response.status_code == 404
        print("✓ PASS - DELETE non-existent returns 404")
    except Exception as e:
        print(f"✗ FAIL: {e}")
    
    # Test GET status for non-existent document
    try:
        response = requests.get(f"{base_url}/api/documents/nonexistent-id/status")
        print(f"GET status non-existent - Status: {response.status_code}")
        assert response.status_code == 404
        print("✓ PASS - GET status non-existent returns 404")
    except Exception as e:
        print(f"✗ FAIL: {e}")
    
    print("\n🎉 All API endpoint tests completed!")

if __name__ == "__main__":
    print("Note: This script requires the FastAPI server to be running on http://127.0.0.1:8000")
    print("Start the server with: uvicorn app.main:app --reload")
    print("Then run this script to verify the API endpoints.")
    
    # For now, just show what the script would test
    print("\nThis script would test the following endpoints:")
    print("- GET /api/documents")
    print("- POST /api/documents/upload")
    print("- DELETE /api/documents/{document_id}")
    print("- GET /api/documents/{document_id}/status")
    print("\nAll these endpoints are already implemented and tested via pytest.")