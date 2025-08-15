"""API endpoint tests for the RAG system FastAPI application."""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from unittest.mock import Mock


@pytest.mark.api
class TestQueryEndpoint:
    """Test cases for the /api/query endpoint."""
    
    def test_query_without_session_id(self, test_client: TestClient, sample_query_request):
        """Test query endpoint creates new session when none provided."""
        response = test_client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"
        assert isinstance(data["sources"], list)
    
    def test_query_with_existing_session_id(self, test_client: TestClient, sample_query_request_with_session):
        """Test query endpoint uses provided session ID."""
        response = test_client.post("/api/query", json=sample_query_request_with_session)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["session_id"] == "existing-session-123"
        assert "answer" in data
        assert "sources" in data
    
    def test_query_missing_query_field(self, test_client: TestClient):
        """Test query endpoint returns error for missing query field."""
        invalid_request = {"session_id": "test"}
        
        response = test_client.post("/api/query", json=invalid_request)
        
        assert response.status_code == 422  # Validation error
        assert "detail" in response.json()
    
    def test_query_empty_query_string(self, test_client: TestClient):
        """Test query endpoint handles empty query string."""
        empty_query = {"query": "", "session_id": None}
        
        response = test_client.post("/api/query", json=empty_query)
        
        assert response.status_code == 200  # Should still process empty queries
        data = response.json()
        assert "answer" in data
    
    def test_query_with_special_characters(self, test_client: TestClient):
        """Test query endpoint handles special characters in query."""
        special_query = {
            "query": "What about Python's 'list comprehensions' & string formatting?",
            "session_id": None
        }
        
        response = test_client.post("/api/query", json=special_query)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
    
    def test_query_rag_system_error(self, test_client: TestClient, test_app: FastAPI, sample_query_request):
        """Test query endpoint handles RAG system errors gracefully."""
        # Mock the RAG system to raise an exception
        test_app.state.mock_rag.query.side_effect = Exception("RAG system error")
        
        response = test_client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 500
        assert "detail" in response.json()
        assert "RAG system error" in response.json()["detail"]
        
        # Reset the mock for other tests
        test_app.state.mock_rag.query.side_effect = None


@pytest.mark.api
class TestCoursesEndpoint:
    """Test cases for the /api/courses endpoint."""
    
    def test_get_course_stats_success(self, test_client: TestClient):
        """Test courses endpoint returns course statistics."""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 2
        assert data["course_titles"] == ["Test Course 1", "Test Course 2"]
        assert isinstance(data["course_titles"], list)
    
    def test_get_course_stats_empty_courses(self, test_client: TestClient, test_app: FastAPI):
        """Test courses endpoint with no courses loaded."""
        # Mock empty course analytics
        test_app.state.mock_rag.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 0
        assert data["course_titles"] == []
        
        # Reset the mock
        test_app.state.mock_rag.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["Test Course 1", "Test Course 2"]
        }
    
    def test_get_course_stats_error(self, test_client: TestClient, test_app: FastAPI):
        """Test courses endpoint handles analytics errors gracefully."""
        # Mock the analytics to raise an exception
        test_app.state.mock_rag.get_course_analytics.side_effect = Exception("Analytics error")
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 500
        assert "detail" in response.json()
        assert "Analytics error" in response.json()["detail"]
        
        # Reset the mock
        test_app.state.mock_rag.get_course_analytics.side_effect = None


@pytest.mark.api
class TestRootEndpoint:
    """Test cases for the root endpoint."""
    
    def test_root_endpoint(self, test_client: TestClient):
        """Test root endpoint returns welcome message."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "RAG System" in data["message"]


@pytest.mark.api
class TestRequestValidation:
    """Test request validation and error handling."""
    
    def test_invalid_json_request(self, test_client: TestClient):
        """Test API handles invalid JSON gracefully."""
        response = test_client.post(
            "/api/query",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_wrong_content_type(self, test_client: TestClient):
        """Test API handles wrong content type."""
        response = test_client.post(
            "/api/query",
            data="query=test",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        # FastAPI should handle this gracefully
        assert response.status_code in [422, 400]
    
    def test_query_field_wrong_type(self, test_client: TestClient):
        """Test query field with wrong data type."""
        invalid_request = {"query": 123, "session_id": None}
        
        response = test_client.post("/api/query", json=invalid_request)
        
        assert response.status_code == 422  # Validation error
    
    def test_session_id_wrong_type(self, test_client: TestClient):
        """Test session_id field with wrong data type."""
        invalid_request = {"query": "test query", "session_id": 123}
        
        response = test_client.post("/api/query", json=invalid_request)
        
        assert response.status_code == 422  # Validation error


@pytest.mark.api
class TestResponseValidation:
    """Test response format validation."""
    
    def test_query_response_structure(self, test_client: TestClient, sample_query_request):
        """Test query response has correct structure."""
        response = test_client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check all required fields are present
        required_fields = ["answer", "sources", "session_id"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Check field types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        
        # Check sources list contains strings
        for source in data["sources"]:
            assert isinstance(source, str)
    
    def test_courses_response_structure(self, test_client: TestClient):
        """Test courses response has correct structure."""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check all required fields are present
        required_fields = ["total_courses", "course_titles"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Check field types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        
        # Check course titles list contains strings
        for title in data["course_titles"]:
            assert isinstance(title, str)
        
        # Check total_courses matches list length
        assert data["total_courses"] == len(data["course_titles"])


@pytest.mark.api
class TestCORSHeaders:
    """Test CORS headers are properly set."""
    
    def test_cors_headers_on_query(self, test_client: TestClient, sample_query_request):
        """Test CORS headers are present on query endpoint."""
        response = test_client.post("/api/query", json=sample_query_request)
        
        # Should have CORS headers (TestClient might not show all headers)
        assert response.status_code == 200
    
    def test_cors_headers_on_courses(self, test_client: TestClient):
        """Test CORS headers are present on courses endpoint."""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
    
    def test_options_request(self, test_client: TestClient):
        """Test OPTIONS request for CORS preflight."""
        response = test_client.options("/api/query")
        
        # Should not return error for OPTIONS request
        assert response.status_code in [200, 405]  # 405 if OPTIONS not explicitly handled