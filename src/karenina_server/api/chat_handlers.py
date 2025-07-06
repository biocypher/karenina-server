"""Chat and session API handlers."""

from typing import Any

from fastapi import HTTPException

try:
    from karenina.llm import ChatRequest, ChatResponse, call_model, delete_session, get_session, list_sessions
    from karenina.llm.interface import LANGCHAIN_AVAILABLE, LLMNotAvailableError, SessionError, chat_sessions

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LANGCHAIN_AVAILABLE = False


def register_chat_routes(app: Any) -> None:
    """Register chat-related routes."""

    @app.post("/api/chat", response_model=ChatResponse)  # type: ignore[misc]
    async def chat_endpoint(request: ChatRequest) -> ChatResponse:
        """Chat with a language model."""
        try:
            return call_model(
                model=request.model,
                provider=request.provider,
                message=request.message,
                session_id=request.session_id,
                system_message=request.system_message,
                temperature=request.temperature,
            )
        except Exception as e:
            # Convert LLM exceptions to HTTP exceptions
            if isinstance(e, LLMNotAvailableError):
                raise HTTPException(status_code=500, detail=str(e)) from e
            elif isinstance(e, SessionError):
                raise HTTPException(status_code=400, detail=str(e)) from e
            else:
                raise HTTPException(status_code=500, detail=f"Error calling model: {e!s}") from e

    @app.get("/api/sessions")  # type: ignore[misc]
    async def list_sessions_endpoint() -> dict[str, Any]:
        """List all active chat sessions."""
        return {"sessions": list_sessions()}

    @app.get("/api/sessions/{session_id}")  # type: ignore[misc]
    async def get_session_endpoint(session_id: str) -> dict[str, Any]:
        """Get details of a specific chat session."""
        session = get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "session_id": session.session_id,
            "model": session.model,
            "provider": session.provider,
            "created_at": session.created_at.isoformat(),
            "last_used": session.last_used.isoformat(),
            "messages": [
                {
                    "type": "system"
                    if msg.__class__.__name__ == "SystemMessage"
                    else ("human" if msg.__class__.__name__ == "HumanMessage" else "ai"),
                    "content": msg.content,
                }
                for msg in session.messages
            ],
        }

    @app.delete("/api/sessions/{session_id}")  # type: ignore[misc]
    async def delete_session_endpoint(session_id: str) -> dict[str, str]:
        """Delete a chat session."""
        if not delete_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")

        return {"message": f"Session {session_id} deleted successfully"}

    @app.get("/api/health")  # type: ignore[misc]
    async def health_check() -> dict[str, Any]:
        """Health check endpoint."""
        return {"status": "healthy", "langchain_available": LANGCHAIN_AVAILABLE, "active_sessions": len(chat_sessions)}

    @app.get("/api/timestamp")  # type: ignore[misc]
    async def get_server_timestamp() -> dict[str, str]:
        """Get current server timestamp in ISO format."""
        from datetime import datetime

        return {"timestamp": datetime.now().isoformat()}
