import secrets

from fastapi import HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ml_service.config import Settings

# API_TOKEN = os.environ["ML_API_TOKEN"]
API_TOKEN = Settings().ML_API_TOKEN

bearer_scheme = HTTPBearer(auto_error=False)


def get_vector_store(request: Request):
    return request.app.state.vector_store


def get_embedder(request: Request):
    return request.app.state.embedder


def get_llm_model(request: Request):
    return request.app.state.llm


def require_bearer(
    creds: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
):
    if creds is None:
        # No Authorization header provided
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token"
        )
    token = creds.credentials  # <-- Swagger gives you JUST the token here
    if not secrets.compare_digest(token, API_TOKEN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token"
        )
