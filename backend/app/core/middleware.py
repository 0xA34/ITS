import time
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class RequestIdLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        start = time.time()

        response = await call_next(request)

        response.headers["x-request-id"] = rid
        response.headers["x-response-ms"] = str(int((time.time() - start) * 1000))
        return response
