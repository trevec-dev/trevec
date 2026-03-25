"""API endpoint handlers for the web service."""

from typing import Optional


class Request:
    """Represents an incoming HTTP request."""

    def __init__(self, method: str, path: str, body: Optional[str] = None):
        self.method = method
        self.path = path
        self.body = body
        self.headers = {}

    def get_header(self, name: str) -> Optional[str]:
        """Get a request header by name."""
        return self.headers.get(name)


class Response:
    """Represents an HTTP response."""

    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body

    @staticmethod
    def ok(body: str) -> "Response":
        """Create a 200 OK response."""
        return Response(200, body)

    @staticmethod
    def not_found() -> "Response":
        """Create a 404 Not Found response."""
        return Response(404, "Not Found")


def handle_login(request: Request) -> Response:
    """Handle a login request and return a session token."""
    if request.method != "POST":
        return Response(405, "Method Not Allowed")
    return Response.ok("session_token_123")


def handle_get_user(request: Request, user_id: int) -> Response:
    """Get user details by ID."""
    if request.method != "GET":
        return Response(405, "Method Not Allowed")
    return Response.ok(f'{{"id": {user_id}, "name": "test"}}')


def route(request: Request) -> Response:
    """Route an incoming request to the appropriate handler."""
    if request.path == "/login":
        return handle_login(request)
    elif request.path.startswith("/users/"):
        user_id = int(request.path.split("/")[-1])
        return handle_get_user(request, user_id)
    return Response.not_found()
