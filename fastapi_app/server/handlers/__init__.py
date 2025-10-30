from fastapi_app.server.handlers.ping import api_router as ping
from fastapi_app.server.handlers.predict import api_router as predict

list_of_routes = [
    ping,
    predict,
]
