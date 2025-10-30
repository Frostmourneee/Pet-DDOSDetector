from server.handlers.ping import api_router as ping
from server.handlers.predict import api_router as predict


list_of_routes = [
    ping,
    predict,
]