# Import core Flask class and functions for rendering templates
from app.api.controllers import api_bp as api_module
from app.auth.controllers import auth_bp as auth_module
from flask import Flask, render_template
# Import SQLAlchemy for ORM (Object Relational Mapping)
from flask_sqlalchemy import SQLAlchemy

# Import CORS to allow cross-origin requests
from flask_cors import CORS

# Import FLask-Socktio for realtime communication
from flask_socketio import SocketIO, emit

# Initialize the Flask application
app = Flask(__name__)
# Load configuration from a separate config file (config.py)
app.config.from_object("config")

# socketio
# initialize SocketIO to support WebSocket communication
# Allow any origin to connect (CORS is set to "*")
socketio = SocketIO(app, cors_allowed_origins="*")

# enable CORS on both /api/* and /auth/*:
CORS(app, resources={
    r"/api/*":  {"origins": "*"},  # Allow all origins to access /api routes
    r"/auth/*": {"origins": "*"},  # Alllow all prigins to access / auth routes
})

# Initialize SQLAlchemy with the Flask app
db = SQLAlchemy(app)

# Define a custom error  handler for 404 Not found errors


@app.errorhandler(404)
def not_found(error):
    # Render a custom 404 error page when a route is not found
    return render_template("404.html"), 404


# Import blueprint modules for authentication and API routes

# register blueprints (For sign in, sign up, etc)
app.register_blueprint(auth_module)
# Register the API blueprint and prefix all it's routes with/ api
app.register_blueprint(api_module, url_prefix="/api")

# Create database tables defines in odels when the app context is ready
with app.app_context():
    db.create_all()
