from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
# from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config.from_object("config")

# socketio
# socketio = SocketIO(app, cors_allowed_origins="*")

# enable CORS on both /api/* and /auth/*:
CORS(app, resources={
    r"/api/*":  {"origins": "*"},
    r"/auth/*": {"origins": "*"},
})

db = SQLAlchemy(app)

@app.errorhandler(404)
def not_found(error):
    return render_template("404.html"), 404

from app.auth.controllers import auth_bp as auth_module
from app.api.controllers  import api_bp  as api_module

# register blueprints
app.register_blueprint(auth_module)
app.register_blueprint(api_module, url_prefix="/api")

with app.app_context():
    db.create_all()
