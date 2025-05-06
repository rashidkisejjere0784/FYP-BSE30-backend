# Import flask, template operators, and CORS
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

# Define the WSGI application object
app = Flask(__name__)

# Load configurations
app.config.from_object('config')

# Enable CORS only on /api/* endpoints
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Or, to enable CORS globally on all routes:
# CORS(app)

# Define the database object
db = SQLAlchemy(app)

# Sample HTTP error handling
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

# Import and register blueprints
from app.auth.controllers import auth_bp as auth_module
from app.api.controllers  import api_bp  as api_module

app.register_blueprint(auth_module)
app.register_blueprint(api_module, url_prefix='/api')

# Build the database
with app.app_context():
    db.create_all()
