# Import flask dependencies
from flask import Blueprint, request

# Import password / encryption helper tools
from werkzeug.security import check_password_hash, generate_password_hash

# Import the database object from the main app module
from app import db

# Import module models (i.e. User)
from app.auth.models import User

# Define the blueprint: 'auth', set its url prefix: app.url/auth
auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

# Set the route and accepted methods
@auth_bp.route('/signin/', methods=['GET', 'POST'], strict_slashes=False)
def signin():
    if request.method == 'POST':
        # Get the post data
        post_data = request.get_json()
        username = post_data.get('username')
        password = post_data.get('password')

        # Query the database for a user with the provided username
        user = User.query.filter_by(name=username).first()

        # Check if the user exists and if the password is correct
        if user and check_password_hash(user.password, password):
            return {
                'message': 'Login successful',
                'user': {
                    'username': user.name,
                    'email': user.email,
                    'location': user.location,
                    'id': user.id,
                    'date_created': user.date_created,
                    'date_modified': user.date_modified
                }
                }, 200
        else:
            return {'message': 'Invalid username or password'}, 401

    return {'message': 'Method not allowed'}, 405


@auth_bp.route('/signup/', methods=['POST'], strict_slashes=False)
def signup():
    if request.method == 'POST':
        # Get the post data
        post_data = request.get_json()
        username = post_data.get('username')
        password = post_data.get('password')
        location = post_data.get('location')
        email = post_data.get('email')
        check_password = post_data.get('check_password')
        
        # Validate the input data
        if check_password != password:
            return {'message': 'Invalid User or Password'}, 400

        # Check if the user already exists
        existing_user = User.query.filter_by(name=username).first()
        if existing_user:
            return {
                'message': 'User already exists'
                }, 409

        # Create a new user
        new_user = User(name=username,
                        password=generate_password_hash(password),
                        location=location, email=email,)
        
        db.session.add(new_user)
        db.session.commit()

        return {
            'message': 'User created successfully',
            'user': {
                'username': new_user.name,
                'email': new_user.email,
                'location': new_user.location,
                'id': new_user.id,
                'date_created': new_user.date_created,
                'date_modified': new_user.date_modified
            }
            }, 201

    return {'message': 'Method not allowed'}, 405