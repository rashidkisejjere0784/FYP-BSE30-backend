# Import flask dependencies for handling routing and HTTP requests
from flask import Blueprint, request

# Import utilities for password hashing and verification
from werkzeug.security import check_password_hash, generate_password_hash

# Import the database object from the main app module
from app import db

# Import the User model from the auth module
from app.auth.models import User

# Define the blueprint: 'auth', set its url prefix: app.url/auth
# 'auth_bp' will handle all routes starting with '/auth'
auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

# Set the route and accepted methods
# Route for user sign-in (login)


@auth_bp.route('/signin/', methods=['GET', 'POST'], strict_slashes=False)
def signin():
    # Only allow POST requests for signing in
    if request.method == 'POST':

        # Parse JSON data from the request body
        post_data = request.get_json()
        username = post_data.get('username')
        password = post_data.get('password')

        # Query the database for a user with the provided username
        # Search for a user in the database with the given username
        user = User.query.filter_by(name=username).first()

        # Check if the user exists and if the password is correct
        if user and check_password_hash(user.password, password):
            # Return a success message with the user's information
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

    # If the method is not POST, return method not allowed
    return {'message': 'Method not allowed'}, 405


# Route for user sign-up (registration)
@auth_bp.route('/signup/', methods=['POST'], strict_slashes=False)
def signup():
    # Only allow POST requests for sign-up
    if request.method == 'POST':
        # Get the post data
        post_data = request.get_json()
        username = post_data.get('username')
        password = post_data.get('password')
        location = post_data.get('location')
        email = post_data.get('email')
        check_password = post_data.get('check_password')

        # Validate the input data and Check if the passwords match
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

        # Add the new user to the database
        db.session.add(new_user)
        db.session.commit()

        # Return a success message with the new user's informatio
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

    # If the method is not POST, return method not allowed
    return {'message': 'Method not allowed'}, 405
