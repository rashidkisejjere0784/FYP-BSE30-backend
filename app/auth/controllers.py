# Import flask dependencies and modules
from flask import Blueprint, request, render_template, \
    flash, g, session, redirect, url_for

# Import password / encryption helper tools used for secure authentication
from werkzeug.security import check_password_hash, generate_password_hash

# Import the database object from the main app module
from app import db

# Import module forms used to collect user credentials
from app.auth.forms import LoginForm

# Import module models (i.e. User) to query users from the database
from app.auth.models import User

# Define the blueprint: 'auth', set its url prefix: app.url/auth for authentication-related routes
# This organizes related views under the '/auth' URL prefix
auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

# Set the route and accepted methods
# Define the route for user sign-in and allow both GET and POST HTTP methods


@auth_bp.route('/signin/', methods=['GET', 'POST'])
def signin():

    # Create an instance of the login form and bind it to request data
    form = LoginForm(request.form)

    # Verify the sign in form if it passes all validation rules
    if form.validate_on_submit():

        # Query the database for a user with the provided email
        user = User.query.filter_by(email=form.email.data).first()

    # Check if user exists and password is correct
        if user and check_password_hash(user.password, form.password.data):

            # Store the user's ID in the session to keep them logged in
            session['user_id'] = user.id

    # Show a success message to the user
            flash('Welcome %s' % user.name)

    # Redirect the user to their homepage or dashboard
            return redirect(url_for('auth.home'))

    # If login fails, show an error messsage
        flash('Wrong email or password', 'error-message')

    # If the form has not been submitted or there is a validation error, render the login page
    return render_template("auth/signin.html", form=form)
