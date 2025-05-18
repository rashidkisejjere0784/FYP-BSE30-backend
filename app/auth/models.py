# Import the database object (db) from the main application module
# We will define this inside /app/__init__.py in the next sections.
from app import db

# Define a base model for other database tables to inherit
# This helps to avoid code duplication (e.g every table having 'id', 'date-modified')


class Base(db.Model):

    __abstract__ = True

    # Unique identifier for each record (Primary Key)
    id = db.Column(db.Integer, primary_key=True)

    # Automatically set the timestamp when the record is created
    date_created = db.Column(db.DateTime,  default=db.func.current_timestamp())

    # Automatically update the timestamp when the record is modified
    date_modified = db.Column(db.DateTime,  default=db.func.current_timestamp(),
                              onupdate=db.func.current_timestamp())

# Define a User model which stores user accounts information


class User(Base):

    __tablename__ = 'auth_user'

    # User Name (Required Field)
    name = db.Column(db.String(128),  nullable=False)

    # Identification Data: email & password, must be unique and is required
    email = db.Column(db.String(128),  nullable=False,
                      unique=True)

    # User's hashed password (Required)
    password = db.Column(db.String(192),  nullable=False)

    # Authorisation Data: role & status
    # Roles define what level of access the user has (e.g, admin= 1, Regular User= 2, etc)
    role = db.Column(db.SmallInteger, nullable=False)

    # Status could represent if the account is active (1), suspended (0), or other values
    status = db.Column(db.SmallInteger, nullable=False)

    # New instance instantiation procedure
    # Constructor method that gets called when creating anew user instance
    def __init__(self, name, email, password):

        self.name = name
        self.email = email
        self.password = password

# This method defines how a User object is represented when printed or logged
    def __repr__(self):
        return '<User %r>' % (self.name)  # e.g <User 'John Doe'>
