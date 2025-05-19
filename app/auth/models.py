# Import the database object (db) from the main application module
# We will define this inside /app/__init__.py in the next sections.
from app import db

# Define a base model for other database tables to inherit
# This prevents code duplication for common fields like id, date_created, etc


class Base(db.Model):

    __abstract__ = True  # This tells SQLAlchemy not to create a table for this class

    # Primary key column
    id = db.Column(db.Integer, primary_key=True)

    # Timestamp when row is created
    date_created = db.Column(db.DateTime,  default=db.func.current_timestamp())

    # Timestamp when the row is updated; automatically updated on change
    date_modified = db.Column(db.DateTime,  default=db.func.current_timestamp(),
                              onupdate=db.func.current_timestamp())

# Define a User model that represents users in the system


class User(Base):

    __tablename__ = 'user'  # Set the name of the table in the database

    # User Name field; required
    name = db.Column(db.String(128),  nullable=False)

    # Identification Data: email & password, must be unique
    email = db.Column(db.String(128),  nullable=False,
                      unique=True)

    # password field; required (stored as a hashed string)
    password = db.Column(db.String(192),  nullable=False)

    # Location field required
    location = db.Column(db.String(128),  nullable=False)

    # New instance instantiation procedure
    # Constructor to initialize a new user object
    def __init__(self, name, email, password, location):

        self.name = name
        self.email = email
        self.password = password  # Expected to be hashed before saving
        self.location = location

    # STring representation of the User object, useful for debugging
    def __repr__(self):
        return '<User %r>' % (self.name)
