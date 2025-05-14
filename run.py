import os

from app import app, socketio

port = int(os.environ.get("FLASK_PORT", 5000))

if __name__ == "__main__":
    socketio.run(app,host='0.0.0.0', port=port)