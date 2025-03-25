from flask import Blueprint, jsonify

api_bp = Blueprint('api', __name__)

@api_bp.route('/status')
def api_status():
    return jsonify(status='ok')