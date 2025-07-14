from flask import Flask, request, jsonify
from api.models import Gemini11ProAPI
from utils.logger import get_logger
from aegis_service import AegisService

app = Flask(__name__)
logger = get_logger(__name__)
aegis = AegisService()

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for prediction requests with security validation"""
    try:
        data = request.get_json()
        if not aegis.validate_request(data):
            return jsonify({'error': 'Security validation failed'}), 403
            
        api_model = Gemini11ProAPI()
        output = api_model.predict(data)
        return jsonify({'output': output})
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/healthcheck')
def healthcheck():
    """System health monitoring endpoint"""
    return jsonify({'status': 'ok', 'hyperintelligence': True})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)
