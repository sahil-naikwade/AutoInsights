from flask import Flask, jsonify
from flask_cors import CORS
from utils.config import Config
from utils.logger import setup_logger

app = Flask(__name__)
CORS(app)
logger = setup_logger(__name__)

@app.route('/')
def home():
    return "AutoBI Dashboard Backend is running!"

@app.route('/api/status')
def status():
    return jsonify({
        "status": "running",
        "database": Config.DB_NAME,
        "host": Config.DB_HOST
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=Config.FLASK_DEBUG)