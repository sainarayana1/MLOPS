from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "Flask app is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    result = {"prediction": "success"}
    return jsonify(result)

def run_app():
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False
    )

if __name__ == "__main__":
    run_app()
