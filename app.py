from flask import Flask, request, jsonify
from flask_cors import CORS
from recommender import SHLRecommender
import os

app = Flask(__name__)
CORS(app)

engine = SHLRecommender("data/shl_catalogue.csv")

@app.route("/")
def home():
    return {"status": "Backend running"}

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    results = engine.recommend(
        job_role=data["job_role"],
        skills=data["skills"],
        experience=data["experience"],
        max_duration=data.get("max_duration"),
        top_n=3
    )
    return jsonify(results)

if __name__ == "__main__":
    app.run()
