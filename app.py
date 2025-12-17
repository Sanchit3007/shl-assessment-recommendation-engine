from flask import Flask, request, jsonify
from flask_cors import CORS
from recommender import SHLRecommender

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load recommendation engine
engine = SHLRecommender("data/shl_catalogue.csv")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "SHL Assessment Recommendation API is running"
    })


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()

    job_role = data.get("job_role")
    skills = data.get("skills")
    experience = data.get("experience")
    max_duration = data.get("max_duration")

    if not job_role or not skills or not experience:
        return jsonify({
            "error": "job_role, skills, and experience are required"
        }), 400

    results = engine.recommend(
        job_role=job_role,
        skills=skills,
        experience=experience,
        max_duration=max_duration
    )

    return jsonify(results.to_dict(orient="records"))


if __name__ == "__main__":
    app.run(debug=True)
