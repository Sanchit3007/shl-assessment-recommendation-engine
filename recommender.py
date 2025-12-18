import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SHLRecommender:
    def __init__(self, csv_path):
        """
        Load SHL catalogue CSV safely across
        local Windows + Render Linux environment
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, csv_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"CSV file not found at: {full_path}")

        self.df = pd.read_csv(full_path)

        # Combine relevant text fields
        self.df["combined_text"] = (
            self.df["skills"].astype(str) + " " +
            self.df["job_role"].astype(str) + " " +
            self.df["description"].astype(str)
        )

        # TF-IDF Vectorization
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df["combined_text"]
        )

    def recommend(
        self,
        job_role,
        skills,
        experience,
        max_duration=None,
        top_n=3
    ):
        # Build query
        query = f"{job_role} {skills} {experience}"

        # Transform query
        query_vector = self.vectorizer.transform([query])

        # Compute similarity
        similarity_scores = cosine_similarity(
            query_vector,
            self.tfidf_matrix
        ).flatten()

        self.df["similarity"] = similarity_scores

        # Optional filter
        if max_duration and "duration" in self.df.columns:
            self.df = self.df[self.df["duration"] <= max_duration]

        # Top N results
        results = (
            self.df
            .sort_values(by="similarity", ascending=False)
            .head(top_n)
        )

        return results[
            ["assessment_name", "similarity"]
        ].to_dict(orient="records")
