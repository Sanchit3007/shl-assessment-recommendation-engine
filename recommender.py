import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SHLRecommender:
    def __init__(self, csv_path):
        # Load the dataset
        self.df = pd.read_csv(csv_path)

        # Combine important text fields
        self.df["combined_text"] = (
            self.df["skills"] + " " +
            self.df["job_role"] + " " +
            self.df["description"]
        )

        # Convert text into TF-IDF vectors
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
        # Create query text
        query = f"{job_role} {skills} {experience}"

        # Convert query to vector
        query_vector = self.vectorizer.transform([query])

        # Compute similarity scores
        similarity_scores = cosine_similarity(
            query_vector, self.tfidf_matrix
        ).flatten()

        # Attach scores
        self.df["similarity_score"] = similarity_scores

        # Apply experience filter
        filtered_df = self.df[
            (self.df["experience_level"] == experience) |
            (self.df["experience_level"] == "Any")
        ]

        # Apply duration filter (if provided)
        if max_duration is not None:
            filtered_df = filtered_df[
                filtered_df["duration"] <= max_duration
            ]

        # Sort and get top results
        results = filtered_df.sort_values(
            by="similarity_score",
            ascending=False
        ).head(top_n)

        return results[
            [
                "assessment_name",
                "skills",
                "duration",
                "experience_level",
                "similarity_score",
            ]
        ]
