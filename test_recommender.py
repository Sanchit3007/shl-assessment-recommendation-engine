from recommender import SHLRecommender

engine = SHLRecommender("data/shl_catalogue.csv")

results = engine.recommend(
    job_role="Data Analyst",
    skills="python sql data analysis",
    experience="Entry",
    max_duration=40,
    top_n=3
)

print("\nFiltered Recommended Assessments:\n")
print(results)
