import pandas as pd
import json
from datetime import datetime

# Load CSV
df = pd.read_csv("dataset/cleaned_combined.csv")

# --- 1. Best category by total reviews ---
cat_reviews = df.groupby("Category")["Reviews"].sum()
best_category = cat_reviews.idxmax()
best_category_reviews = cat_reviews.max()
best_category_rating = df[df["Category"] == best_category]["Rating"].mean()

# --- 2. Free vs Paid rating comparison ---
free_avg = df[df["Type"] == "Free"]["Rating"].mean()
paid_avg = df[df["Type"] == "Paid"]["Rating"].mean()

# --- 3. Top 5 categories by number of apps ---
top_categories = df["Category"].value_counts().head(5).to_dict()

# --- 4. Most expensive category (avg price) ---
cat_prices = df.groupby("Category")["Price"].mean()
most_expensive_category = cat_prices.idxmax()
avg_price_expensive = cat_prices.max()

# --- 5. Content Rating distribution ---
content_distribution = df["Content Rating"].value_counts(normalize=True).round(2).to_dict()

# --- 6. Highest rated category (avg rating) ---
cat_ratings = df.groupby("Category")["Rating"].mean()
highest_rated_category = cat_ratings.idxmax()
highest_rated_value = cat_ratings.max()

# --- 7. Lowest rated category (avg rating) ---
lowest_rated_category = cat_ratings.idxmin()
lowest_rated_value = cat_ratings.min()

# --- 8. Most common genre ---
top_genre = df["Genres"].value_counts().idxmax()
top_genre_count = df["Genres"].value_counts().max()

# --- 9. Store comparison ---
store_counts = df["Store"].value_counts().to_dict()

# --- 10. Recently updated vs older apps (rating difference) ---
df["Last Updated"] = pd.to_datetime(df["Last Updated"], errors="coerce")
recent_cutoff = pd.to_datetime("2022-01-01")
recent_avg = df[df["Last Updated"] >= recent_cutoff]["Rating"].mean()
older_avg = df[df["Last Updated"] < recent_cutoff]["Rating"].mean()

# --- Build insights dictionary ---
insights = {
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "insights": [
        {
            "id": "best_category",
            "title": "Best performing category (by reviews)",
            "insight": f"{best_category} has the most reviews ({int(best_category_reviews)} total) and an average rating of {best_category_rating:.2f}.",
            "confidence": 0.9,
            "supporting_stats": {
                "total_reviews": int(best_category_reviews),
                "avg_rating": round(float(best_category_rating), 2)
            },
            "recommendation": f"Prioritize {best_category} apps for marketing."
        },
        {
            "id": "free_vs_paid",
            "title": "Free vs Paid apps",
            "insight": f"Free apps average rating = {free_avg:.2f}, Paid apps average rating = {paid_avg:.2f}.",
            "confidence": 0.8,
            "supporting_stats": {
                "avg_rating_free": round(float(free_avg), 2),
                "avg_rating_paid": round(float(paid_avg), 2),
                "free_count": int(df[df['Type']=='Free'].shape[0]),
                "paid_count": int(df[df['Type']=='Paid'].shape[0])
            },
            "recommendation": "Evaluate if premium pricing correlates with higher quality."
        },
        {
            "id": "top_categories",
            "title": "Top 5 categories by app count",
            "insight": f"Most apps belong to: {list(top_categories.keys())}.",
            "confidence": 0.85,
            "supporting_stats": top_categories,
            "recommendation": "Expect strong competition in these categories."
        },
        {
            "id": "most_expensive_category",
            "title": "Most expensive app category",
            "insight": f"On average, {most_expensive_category} apps cost ${avg_price_expensive:.2f}.",
            "confidence": 0.75,
            "supporting_stats": {
                "category": most_expensive_category,
                "avg_price": round(float(avg_price_expensive), 2)
            },
            "recommendation": f"Consider premium positioning in {most_expensive_category}."
        },
        {
            "id": "content_rating",
            "title": "Content rating distribution",
            "insight": "Apps are spread across content ratings.",
            "confidence": 0.95,
            "supporting_stats": content_distribution,
            "recommendation": "Align marketing with the dominant audience segment."
        },
        {
            "id": "highest_rated_category",
            "title": "Highest rated category",
            "insight": f"{highest_rated_category} has the highest average rating ({highest_rated_value:.2f}).",
            "confidence": 0.9,
            "supporting_stats": {
                "category": highest_rated_category,
                "avg_rating": round(float(highest_rated_value), 2)
            },
            "recommendation": f"Learn from what makes {highest_rated_category} successful."
        },
        {
            "id": "lowest_rated_category",
            "title": "Lowest rated category",
            "insight": f"{lowest_rated_category} has the lowest average rating ({lowest_rated_value:.2f}).",
            "confidence": 0.9,
            "supporting_stats": {
                "category": lowest_rated_category,
                "avg_rating": round(float(lowest_rated_value), 2)
            },
            "recommendation": f"Investigate issues in {lowest_rated_category} apps."
        },
        {
            "id": "top_genre",
            "title": "Most popular genre",
            "insight": f"The most common genre is {top_genre} ({top_genre_count} apps).",
            "confidence": 0.85,
            "supporting_stats": {
                "genre": top_genre,
                "count": int(top_genre_count)
            },
            "recommendation": "Expect heavy competition in this genre."
        },
        {
            "id": "store_comparison",
            "title": "Store comparison",
            "insight": f"App distribution by store: {store_counts}.",
            "confidence": 0.95,
            "supporting_stats": store_counts,
            "recommendation": "Balance strategy between Google Play and Apple Store."
        },
        {
            "id": "recent_vs_old",
            "title": "Recently updated vs older apps",
            "insight": f"Recently updated apps (since 2022) average {recent_avg:.2f} stars vs older apps {older_avg:.2f}.",
            "confidence": 0.85,
            "supporting_stats": {
                "recent_avg_rating": round(float(recent_avg), 2),
                "older_avg_rating": round(float(older_avg), 2)
            },
            "recommendation": "Frequent updates may help maintain higher ratings."
        }
    ]
}

# Save JSON
with open("updated_insights.json", "w", encoding="utf-8") as f:
    json.dump(insights, f, indent=2, ensure_ascii=False)

print("✅ 10 insights saved to insights.json")
