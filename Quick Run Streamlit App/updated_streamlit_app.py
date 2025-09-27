# streamlit_app.py
import streamlit as st
import pandas as pd
import json
import os
import openai

# ============ HELPER FUNCTIONS ============

def normalize_columns(df):
    """Lowercase and replace spaces with underscores for consistency."""
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
    return df

def load_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return normalize_columns(df)

def load_insights(uploaded_file):
    return json.load(uploaded_file)

def ai_summary(df, insights, ttests):
    """
    Generate a detailed, structured executive summary from dataset stats, insights, and validation results.
    """

    # Build extended context
    context = {
        "dataset_overview": {
            "total_apps": len(df),
            "num_categories": df["category"].nunique(),
            "avg_rating": round(df["rating"].mean(), 2),
            "median_rating": round(df["rating"].median(), 2),
            "avg_reviews": round(df["reviews"].mean(), 2),
            "median_reviews": round(df["reviews"].median(), 2),
            "total_reviews": int(df["reviews"].sum()),
            "avg_installs": round(df["installs"].mean(), 2),
            "median_installs": round(df["installs"].median(), 2),
            "content_ratings_count": df["content_rating"].nunique(),
            "platforms": df["store"].unique().tolist(),
            "min_age_range": [int(df["min_age"].min()), int(df["min_age"].max())],
            "avg_price": round(df["price"].mean(), 2),
            "median_price": round(df["price"].median(), 2),
            "free_vs_paid_split": df["type"].value_counts(normalize=True).to_dict(),
            "category_distribution_top10": df["category"].value_counts().head(10).to_dict(),
            "genre_distribution_top10": df["genres"].value_counts().head(10).to_dict(),
            "size_MB_stats": {
                "mean": round(df["size"].mean(), 2),
                "median": round(df["size"].median(), 2),
                "min": round(df["size"].min(), 2),
                "max": round(df["size"].max(), 2),
            },
            "os_version_range": [df["os_version"].min(), df["os_version"].max()],
            "version_range": [df["version"].min(), df["version"].max()],
            "last_updated_range": [str(df["last_updated"].min()), str(df["last_updated"].max())],
        },
        "insights": insights.get("insights", []),
        "ttests": ttests,
    }

    # Structured prompt
    messages = [
        {"role": "system", "content": "You are a strategy consultant specializing in the mobile app ecosystem. Write clear, structured executive reports."},
        {"role": "user", "content": f"""
        Use the following context to create a comprehensive executive summary:
        {json.dumps(context, indent=2)}

        Requirements:
        - Organize into 5–7 sections (Market Overview, Category Trends, Free vs Paid, Content Ratings & Demographics, Platform Insights, Update & Size Analysis, Strategic Recommendations).
        - Highlight statistically validated findings (mention confidence levels or p-values).
        - Compare free vs paid apps, small vs large, recent vs old, and top categories.
        - Provide actionable recommendations for investors, developers, and strategists.
        - Use clear, professional business language suitable for executives.
        """}
    ]

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.5
    )

    return response.choices[0].message.content



def ai_qa(df, insights, ttests, question):
    """
    Answer business questions using dataset stats, insights, and statistical validations,
    with structured output (Finding, Supporting Stats, Recommendation).
    """

    # Extended dataset-level stats
    context = {
        "dataset_overview": {
            "total_apps": len(df),
            "num_categories": df["category"].nunique(),
            "avg_rating": round(df["rating"].mean(), 2),
            "median_rating": round(df["rating"].median(), 2),
            "avg_reviews": round(df["reviews"].mean(), 2),
            "median_reviews": round(df["reviews"].median(), 2),
            "avg_installs": round(df["installs"].mean(), 2),
            "median_installs": round(df["installs"].median(), 2),
            "avg_price": round(df["price"].mean(), 2),
            "median_price": round(df["price"].median(), 2),
            "avg_size_mb": round(df["size"].mean(), 2),
            "median_size_mb": round(df["size"].median(), 2),
            "platforms": df["store"].unique().tolist(),
            "free_vs_paid_split": df["type"].value_counts(normalize=True).to_dict(),
            "content_ratings_unique": df["content_rating"].nunique(),
            "min_age_range": [int(df["min_age"].min()), int(df["min_age"].max())],
            "category_distribution_top10": df["category"].value_counts().head(10).to_dict(),
            "genre_distribution_top10": df["genres"].value_counts().head(10).to_dict(),
            "most_recent_update": str(df["last_updated"].max()),
            "oldest_update": str(df["last_updated"].min()),
            "os_version_range": [df["os_version"].min(), df["os_version"].max()],
            "version_range": [df["version"].min(), df["version"].max()],
        },
        "insights": insights.get("insights", []),
        "ttests": ttests,
        "user_question": question,
    }

    # Few-shot examples
    examples = """
    === EXAMPLES OF GOOD ANSWERS ===

    Example 1:
    Question: "Which performs better: free or paid apps?"
    **Finding:** Paid apps achieve higher average ratings (4.3 vs 4.1) than free apps.
    **Supporting Stats:** T-test p-value = 0.03 (significant). Free apps = 9,600, Paid apps = 1,200.
    **Recommendation:** A premium model could unlock higher ratings, while free apps ensure market reach.

    Example 2:
    Question: "Do recently updated apps perform better?"
    **Finding:** Recently updated apps have notably higher ratings (4.30 vs 4.05).
    **Supporting Stats:** T-test p-value = 0.01 (significant). Recent apps = 7,800, Old apps = 3,000.
    **Recommendation:** Prioritize regular updates to maintain competitiveness.

    Example 3:
    Question: "How does app size affect user ratings?"
    **Finding:** Larger apps (> median 50MB) show slightly higher ratings (4.25 vs 4.18).
    **Supporting Stats:** Difference not statistically significant (p=0.08). Small apps = 5,200, Large apps = 5,600.
    **Recommendation:** Focus on feature quality, not size inflation, since differences are minimal.

    === END OF EXAMPLES ===
    """

    # Refined prompt
    messages = [
        {
            "role": "system",
            "content": (
                "You are a data-driven strategy advisor for the mobile app market. "
                "Always structure answers into three parts: **Finding**, **Supporting Stats**, **Recommendation**. "
                "Be concise, professional, and business-friendly. "
                "Highlight statistical significance (p-values) where available."
            ),
        },
        {
            "role": "user",
            "content": f"""
            Context: {json.dumps(context, indent=2)}

            {examples}

            Task:
            - Answer the user's business question: "{question}"
            - Use dataset stats, insights, and validation results where relevant.
            - Compare averages vs medians if useful.
            - Mention top-performing categories/genres when relevant.
            - Keep tone executive-friendly.
            - Output must follow: Finding → Supporting Stats → Recommendation.
            """,
        },
    ]

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.6
    )

    return response.choices[0].message.content


from scipy import stats
import pandas as pd

def validate_free_vs_paid(df):
    free = df[df["type"] == "Free"]["rating"].dropna()
    paid = df[df["type"] == "Paid"]["rating"].dropna()
    t_stat, p_val = stats.ttest_ind(free, paid, equal_var=False)
    significant = p_val < 0.05
    interpretation = (
        f"Paid apps ({paid.mean():.2f}) have {'higher' if paid.mean() > free.mean() else 'lower'} "
        f"average ratings than Free apps ({free.mean():.2f}). "
        + ("The difference is statistically significant." if significant else "The difference is not statistically significant.")
    )
    return {
        "id": "free_vs_paid_ttest",
        "title": "Free vs Paid Ratings",
        "insight": interpretation,
        "supporting_stats": {
            "Measure of the difference between the two groups (t-statistic)": round(t_stat, 3),
            "Probability of observing the difference (p-value)": round(p_val, 4),
            "Average Rating of Free Apps": round(free.mean(), 2),
            "Average Rating of Paid Apps": round(paid.mean(), 2),
            "Number of Free Apps": len(free),
            "Number of Paid Apps": len(paid)
        },
        "confidence": round(1 - p_val, 3),
        "recommendation": "If ratings are significantly higher for Paid apps, consider a premium model. If not, focus on scale and ads."
    }


def validate_recent_vs_old(df, cutoff="2018-01-01"):
    df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
    recent = df[df["last_updated"] >= cutoff]["rating"].dropna()
    old = df[df["last_updated"] < cutoff]["rating"].dropna()
    t_stat, p_val = stats.ttest_ind(recent, old, equal_var=False)
    significant = p_val < 0.05
    interpretation = (
        f"Recently updated apps ({recent.mean():.2f}) have {'higher' if recent.mean() > old.mean() else 'lower'} "
        f"average ratings than older apps ({old.mean():.2f}). "
        + ("This difference is statistically significant." if significant else "No significant difference was found.")
    )
    return {
        "id": "recent_vs_old_ttest",
        "title": "Recently Updated vs Old Apps",
        "insight": interpretation,
        "supporting_stats": {
            "Measure of the difference between the two groups (t-statistic)": round(t_stat, 3),
            "Probability of observing the difference (p-value)": round(p_val, 4),
            "Average Rating of Recent Apps": round(recent.mean(), 2),
            "Average Rating of Old Apps": round(old.mean(), 2),
            "Number of Recent Apps": len(recent),
            "Number of Old Apps": len(old)
        },
        "confidence": round(1 - p_val, 3),
        "recommendation": "If recently updated apps score better, maintain frequent updates to keep users happy."
    }


def validate_category_comparison(df, cat1="GAME", cat2="PRODUCTIVITY"):
    g1 = df[df["category"].str.contains(cat1, case=False, na=False)]["rating"].dropna()
    g2 = df[df["category"].str.contains(cat2, case=False, na=False)]["rating"].dropna()
    t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
    significant = p_val < 0.05
    interpretation = (
        f"{cat1} apps ({g1.mean():.2f}) vs {cat2} apps ({g2.mean():.2f}). "
        + ("The rating difference is statistically significant." if significant else "No significant difference in ratings.")
    )
    return {
        "id": "category_vs_category_ttest",
        "title": f"{cat1} vs {cat2} Ratings",
        "insight": interpretation,
        "supporting_stats": {
            "Measure of the difference between the two groups (t-statistic)": round(t_stat, 3),
            "Probability of observing the difference (p-value)": round(p_val, 4),
            f"Average Rating of {cat1.lower()} Apps": round(g1.mean(), 2) if len(g1) else None,
            f"Average Rating of {cat2.lower()} Apps": round(g2.mean(), 2) if len(g2) else None,
            f"Number of {cat1.lower()} Apps": len(g1),
            f"Number of {cat2.lower()} Apps": len(g2)
        },
        "confidence": round(1 - p_val, 3),
        "recommendation": f"If {cat1} significantly outperforms {cat2}, prioritize {cat1} apps in your strategy."
    }


def validate_small_vs_large(df):
    median_size = df["size"].median()
    small = df[df["size"] <= median_size]["rating"].dropna()
    large = df[df["size"] > median_size]["rating"].dropna()
    t_stat, p_val = stats.ttest_ind(small, large, equal_var=False)
    significant = p_val < 0.05
    interpretation = (
        f"Smaller apps ({small.mean():.2f}) vs Larger apps ({large.mean():.2f}). "
        + ("The difference is statistically significant." if significant else "No significant difference in ratings.")
    )
    return {
        "id": "small_vs_large_ttest",
        "title": "Small vs Large Apps Ratings",
        "insight": interpretation,
        "supporting_stats": {
            "Measure of the difference between the two groups (t-statistic)": round(t_stat, 3),
            "Probability of observing the difference (p-value)": round(p_val, 4),
            "Average Rating of Small Apps": round(small.mean(), 2),
            "Average Rating of Large Apps": round(large.mean(), 2),
            "Number of Small Apps": len(small),
            "Number of Large Apps": len(large)
        },
        "confidence": round(1 - p_val, 3),
        "recommendation": "If larger apps significantly outperform smaller ones, invest in richer features."
    }



import matplotlib.pyplot as plt

def plot_top_categories(df, out_path="top_categories.png"):
    top = df["category"].value_counts().head(10)
    top.plot(kind="barh", figsize=(8,5), color="skyblue")
    plt.title("Top 10 Categories by App Count")
    plt.xlabel("Number of Apps")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_rating_distribution(df, out_path="rating_dist.png"):
    df["rating"].dropna().plot(kind="hist", bins=20, color="lightgreen", edgecolor="black")
    plt.title("App Ratings Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_content_rating(df, out_path="content_rating.png"):
    df["content_rating"].value_counts().plot(kind="pie", autopct="%1.1f%%", figsize=(6,6))
    plt.title("Content Rating Distribution")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, ListFlowable, ListItem
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from datetime import datetime

def generate_report(df, insights, ai_summary, ttests, out_file="executive_report.pdf"):
    doc = SimpleDocTemplate(out_file, pagesize=A4)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'TitleStyle', parent=styles['Title'], alignment=TA_CENTER, fontSize=20, spaceAfter=20
    )
    section_header = ParagraphStyle(
        'SectionHeader', parent=styles['Heading2'], spaceBefore=15, spaceAfter=10
    )
    subheader = ParagraphStyle(
        'SubHeader', parent=styles['Heading3'], textColor="blue", spaceBefore=10
    )
    normal_text = ParagraphStyle(
        'NormalText', parent=styles['Normal'], fontSize=10, leading=14
    )

    story = []

    # --- COVER PAGE ---
    story.append(Paragraph("AI - POWERED MARKET INTELLIGENCE", title_style))
    story.append(Paragraph(datetime.now().strftime("%B %d, %Y"), styles["Normal"]))
    story.append(Spacer(1, 40))
    story.append(Paragraph("Prepared by: Vishnu Vardhan P R", styles["Italic"]))
    story.append(PageBreak())

    # --- EXECUTIVE SUMMARY ---
    story.append(Paragraph("🤖 Executive Summary", section_header))
    for line in ai_summary.split("\n"):
        if line.strip():
            story.append(Paragraph(line.strip(), normal_text))
    story.append(PageBreak())

    # --- INSIGHTS ---
    story.append(Paragraph("💡 Key Insights", section_header))
    items = []
    for ins in insights["insights"]:
        text = f"<b>{ins['title']}</b>: {ins['insight']} <br/><i>Recommendation:</i> {ins['recommendation']}"
        items.append(ListItem(Paragraph(text, normal_text), bulletColor="black"))
    story.append(ListFlowable(items, bulletType="bullet"))
    story.append(PageBreak())

    # --- STATISTICAL VALIDATION ---
    story.append(Paragraph("📊 Statistical Validation (t-tests)", section_header))
    for test in ttests:
        story.append(Paragraph(f"<b>{test['title']}</b>", subheader))
        story.append(Paragraph(test["insight"], normal_text))
        stats_text = ", ".join([f"{k}: {v}" for k, v in test["supporting_stats"].items()])
        story.append(Paragraph(f"<i>Stats:</i> {stats_text}", styles["Code"]))
        story.append(Paragraph(f"Confidence: {test['confidence']:.2f}", styles["Italic"]))
        story.append(Spacer(1, 10))
    story.append(PageBreak())

    # --- EDA PLOTS ---
    story.append(Paragraph("📈 Exploratory Data Analysis", section_header))
    plot_top_categories(df)
    plot_rating_distribution(df)
    plot_content_rating(df)

    story.append(Paragraph("Top 10 Categories by App Count", styles["Italic"]))
    story.append(Image("top_categories.png", width=400, height=250))
    story.append(Spacer(1, 15))

    story.append(Paragraph("App Ratings Distribution", styles["Italic"]))
    story.append(Image("rating_dist.png", width=400, height=250))
    story.append(Spacer(1, 15))

    story.append(Paragraph("Content Rating Distribution", styles["Italic"]))
    story.append(Image("content_rating.png", width=300, height=300))
    story.append(PageBreak())

    # --- T-TEST PLOTS (OPTIONAL) ---
    story.append(Paragraph("📉 Validation Plots", section_header))
    st_plots = [
        plot_free_vs_paid(df),
        plot_recent_vs_old(df),
        plot_category_comparison(df),
        plot_small_vs_large(df)
    ]
    captions = [
        "Free vs Paid Ratings",
        "Recent vs Old App Ratings",
        "Games vs Productivity Ratings",
        "Small vs Large App Ratings"
    ]
    for p, c in zip(st_plots, captions):
        story.append(Paragraph(c, styles["Italic"]))
        story.append(Image(p, width=400, height=250))
        story.append(Spacer(1, 15))

    # Build report
    doc.build(story)
    return out_file


import matplotlib.pyplot as plt

def plot_free_vs_paid(df, out_path="free_vs_paid.png"):
    data = [
        df[df["type"]=="Free"]["rating"].dropna(),
        df[df["type"]=="Paid"]["rating"].dropna()
    ]
    plt.figure(figsize=(6,5))
    plt.boxplot(data, labels=["Free", "Paid"])
    plt.title("Free vs Paid App Ratings")
    plt.ylabel("Rating")
    plt.savefig(out_path)
    plt.close()
    return out_path

def plot_recent_vs_old(df, cutoff="2018-01-01", out_path="recent_vs_old.png"):
    df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
    recent = df[df["last_updated"] >= cutoff]["rating"].dropna()
    old = df[df["last_updated"] < cutoff]["rating"].dropna()
    plt.figure(figsize=(6,5))
    plt.bar(["Recent", "Old"], [recent.mean(), old.mean()], color=["green","orange"])
    plt.title("Average Ratings: Recent vs Old Apps")
    plt.ylabel("Rating")
    plt.savefig(out_path)
    plt.close()
    return out_path

def plot_category_comparison(df, cat1="GAME", cat2="PRODUCTIVITY", out_path="category_comparison.png"):
    g1 = df[df["category"].str.contains(cat1, case=False, na=False)]["rating"].dropna()
    g2 = df[df["category"].str.contains(cat2, case=False, na=False)]["rating"].dropna()
    plt.figure(figsize=(6,5))
    plt.bar([cat1, cat2], [g1.mean(), g2.mean()], color=["blue","purple"])
    plt.title(f"Average Ratings: {cat1} vs {cat2}")
    plt.ylabel("Rating")
    plt.savefig(out_path)
    plt.close()
    return out_path

def plot_small_vs_large(df, out_path="small_vs_large.png"):
    median_size = df["size"].median()
    small = df[df["size"] <= median_size]["rating"].dropna()
    large = df[df["size"] > median_size]["rating"].dropna()
    plt.figure(figsize=(6,5))
    plt.boxplot([small, large], labels=["Small Apps", "Large Apps"])
    plt.title("Ratings: Small vs Large Apps")
    plt.ylabel("Rating")
    plt.savefig(out_path)
    plt.close()
    return out_path




# ============ STREAMLIT UI ============

st.set_page_config(page_title="App Market Intelligence", layout="wide")
st.title("📊 App Market Intelligence Dashboard")

# Sidebar for file uploads
st.sidebar.header("Upload Data")
csv_file = st.sidebar.file_uploader("Upload combined CSV", type=["csv"])
json_file = st.sidebar.file_uploader("Upload insights JSON", type=["json"])

df, insights = None, None

# Load CSV
if csv_file:
    df = load_csv(csv_file)
    st.sidebar.success(f"Loaded CSV with {df.shape[0]} rows.")

# Load insights JSON
if json_file:
    insights = load_insights(json_file)
    st.sidebar.success("Loaded insights.json")

# ---- DATA PREVIEW ----
if df is not None:
    st.subheader("📂 Data Preview")
    st.dataframe(df.head(20))

# ---- INSIGHTS ----
if insights:
    st.subheader("💡 Insights (from insights.json)")
    for ins in insights["insights"]:
        with st.expander(f"{ins['title']}"):
            st.write(ins["insight"])
            st.json(ins["supporting_stats"])
            st.markdown(f"**Recommendation:** {ins['recommendation']}")
            st.caption(f"Confidence: {ins['confidence']:.2f}")

    st.subheader("📊 Statistical Validation (t-tests)")
    try:
        # Run all tests
        ttests = [
            validate_free_vs_paid(df),
            validate_recent_vs_old(df),
            validate_category_comparison(df, "GAME", "PRODUCTIVITY"),
            validate_small_vs_large(df)
        ]

        # Show results
        for test in ttests:
            with st.expander(test["title"]):
                st.write(test["insight"])  # plain-English interpretation
                st.json(test["supporting_stats"])  # raw stats
                st.caption(f"Confidence: {test['confidence']:.2f}")
                st.markdown(f"**Recommendation:** {test['recommendation']}")
    except Exception as e:
        st.error(f"Error running statistical tests: {e}")
        ttests = []

    if st.checkbox("Show validation plots"):
        try:
            plot_free_vs_paid(df)
            st.image("free_vs_paid.png")

            plot_recent_vs_old(df)
            st.image("recent_vs_old.png")

            plot_category_comparison(df, "GAME", "PRODUCTIVITY")
            st.image("category_comparison.png")

            plot_small_vs_large(df)
            st.image("small_vs_large.png")
        except Exception as e:
            st.error(f"Error displaying validation plots: {e}")

    # ---- AI SUMMARY ----
    st.subheader("🤖 AI-Generated Executive Summary")
    if st.button("Generate AI Summary"):
        try:
            summary = ai_summary(df, insights, ttests)
            st.success("Executive Summary:")
            st.write(summary)
        except Exception as e:
            st.error(f"Error generating AI summary: {e}")

    # ---- FREE-TEXT Q&A ----
    st.subheader("🔎 Ask the AI a Question")
    user_q = st.text_input("Type your question here (e.g. 'Which categories should we invest in?')")

    if user_q and st.button("Ask AI"):
        try:
            answer = ai_qa(df, insights, ttests, user_q)
            st.write(answer)
        except Exception as e:
            st.error(f"Error during AI Q&A: {e}")


    import tempfile

    st.subheader("📑 Automated Report Generation")

    if st.button("Generate Executive Report (PDF)"):
        try:
            ai_summary = ai_summary(df, insights, ttests)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                pdf_path = generate_report(df, insights, ai_summary, ttests, tmp.name)
                with open(pdf_path, "rb") as f:
                    st.download_button("⬇️ Download Report", f, file_name="executive_report.pdf")
        except Exception as e:
            st.error(f"Error during Report Generation: {e}")
