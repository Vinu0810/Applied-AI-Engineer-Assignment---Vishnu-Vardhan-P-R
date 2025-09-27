import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

# Load .env file (so we can use RAPIDAPI_KEY and RAPIDAPI_HOST)
load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST")

if not RAPIDAPI_KEY or not RAPIDAPI_HOST:
    raise RuntimeError("Set RAPIDAPI_KEY and RAPIDAPI_HOST in your .env file!")

DEFAULT_HEADERS = {
    "x-rapidapi-key": RAPIDAPI_KEY,
    "x-rapidapi-host": RAPIDAPI_HOST,
    "Accept": "application/json"
}

def rapidapi_get(url, params=None, headers=None, retries=2, backoff_factor=1.5, timeout=30):
    """ Generic GET wrapper for RapidAPI endpoints with retries """
    if headers is None:
        headers = DEFAULT_HEADERS

    attempt = 0
    while attempt < retries:
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)

            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError:
                    return {}

            if r.status_code in (429, 503):  # Rate limit or server busy
                print(f"⚠️ Rate limited (status {r.status_code}).")
                return {"rate_limited": True}  # ✅ special flag

            # 🚨 skip retries on client errors
            if r.status_code in (400, 401, 403, 404, 422):
                print(f"❌ Bad request ({r.status_code}) for URL={url} params={params}")
                return {}

            r.raise_for_status()

        except requests.RequestException as e:
            print(f"Request error: {e}. Retrying...")
            attempt += 1
            time.sleep(backoff_factor * (2 ** attempt))

    print(f"❌ Failed after {retries} attempts for URL: {url}")
    return {}

def transform_appstore_data(data, app_name):
    """ Transform raw API response into a flat dict suitable for CSV """
    return {
        "App": app_name,
        "as_app_id": data.get("id"),
        "as_title": data.get("title"),
        "as_store": "apple",
        "as_genres": ", ".join(data.get("genres", [])),
        "as_primary_genre": data.get("primaryGenre"),
        "as_content_rating": data.get("contentRating"),
        "as_size": data.get("size"),
        "as_required_os_version": data.get("requiredOsVersion"),
        "as_released": data.get("released"),
        "as_updated": data.get("updated"),
        "as_version": data.get("version"),
        "as_price": data.get("price"),
        "as_free": data.get("free"),
        "as_score": data.get("score"),
        "as_reviews": data.get("reviews"),
        "as_current_version_score": data.get("currentVersionScore"),
        "as_current_version_reviews": data.get("currentVersionReviews"),
    }

if __name__ == "__main__":
    # Load Play Store apps
    df_playstore = pd.read_csv("dataset/cleaned_googleplaystore.csv")
    app_names = df_playstore["App"].dropna().unique().tolist()[1572 : ]
    print(f"Loaded {len(app_names)} app names from Play Store data.")

    results = []
    search_url = "https://appstore-scrapper-api.p.rapidapi.com/v1/app-store-api/search"
    detail_url = "https://appstore-scrapper-api.p.rapidapi.com/v1/app-store-api/detail"

    output_file = "appstore_apps_by_name_4.csv"

    for i, app_name in enumerate(app_names, 1):
        print(f"[{i}/{len(app_names)}] Searching App Store for '{app_name}'...")
        search_params = {"query": app_name, "country": "us", "lang": "en", "num": 1}
        search_data = rapidapi_get(search_url, params=search_params)

        # ✅ Stop and save progress if rate limited
        if isinstance(search_data, dict) and search_data.get("rate_limited"):
            print("⚠️ Rate limit reached. Saving progress and exiting...")
            pd.DataFrame(results).to_csv(output_file, index=False, encoding="utf-8")
            print(f"✅ Partial data saved to {output_file}")
            break

        if not search_data or len(search_data) == 0:
            print(f"⚠️ No results for {app_name}")
            continue

        app_id = search_data[0]["id"]
        detail_params = {"id": app_id, "country": "us", "lang": "en"}
        detail_data = rapidapi_get(detail_url, params=detail_params)

        if isinstance(detail_data, dict) and detail_data.get("rate_limited"):
            print("⚠️ Rate limit reached during detail fetch. Saving progress and exiting...")
            pd.DataFrame(results).to_csv(output_file, index=False, encoding="utf-8")
            print(f"✅ Partial data saved to {output_file}")
            break

        if detail_data:
            transformed = transform_appstore_data(detail_data, app_name)
            results.append(transformed)

        # 💤 slow down slightly to avoid hitting limit too fast
        time.sleep(0.5)

    # Save results if loop completes
    if results:
        df_appstore = pd.DataFrame(results)
        df_appstore.to_csv(output_file, index=False, encoding="utf-8")
        print(f"✅ Saved {len(df_appstore)} apps with details to {output_file}")