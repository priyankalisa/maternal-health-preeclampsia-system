import json
import os

# Store cache file inside the same folder (app/)
CACHE_FILE = os.path.join(os.path.dirname(__file__), "chat_cache.json")


def load_cache():
    """Load cache from file safely"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception):
            # If file is corrupted or empty
            return {}
    return {}


def save_cache(cache):
    """Save cache to file safely"""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Cache save error:", e)


def get_cached_response(question: str):
    """Return cached answer if exists"""
    cache = load_cache()
    return cache.get(question.lower().strip())


def set_cache(question: str, answer: str):
    """Save question-answer pair in cache"""
    cache = load_cache()
    cache[question.lower().strip()] = answer
    save_cache(cache)