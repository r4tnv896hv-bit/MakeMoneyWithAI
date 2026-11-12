import requests
import os
import json
import csv
import time
from datetime import datetime, timedelta

GITHUB_API = "https://api.github.com/search/repositories"
HEADERS = {"Authorization": f"token {os.getenv('GITHUB_TOKEN')}"}
TOPICS = ["ai", "ai-agent", "llm"]  # Fetch separately for each topic
PER_PAGE = 50
PAGES = 10
MIN_STARS = 10000
MAX_REPOS = 100
OUTPUT_FILE = "README.md"  # Output file name changed to README.md
EXCLUDE_FILE = "excluded-repos.txt"  # File containing repos to exclude
EXTRA_FILE = "extra-repos.txt"  # File containing repos that are always included
CSV_FILE = "repos.csv"  # CSV file to store repository data
LOGO_CSV_FILE = "logos.csv"
AVATAR_URL = "https://github.com/{owner}.png?size=28"  # Used when a repo has no logo in logos.csv
REQUEST_TIMEOUT = 30  # Seconds; avoids a hung request stalling the whole run
OPENAI_TIMEOUT = 120
MAX_ATTEMPTS = 4  # Tries per GitHub request before giving up
MAX_RETRY_WAIT = 60  # Upper bound in seconds for a single wait between tries

# OpenAI API configuration
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-5-mini"

# Other services with an OpenAI-compatible API, picked by the prefix of the key
# stored in OPENAI_API_KEY. Both offer a free tier, so no paid account is needed.
GEMINI = ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "gemini-2.5-flash")
KEY_PREFIX_PROVIDERS = {
    "AIza": GEMINI,  # Older Google AI Studio keys
    "AQ.": GEMINI,  # Current Google AI Studio keys
    "gsk_": ("https://api.groq.com/openai/v1/chat/completions", "llama-3.3-70b-versatile"),
}
MAX_GENERATION_FAILURES = 3  # Consecutive failures before generation is skipped for the run
GENERATION_DELAY = 5  # Seconds between calls on free tiers, which allow few requests per minute


class RateLimited(Exception):
    """The model provider refused the call because its quota is used up."""


def model_provider():
    """Return (url, api key, model, delay between calls) for the configured key.

    OPENAI_API_URL and OPENAI_MODEL override the endpoint and model, so any
    OpenAI-compatible service can be used.
    """
    api_key = (os.getenv('OPENAI_API_KEY') or '').strip()
    url, model, delay = OPENAI_API_URL, OPENAI_MODEL, 0
    for prefix, (provider_url, provider_model) in KEY_PREFIX_PROVIDERS.items():
        if api_key.startswith(prefix):
            url, model, delay = provider_url, provider_model, GENERATION_DELAY
    url = (os.getenv('OPENAI_API_URL') or '').strip() or url
    model = (os.getenv('OPENAI_MODEL') or '').strip() or model
    return url, api_key, model, delay


def retry_wait(response, attempt):
    """Seconds to wait before the next try, honouring GitHub rate-limit headers."""
    wait = 2 ** attempt
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        reset = response.headers.get("X-RateLimit-Reset")
        try:
            if retry_after:
                wait = int(retry_after)
            elif response.headers.get("X-RateLimit-Remaining") == "0" and reset:
                wait = int(reset) - int(time.time()) + 1
        except ValueError:
            pass
    return max(1, min(wait, MAX_RETRY_WAIT))


def github_get(url, params=None):
    """GET a GitHub API URL, retrying network errors, rate limits and 5xx responses.

    Returns the last response (which may still be an error), or raises the last
    network error if no response was ever received.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = None
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            if attempt == MAX_ATTEMPTS:
                raise
            print(f"Request error for {url}: {e}")
        else:
            if response.status_code not in (403, 429) and response.status_code < 500:
                return response
            if attempt == MAX_ATTEMPTS:
                return response
            print(f"GitHub returned {response.status_code} for {url}")

        wait = retry_wait(response, attempt)
        print(f"Retrying in {wait}s (attempt {attempt} of {MAX_ATTEMPTS})")
        time.sleep(wait)


def fetch_repositories(topic):
    repos = []
    one_year_ago = datetime.now() - timedelta(days=365)  # Calculate the date one year ago
    for page in range(1, PAGES + 1):
        params = {
            "q": f"topic:{topic} stars:>={MIN_STARS} pushed:>={one_year_ago.strftime('%Y-%m-%d')}",
            "sort": "stars",  # Sort by stars
            "order": "desc",   # Descending order
            "per_page": PER_PAGE,
            "page": page
        }
        response = github_get(GITHUB_API, params=params)
        if response.status_code != 200:
            # Stop the run: continuing would publish a list built from partial results
            raise RuntimeError(
                f"GitHub search failed for topic [{topic}] page [{page}]: "
                f"{response.status_code} - {response.text[:200]}"
            )
        repos_curr = response.json().get("items", [])
        if len(repos_curr) == 0:
            break
        repos.extend(repos_curr)
        print(f"Repos added {len(repos_curr)} of {len(repos)} at topic [{topic}] page [{page}]")
    
    return repos


def format_stars(stars):
    """Format the number of stars to be in k or m units."""
    if stars >= 1_000_000:
        return f"{stars / 1_000_000:.1f}m"  # Format to 'm' for millions
    elif stars >= 1_000:
        return f"{stars / 1_000:.1f}k"  # Format to 'k' for thousands
    else:
        return str(stars)  # No formatting for less than 1000

def load_excluded():
    """
    Load excluded repositories from excluded-repos.txt.

    Supports two formats per line:
    - owner/repo       → matches that exact full_name
    - name-or-prefix   → matches repository name only
    """
    excluded_full_names = set()
    excluded_names = set()

    if not os.path.exists(EXCLUDE_FILE):
        return excluded_full_names, excluded_names

    with open(EXCLUDE_FILE, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            line_lower = line.lower()
            if "/" in line_lower:
                excluded_full_names.add(line_lower)
            else:
                excluded_names.add(line_lower)

    return excluded_full_names, excluded_names

def load_extra_repos():
    """Load additional repositories from extra-repos.txt."""
    extra_repos = []
    if not os.path.exists(EXTRA_FILE):
        return extra_repos

    with open(EXTRA_FILE, "r") as file:
        for line in file:
            owner_repo = line.strip()
            if owner_repo:
                owner, repo = owner_repo.split("/")
                extra_repos.append({"owner": owner, "name": repo})
    return extra_repos

def select_repos(repos_dict):
    """Apply exclusion rules and return at most MAX_REPOS repos sorted by stars.

    Repositories listed in extra-repos.txt are always kept, even when their star
    count would not place them in the top MAX_REPOS; the remaining slots are
    filled with the most-starred repositories.
    """
    excluded_full_names, excluded_names = load_excluded()
    pinned = {f"{repo['owner']}/{repo['name']}".lower() for repo in load_extra_repos()}

    candidates = []
    for repo_data in repos_dict.values():
        full_name_lower = f"{repo_data['owner']}/{repo_data['name']}".lower()
        if full_name_lower in excluded_full_names or repo_data["name"].lower() in excluded_names:
            continue
        candidates.append(repo_data)

    candidates.sort(key=lambda x: x['stars'], reverse=True)
    extras = [r for r in candidates if f"{r['owner']}/{r['name']}".lower() in pinned][:MAX_REPOS]
    others = [r for r in candidates if f"{r['owner']}/{r['name']}".lower() not in pinned]
    selected = extras + others[:MAX_REPOS - len(extras)]
    selected.sort(key=lambda x: x['stars'], reverse=True)
    return selected


def read_existing_repos_from_csv():
    """Read existing repositories from CSV file."""
    existing_repos = {}
    existing_repo_ids = set()
    
    if not os.path.exists(CSV_FILE):
        return existing_repos, existing_repo_ids
    
    try:
        with open(CSV_FILE, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                repo_id = int(row['id'])
                existing_repo_ids.add(repo_id)
                existing_repos[repo_id] = {
                    'id': repo_id,
                    'owner': row['owner'],
                    'name': row['name'],
                    'stars': int(row['stars']),
                    'url': row['url'],
                    'business_model': row.get('business_model', '')
                }
        print(f"Loaded {len(existing_repos)} repositories from CSV")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return {}, set()
    
    return existing_repos, existing_repo_ids

def fetch_extra_repo_details(extra_repos):
    """Fetch details for repositories listed in extra-repos.txt."""
    detailed_repos = []
    for repo in extra_repos:
        url = f"https://api.github.com/repos/{repo['owner']}/{repo['name']}"
        response = github_get(url)
        if response.status_code == 200:
            detailed_repos.append(response.json())
        else:
            print(f"Failed to fetch details for {repo['owner']}/{repo['name']}")
    return detailed_repos

def refresh_existing_repos(existing_repos):
    """Refresh cached metadata and remove repositories that are archived or gone."""
    for repo_id, repo_data in list(existing_repos.items()):
        url = f"https://api.github.com/repos/{repo_data['owner']}/{repo_data['name']}"
        response = github_get(url)
        if response.status_code == 404:
            # Deleted or made private: the README link would be dead
            print(f"Removing {repo_data['owner']}/{repo_data['name']}: no longer available")
            del existing_repos[repo_id]
            continue

        if response.status_code != 200:
            print(f"Failed to refresh details for {repo_data['owner']}/{repo_data['name']}")
            continue

        repository = response.json()
        if repository.get("archived"):
            del existing_repos[repo_id]
            continue

        owner = repository["owner"]["login"]
        existing_repos[repo_id].update({
            "owner": owner,
            "name": repository["name"],
            "stars": repository["stargazers_count"],
            "url": repository["html_url"]
        })

def needs_business_model(stored, description):
    """Return True when the stored text is missing or is only a fallback.

    When the OpenAI call fails we store the plain repository description so the
    row is not empty. Such rows must be retried on later runs, otherwise they
    keep the fallback forever. Generated analyses always contain bold keywords.
    """
    stored = (stored or "").strip()
    if not stored:
        return True
    if stored == (description or "").strip():
        return True
    return "**" not in stored


def generate_business_model(repository):
    prompt = f"""
You are an AI business consultant. Describe following repository in one sentence (around 20 words) to describe what capacities it has that can help me make money. 

- Repository: {repository['name']}
- Description: {repository['description'] or 'No description'}
- URL: {repository['html_url']}
- Stars: {repository['stargazers_count']}

Note:
- Highlight keywords in bold.
- Return only the 25-word business analysis as plain text (no JSON, no formatting, no extra explanation). 
- Do not include any other text than the business analysis.
- Do not include the repository name and URL in the business analysis.
"""

    url, api_key, model, delay = model_provider()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a business analyst expert in AI monetization. Provide concise, actionable business insights."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
    }
    
    if delay:
        time.sleep(delay)
    response = requests.post(url, headers=headers, json=payload, timeout=OPENAI_TIMEOUT)

    if response.status_code == 429:
        raise RateLimited(f"{model} rate limit reached: {response.text[:200]}")
    if response.status_code != 200:
        raise Exception(f"Model API error ({model}): {response.status_code} - {response.text[:300]}")
    
    try:
        business_model = response.json()['choices'][0]['message']['content'].strip()
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        hops = [f"{r.status_code} {r.url}" for r in getattr(response, "history", [])]
        raise Exception(
            f"Unexpected response from {model} "
            f"({response.headers.get('Content-Type')}, final URL {getattr(response, 'url', url)}, "
            f"redirects {hops}): {response.text[:300]!r}"
        )
    if not business_model:
        raise Exception(f"Empty answer from {model}")

    print(f"Generated business model for {repository['name']}")
    return business_model


def convert_csv_to_readme():
    """Read repository data and render the sorted README table with available logos.

    Applies the same exclusion rules as the fetch step so anything listed in
    excluded-repos.txt is filtered out from the final markdown as well.
    """
    # Read existing repos from CSV (including any new ones we just added)
    existing_repos, _ = read_existing_repos_from_csv()
    
    if not existing_repos:
        print("No repositories found in CSV")
        return
    
    # Apply exclusions, keep pinned extras, and sort by stars
    repos_list = select_repos(existing_repos)

    logos = {}
    with open(LOGO_CSV_FILE, 'r', newline='', encoding='utf-8') as file:
        for row in csv.DictReader(file):
            key = (row['owner'].lower(), row['name'].lower())
            logos.setdefault(key, row['logo'])

    def escape_cell(value):
        return str(value).replace('\n', ' ').replace('|', '\\|')

    with open(OUTPUT_FILE, "w") as file:
        file.write("# Make Money With AI\n\n")
        file.write("**Make Money With AI** is a curated list of AI tools and projects that help you turn open-source into income.\n\n")

        file.write("| NO | Name | Stars | How to make money with this repository |\n")
        file.write("| --- | --- | --- | --- |\n")
        for index, repo_data in enumerate(repos_list, start=1):
            stars = format_stars(repo_data['stars'])
            business_model = repo_data.get('business_model', '')
            logo = logos.get((repo_data['owner'].lower(), repo_data['name'].lower()))
            if not logo:
                # No curated logo: fall back to the owner's GitHub avatar
                logo = AVATAR_URL.format(owner=repo_data['owner'])
            image = f'<img src="{logo}" width="14" height="14" /> '
            name = f"{image}**[{repo_data['name']}]({repo_data['url']})**"
            file.write(
                f"| {index} | {escape_cell(name)} | "
                f"{escape_cell(f'⭐{stars}')} | {escape_cell(business_model)} |\n"
            )
    
    print(f"Converted {len(repos_list)} repositories from CSV to README.md")

def save_repos_to_csv(repos_dict):
    """Save all repositories to CSV file, applying exclusion rules."""
    if not repos_dict:
        print("No repositories to save to CSV")
        return
    
    # Apply exclusions, keep pinned extras, and sort by stars for CSV consistency
    repos_list = select_repos(repos_dict)
    
    # Save to CSV file
    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'owner', 'name', 'stars', 'url', 'business_model'])
        
        for repo_data in repos_list:
            writer.writerow([
                repo_data['id'],
                repo_data['owner'],
                repo_data['name'],
                repo_data['stars'],
                repo_data['url'],
                repo_data.get('business_model', '')
            ])
    
    print(f"Saved {len(repos_list)} repositories to CSV file")


def main(max_repos=None):
    if not os.getenv('GITHUB_TOKEN'):
        raise ValueError("GITHUB_TOKEN is not set")
    
    # Without a key the list is still refreshed; new repos show their plain
    # description until a later run with a key generates the analysis
    generation_enabled = bool(model_provider()[1])
    if generation_enabled:
        print(f"Generating business models with {model_provider()[2]}")
    else:
        print("OPENAI_API_KEY is not set: skipping business model generation")
    failures_in_a_row = 0

    # 1. Read existing repos from CSV file
    print("Reading existing repositories from CSV...")
    existing_repos, existing_repo_ids = read_existing_repos_from_csv()
    refresh_existing_repos(existing_repos)
    
    # 2. Fetch all repos from GitHub API
    excluded_full_names, excluded_names = load_excluded()

    # Add extra repositories first so they are always included even when max_repos is set
    extra_repos = load_extra_repos()
    extra_repo_details = fetch_extra_repo_details(extra_repos)
    print(f"Loaded {len(extra_repo_details)} extra repositories")

    all_repos = list(extra_repo_details)
    for topic in TOPICS:
        all_repos.extend(fetch_repositories(topic))
    
    # 3. Deduplicate and filter the fetched repos
    seen = set()
    unique_repos = []
    
    # Respect optional max_repos limit
    repos_to_process = all_repos if max_repos is None else all_repos[:max_repos]

    for repo in repos_to_process:
        # Skip archived repositories
        if repo.get("archived"):
            continue

        # Build identifiers for exclusion checks
        owner_login = repo["owner"]["login"]
        name = repo["name"]
        full_name = repo.get("full_name", f"{owner_login}/{name}")

        full_name_lower = full_name.lower()
        name_lower = name.lower()

        # Exclude if matches either full_name or name-only rule
        if full_name_lower in excluded_full_names or name_lower in excluded_names:
            continue

        if repo["id"] in seen:
            continue

        seen.add(repo["id"])
        unique_repos.append(repo)

        # Register repos that are not in the CSV yet; the text is filled in below
        if repo["id"] not in existing_repos:
            existing_repos[repo["id"]] = {
                'id': repo["id"],
                'owner': owner_login,
                'name': name,
                'stars': repo['stargazers_count'],
                'url': repo['html_url'],
                'business_model': ''
            }

    # Generate business models only for repos that make the final list. Several
    # hundred repos match the search, and calling OpenAI for the ones that are
    # cut by MAX_REPOS would repeat the same wasted calls on every run.
    fetched = {repo["id"]: repo for repo in unique_repos}
    for repo_data in select_repos(existing_repos):
        repo = fetched.get(repo_data['id'])
        if repo is None:
            continue

        # New repos, and repos that only have the fallback description from an
        # earlier failed OpenAI call
        stored = repo_data.get('business_model', '')
        if needs_business_model(stored, repo.get("description")):
            full_name = f"{repo_data['owner']}/{repo_data['name']}"
            business_model = stored or repo.get("description") or ""
            if generation_enabled:
                try:
                    business_model = generate_business_model(repo)
                    failures_in_a_row = 0
                except RateLimited as e:
                    # Quota used up: the remaining repos are retried on the next run
                    print(f"Stopping generation for this run at {full_name}: {e}")
                    generation_enabled = False
                except Exception as e:
                    # If the call fails for any other reason, fall back gracefully
                    print(f"Failed to generate business model for {full_name}: {e}")
                    failures_in_a_row += 1
                    if failures_in_a_row >= MAX_GENERATION_FAILURES:
                        # The provider is down or misconfigured: do not keep waiting on it
                        print("Stopping generation for this run after repeated failures")
                        generation_enabled = False
            repo_data['business_model'] = business_model

    print(f"Total unique repos: {len(unique_repos)}")
    
    # 4. Save all repositories (existing + new) to CSV
    save_repos_to_csv(existing_repos)
    
    # 5. Read from CSV and convert to README.md
    convert_csv_to_readme()


if __name__ == "__main__":
    import sys
    max_repos = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(max_repos)