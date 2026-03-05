from __future__ import annotations

import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

API_BASE = "https://api.github.com"
UPLOAD_BASE = "https://uploads.github.com"
TAG = "latest"
ASSET_NAME = "podcast.mp3"


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _get_default_branch(
    client: httpx.Client, owner: str, repo: str, token: str
) -> str:
    """Get the default branch name for the repo."""
    resp = client.get(f"{API_BASE}/repos/{owner}/{repo}", headers=_headers(token))
    resp.raise_for_status()
    return resp.json()["default_branch"]


def _ensure_repo_initialized(
    client: httpx.Client, owner: str, repo: str, token: str, branch: str
) -> None:
    """Create an initial commit if the repo is empty (size == 0)."""
    resp = client.get(f"{API_BASE}/repos/{owner}/{repo}", headers=_headers(token))
    resp.raise_for_status()
    if resp.json().get("size", 0) > 0:
        return

    logger.info("Repo is empty — creating initial commit")
    resp = client.put(
        f"{API_BASE}/repos/{owner}/{repo}/contents/README.md",
        headers=_headers(token),
        json={
            "message": "Initial commit",
            "content": "IyBEYWlseSBQb2RjYXN0Cg==",  # base64("# Daily Podcast\n")
            "branch": branch,
        },
    )
    resp.raise_for_status()
    logger.info("Created initial commit with README.md")


def _get_or_create_release(
    client: httpx.Client, owner: str, repo: str, token: str
) -> int:
    """Get the 'latest' release, or create it if it doesn't exist. Returns release ID."""
    url = f"{API_BASE}/repos/{owner}/{repo}/releases/tags/{TAG}"
    resp = client.get(url, headers=_headers(token))

    if resp.status_code == 200:
        release_id = resp.json()["id"]
        logger.info(f"Found existing release (id={release_id})")
        return release_id

    # Ensure there's at least one commit so the tag can be created
    branch = _get_default_branch(client, owner, repo, token)
    _ensure_repo_initialized(client, owner, repo, token, branch)

    # Create the release
    resp = client.post(
        f"{API_BASE}/repos/{owner}/{repo}/releases",
        headers=_headers(token),
        json={
            "tag_name": TAG,
            "target_commitish": branch,
            "name": "Daily Podcast",
            "body": "Latest daily podcast episode. This release is automatically updated each day.",
            "draft": False,
            "prerelease": False,
        },
    )
    resp.raise_for_status()
    release_id = resp.json()["id"]
    logger.info(f"Created new release (id={release_id})")
    return release_id


def _delete_existing_assets(
    client: httpx.Client, owner: str, repo: str, release_id: int, token: str
) -> None:
    """Delete all existing assets on the release."""
    url = f"{API_BASE}/repos/{owner}/{repo}/releases/{release_id}/assets"
    resp = client.get(url, headers=_headers(token))
    resp.raise_for_status()

    for asset in resp.json():
        asset_url = f"{API_BASE}/repos/{owner}/{repo}/releases/assets/{asset['id']}"
        del_resp = client.delete(asset_url, headers=_headers(token))
        del_resp.raise_for_status()
        logger.info(f"Deleted old asset: {asset['name']}")


def upload_to_github_release(mp3_path: Path, github_config: dict) -> str:
    """Upload an MP3 to a GitHub Release tagged 'latest'.

    Deletes any existing assets first so the release always has only today's file.

    Args:
        mp3_path: Path to the MP3 file.
        github_config: Dict with keys: owner, repo, token.

    Returns:
        Public download URL for the uploaded asset.
    """
    owner = github_config["owner"]
    repo = github_config["repo"]
    token = github_config["token"]

    with httpx.Client(timeout=300) as client:
        release_id = _get_or_create_release(client, owner, repo, token)
        _delete_existing_assets(client, owner, repo, release_id, token)

        # Upload the new asset
        upload_url = (
            f"{UPLOAD_BASE}/repos/{owner}/{repo}/releases/{release_id}/assets"
            f"?name={ASSET_NAME}"
        )
        mp3_bytes = mp3_path.read_bytes()
        size_mb = len(mp3_bytes) / (1024 * 1024)
        logger.info(f"Uploading {ASSET_NAME} ({size_mb:.1f} MB) to GitHub Release")

        headers = {
            **_headers(token),
            "Content-Type": "application/octet-stream",
        }
        resp = client.post(upload_url, headers=headers, content=mp3_bytes)
        resp.raise_for_status()

        player_url = f"https://{owner}.github.io/{repo}/"
        download_url = f"https://github.com/{owner}/{repo}/releases/download/{TAG}/{ASSET_NAME}"
        logger.info(f"Uploaded to GitHub Release: {download_url}")
        logger.info(f"Player URL: {player_url}")
        return player_url
