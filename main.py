#!/usr/bin/env python3
"""Daily Podcast — fetch news, generate a podcast, and email a digest."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from utils.config import load_config
from sources.base import Article
from sources.registry import build_fetchers
from pipeline.normalize import normalize_articles
from pipeline.dedup import deduplicate
from pipeline.scorer import score_articles
from pipeline.script import generate_script
from pipeline.world_script import generate_world_script
from pipeline.digest import generate_digest
from audio.tts import generate_audio
from delivery.email import send_digest
from storage.github_release import upload_to_github_release

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("podcast")


def main():
    parser = argparse.ArgumentParser(description="Daily Podcast")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate podcast and digest locally without uploading or emailing",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # 1. Fetch from all sources
    logger.info("=== Fetching articles ===")
    fetchers = build_fetchers(config["sources"])
    all_articles = []
    for fetcher in fetchers:
        try:
            articles = fetcher.fetch()
            all_articles.extend(articles)
        except Exception as e:
            logger.error(f"Fetcher {fetcher.name} failed: {e}")
    logger.info(f"Total fetched: {len(all_articles)} articles")

    if not all_articles:
        logger.warning("No articles fetched. Check your network and source configs.")
        sys.exit(1)

    # 2. Normalize
    logger.info("=== Normalizing ===")
    all_articles = normalize_articles(all_articles)

    # 3. Deduplicate
    logger.info("=== Deduplicating ===")
    all_articles = deduplicate(all_articles)
    logger.info(f"After dedup: {len(all_articles)} articles")

    # 4. Score with Claude
    logger.info("=== Scoring ===")
    all_articles = score_articles(
        all_articles,
        topics=config.get("topics", {}),
        model=config.get("scoring", {}).get("model", "claude-sonnet-4-5-20250929"),
        api_key=config.get("anthropic_api_key"),
    )

    # Filter by minimum score
    min_score = config.get("scoring", {}).get("min_score", 4.0)
    scored = [a for a in all_articles if a.score >= min_score]
    scored.sort(key=lambda a: a.score, reverse=True)
    logger.info(f"Articles above {min_score}: {len(scored)}")

    if not scored:
        logger.warning("No articles passed the relevance threshold.")
        sys.exit(0)

    # 5. Generate podcast script
    logger.info("=== Generating podcast script ===")
    digest_config = config.get("digest", {})
    script = generate_script(
        scored,
        model=digest_config.get("model", "claude-sonnet-4-5-20250929"),
        max_stories=digest_config.get("max_stories", 150),
        api_key=config.get("anthropic_api_key"),
    )

    # Save script to file
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    today = date.today().isoformat()
    script_path = output_dir / f"script-{today}.txt"
    script_path.write_text(script)
    logger.info(f"Script saved to {script_path}")

    # 6. Generate audio via OpenAI TTS
    logger.info("=== Generating audio ===")
    tts_config = config.get("tts", {})
    mp3_path = output_dir / f"podcast-{today}.mp3"
    generate_audio(
        script=script,
        output_path=mp3_path,
        model=tts_config.get("model", "tts-1-hd"),
        voice=tts_config.get("voice", "nova"),
        api_key=config.get("openai_api_key"),
    )

    # 6b. Generate world news podcast
    world_mp3_path = None
    world_cfg = config.get("world_news_podcast", {})
    if world_cfg:
        logger.info("=== Scoring articles for world news podcast ===")
        world_topics = world_cfg.get("topics", {})
        world_scored = score_articles(
            # Re-score a copy of all articles with world news topics
            [Article(title=a.title, url=a.url, source=a.source,
                     published=a.published, summary=a.summary)
             for a in all_articles],
            topics=world_topics,
            model=config.get("scoring", {}).get("model", "claude-sonnet-4-5-20250929"),
            api_key=config.get("anthropic_api_key"),
        )
        world_min = world_cfg.get("min_score", 3.0)
        world_scored = [a for a in world_scored if a.score >= world_min]
        world_scored.sort(key=lambda a: a.score, reverse=True)
        logger.info(f"World news articles above {world_min}: {len(world_scored)}")

        if world_scored:
            logger.info("=== Generating world news script ===")
            world_script = generate_world_script(
                world_scored,
                model=world_cfg.get("model", "claude-sonnet-4-5-20250929"),
                max_stories=world_cfg.get("max_stories", 150),
                api_key=config.get("anthropic_api_key"),
            )
            world_script_path = output_dir / f"world-script-{today}.txt"
            world_script_path.write_text(world_script)
            logger.info(f"World news script saved to {world_script_path}")

            logger.info("=== Generating world news audio ===")
            world_mp3_path = output_dir / f"world-news-{today}.mp3"
            generate_audio(
                script=world_script,
                output_path=world_mp3_path,
                model=tts_config.get("model", "tts-1-hd"),
                voice=tts_config.get("voice", "nova"),
                api_key=config.get("openai_api_key"),
            )

    # 7. Upload MP3s to GitHub Release
    mp3_url = None
    if not args.dry_run:
        github_cfg = config.get("github", {})
        github_token = config.get("github_token", "")
        if github_cfg.get("owner") and github_cfg.get("repo") and github_token:
            logger.info("=== Uploading MP3s to GitHub Release ===")
            mp3_url = upload_to_github_release(
                mp3_path,
                {
                    "owner": github_cfg["owner"],
                    "repo": github_cfg["repo"],
                    "token": github_token,
                },
                world_mp3_path=world_mp3_path,
            )
        else:
            logger.warning("GitHub config incomplete — skipping MP3 upload")

    # 8. Generate email digest
    logger.info("=== Generating email digest ===")
    html = generate_digest(
        scored,
        model=digest_config.get("model", "claude-sonnet-4-5-20250929"),
        max_stories=digest_config.get("max_stories", 150),
        api_key=config.get("anthropic_api_key"),
        mp3_url=mp3_url,
    )

    # 9. Deliver
    if args.dry_run:
        digest_path = output_dir / f"digest-{today}.html"
        digest_path.write_text(html)
        logger.info(f"Dry run complete!")
        print(f"\nScript:  {script_path.absolute()}")
        print(f"Audio:   {mp3_path.absolute()}")
        if world_mp3_path:
            print(f"World:   {world_mp3_path.absolute()}")
        print(f"Digest:  {digest_path.absolute()}")
        print(f"Open digest: file://{digest_path.absolute()}")
    else:
        logger.info("=== Sending email ===")
        success = send_digest(
            html=html,
            email_config=config.get("email", {}),
            api_key=config.get("resend_api_key", ""),
            mp3_url=mp3_url,
        )
        if success:
            logger.info("Digest sent successfully!")
            print(f"\nPodcast: {mp3_path.absolute()}")
            if mp3_url:
                print(f"MP3 URL: {mp3_url}")
            print("Email sent!")
        else:
            logger.error("Failed to send digest email.")
            sys.exit(1)


if __name__ == "__main__":
    main()
