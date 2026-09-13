from __future__ import annotations

import logging
from datetime import date

import anthropic

from sources.base import Article

logger = logging.getLogger(__name__)


def _build_world_script_prompt(articles: list[Article], today: date) -> str:
    articles_text = []
    for a in articles:
        articles_text.append(
            f"- [{a.score:.1f}] {a.title} ({a.source})\n"
            f"  URL: {a.url}\n"
            f"  Why: {a.score_reason}\n"
            f"  Summary: {a.summary[:200]}"
        )

    return f"""You are writing the script for a 20-minute daily news podcast for {today.strftime('%A, %B %d, %Y')}.
This podcast focuses on US and world news. The audience is an informed citizen who wants comprehensive, balanced coverage.

CRITICAL EDITORIAL GUIDELINES:
- When multiple sources cover the same story differently, explicitly note the disagreements.
  Example: "Al Jazeera frames this as X, while Fox News characterizes it as Y."
- When a story has a clear political angle or spin from a source, call it out directly.
  Example: "The Intercept's reporting emphasizes the civil liberties angle, while Politico focuses on the political calculus."
- Present all sides fairly but don't create false equivalence. If the facts clearly support one interpretation, say so.
- Use specific source names when attributing perspectives — don't say "some sources say."

Here are today's top articles, sorted by relevance score:

{chr(10).join(articles_text)}

Write a podcast script (~3,500 words, targeting 20 minutes at speaking pace) with this structure:

1. **Opening** — Brief greeting and date. "Good morning. This is your US and World News briefing for [date]."

2. **Lead Story** — The single most important story. Deep dive with 5-6 sentences of context, multiple source perspectives, and why it matters. If sources disagree on framing, highlight that.

3. **US News** — 5-7 stories. This is the core section. For each: what happened, why it matters, and note where sources offer competing narratives or political framing. 3-4 sentences each.

4. **World News** — 5-7 international stories. Prioritize conflicts, diplomacy, and events with US implications. Include perspectives from international outlets (Al Jazeera, BBC, France24, etc.) that may differ from US media framing.

5. **Analysis Corner** — Pick 1-2 stories where source disagreement or political framing is most stark. Spend a paragraph breaking down the different angles and what the listener should consider.

6. **Quick Hits** — 4-6 one-sentence mentions of other notable stories.

7. **Closing** — Brief sign-off. "That's your briefing for today. Stay informed, stay critical."

Style rules:
- Conversational but authoritative. Like a thoughtful journalist briefing a colleague.
- Natural transitions between stories and sections.
- No clickbait or breathless hype. Measured, nuanced tone.
- Always attribute reporting to specific outlets: "according to Reuters...", "Al Jazeera reports...", "as Politico notes..."
- When covering US politics, avoid taking sides but DO point out when coverage reveals a clear partisan lens.
- Write for spoken delivery — short sentences, avoid complex clause structures.
- Do NOT include any stage directions, speaker labels, or formatting markers.
- Write as continuous prose organized into clear paragraphs (one per story/topic).
- Use a blank line between paragraphs.

Return ONLY the script text, no markdown formatting or headers."""


def generate_world_script(
    articles: list[Article],
    model: str = "claude-sonnet-4-5-20250929",
    max_stories: int = 150,
    api_key: str | None = None,
) -> str:
    if not articles:
        return "No relevant articles found today. That's your briefing — stay informed, stay critical."

    top = sorted(articles, key=lambda a: a.score, reverse=True)[:max_stories]

    client = anthropic.Anthropic(api_key=api_key, timeout=120.0, max_retries=3)
    prompt = _build_world_script_prompt(top, date.today())

    logger.info(f"Generating world news script from {len(top)} articles with {model}")

    response = client.messages.create(
        model=model,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
    )

    script = response.content[0].text.strip()

    if script.startswith("```"):
        lines = script.split("\n")
        script = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

    word_count = len(script.split())
    logger.info(f"World news script generated ({word_count} words, {len(script)} chars)")
    return script
