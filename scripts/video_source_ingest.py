from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DERIVED_DIR = DATA_DIR / "derived"

USER_AGENT = "worldcup-2026-ai-assistant/0.1 (+metadata-only research prototype)"
KEYWORDS = [
    "world cup",
    "fifa",
    "training",
    "behind the scenes",
    "highlights",
    "extended highlights",
    "press conference",
    "interview",
    "analysis",
    "tactical",
    "qualifier",
    "friendly",
    "squad",
]

CORE_RELEVANCE_KEYWORDS = [
    "world cup",
    "fifa world cup",
    "2026",
    "qualifier",
    "training",
    "press conference",
    "tactical",
    "analysis",
    "squad",
    "friendly",
    "usmnt",
    "canmnt",
    "three lions",
    "men's national",
    "mens national",
]

EXCLUSION_KEYWORDS = {
    "women": "women_or_girls_football",
    "women's": "women_or_girls_football",
    "womens": "women_or_girls_football",
    "lionesses": "women_or_girls_football",
    "female": "women_or_girls_football",
    "girls": "women_or_girls_football",
    "uwcl": "women_or_girls_football",
    "wsl": "women_or_girls_football",
    "nwsd": "women_or_girls_football",
    "nwsl": "women_or_girls_football",
    "uswnt": "women_or_girls_football",
    "canwnt": "women_or_girls_football",
    "wnt": "women_or_girls_football",
    "u17": "youth_football",
    "u-17": "youth_football",
    "u20": "youth_football",
    "u-20": "youth_football",
    "u21": "youth_football",
    "u-21": "youth_football",
    "u23": "youth_football",
    "u-23": "youth_football",
    "under-17": "youth_football",
    "under 17": "youth_football",
    "under-20": "youth_football",
    "under 20": "youth_football",
    "under-21": "youth_football",
    "under 21": "youth_football",
    "under-23": "youth_football",
    "under 23": "youth_football",
    "futsal": "non_standard_football",
    "beach soccer": "non_standard_football",
    "efootball": "non_standard_football",
    "nfl": "other_sport",
    "football 🏈": "other_sport",
}

SOURCE_REQUIRED_TERMS = {
    "USA": ["usmnt", "men's national", "mens national", "world cup", "qualifier", "friendly"],
    "Canada": ["canmnt", "men's national", "mens national", "world cup", "qualifier", "friendly"],
    "England": ["england men", "men's", "mens", "three lions", "world cup", "qualifier", "friendly"],
    "Media": ["soccer", "fifa", "world cup", "champions league", "premier league", "laliga", "serie a", "bundesliga"],
}


@dataclass
class ChannelResult:
    source_name: str
    platform: str
    team_or_scope: str
    channel_url: str
    channel_id: str
    feed_url: str
    status: str
    notes: str


def fetch_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=45) as response:
        return response.read().decode("utf-8", errors="replace")


def resolve_youtube_channel_id(channel_url: str, handle_or_id: str) -> tuple[str, str]:
    if handle_or_id.startswith("UC"):
        return handle_or_id, "seeded channel id"

    candidates = [
        channel_url,
        f"https://www.youtube.com/{quote(handle_or_id)}" if handle_or_id else "",
    ]
    patterns = [
        r'"channelId":"(UC[a-zA-Z0-9_-]{20,})"',
        r'"externalId":"(UC[a-zA-Z0-9_-]{20,})"',
        r'<meta itemprop="channelId" content="(UC[a-zA-Z0-9_-]{20,})"',
    ]
    for candidate in [item for item in candidates if item]:
        try:
            html = fetch_text(candidate)
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        for pattern in patterns:
            match = re.search(pattern, html)
            if match:
                return match.group(1), "resolved from public channel page"
    return "", locals().get("last_error", "channel id not found")


def parse_youtube_feed(feed_url: str, source_row: pd.Series) -> list[dict[str, str]]:
    try:
        xml_text = fetch_text(feed_url)
    except (HTTPError, URLError, TimeoutError):
        return []

    root = ET.fromstring(xml_text)
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "yt": "http://www.youtube.com/xml/schemas/2015",
        "media": "http://search.yahoo.com/mrss/",
    }

    rows = []
    for entry in root.findall("atom:entry", ns):
        video_id = entry.findtext("yt:videoId", default="", namespaces=ns)
        title = entry.findtext("atom:title", default="", namespaces=ns)
        published = entry.findtext("atom:published", default="", namespaces=ns)
        updated = entry.findtext("atom:updated", default="", namespaces=ns)
        author = entry.findtext("atom:author/atom:name", default="", namespaces=ns)
        link_node = entry.find("atom:link", ns)
        media_group = entry.find("media:group", ns)
        description = ""
        thumbnail_url = ""
        if media_group is not None:
            description = media_group.findtext("media:description", default="", namespaces=ns)
            thumbnail = media_group.find("media:thumbnail", ns)
            if thumbnail is not None:
                thumbnail_url = thumbnail.attrib.get("url", "")
        link = link_node.attrib.get("href", "") if link_node is not None else f"https://www.youtube.com/watch?v={video_id}"
        text = f"{title} {description}".lower()
        matched_keywords = [keyword for keyword in KEYWORDS if keyword in text]
        exclusion_reasons = sorted({reason for keyword, reason in EXCLUSION_KEYWORDS.items() if keyword in text})
        core_match = any(keyword in text for keyword in CORE_RELEVANCE_KEYWORDS)
        required_terms = SOURCE_REQUIRED_TERMS.get(str(source_row["team_or_scope"]), [])
        source_match = not required_terms or any(term in text for term in required_terms)
        is_mens_world_cup_relevant = bool(matched_keywords) and core_match and source_match and not exclusion_reasons

        rows.append(
            {
                "source_name": source_row["source_name"],
                "platform": "YouTube",
                "team_or_scope": source_row["team_or_scope"],
                "video_id": video_id,
                "title": title,
                "published": published,
                "updated": updated,
                "author": author,
                "url": link,
                "embed_url": f"https://www.youtube.com/embed/{video_id}" if video_id else "",
                "thumbnail_url": thumbnail_url,
                "matched_keywords": "|".join(matched_keywords),
                "relevance_score": len(matched_keywords),
                "is_mens_world_cup_relevant": is_mens_world_cup_relevant,
                "exclusion_reason": "|".join(exclusion_reasons),
                "rights_posture": source_row["rights_posture"],
                "analysis_allowed_v1": "metadata_embed_transcript_if_available",
                "download_allowed": "no",
                "retrieved_on": date.today().isoformat(),
            }
        )
    return rows


def build_channel_results(seeds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    channel_results: list[ChannelResult] = []
    video_rows: list[dict[str, str]] = []

    for _, row in seeds.iterrows():
        if row["platform"] != "YouTube":
            channel_results.append(
                ChannelResult(
                    source_name=row["source_name"],
                    platform=row["platform"],
                    team_or_scope=row["team_or_scope"],
                    channel_url=row.get("channel_url", ""),
                    channel_id="",
                    feed_url="",
                    status="manual_or_api_only",
                    notes=row.get("notes", ""),
                )
            )
            continue

        channel_id, note = resolve_youtube_channel_id(str(row.get("channel_url", "")), str(row.get("handle_or_id", "")))
        feed_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}" if channel_id else ""
        status = "resolved" if channel_id else "not_resolved"
        channel_results.append(
            ChannelResult(
                source_name=row["source_name"],
                platform=row["platform"],
                team_or_scope=row["team_or_scope"],
                channel_url=row.get("channel_url", ""),
                channel_id=channel_id,
                feed_url=feed_url,
                status=status,
                notes=note,
            )
        )
        if feed_url:
            video_rows.extend(parse_youtube_feed(feed_url, row))

    return pd.DataFrame([item.__dict__ for item in channel_results]), pd.DataFrame(video_rows)


def write_analysis_design() -> None:
    rows = [
        {
            "stage": "Discovery",
            "input": "official channel URLs, RSS/API metadata, Kaggle dataset catalog",
            "output": "source reliability and rights posture",
            "copyright_safe_design": "store links and metadata, not copied video files",
        },
        {
            "stage": "Metadata filtering",
            "input": "titles descriptions dates teams keywords",
            "output": "ranked relevant clips for each team or match",
            "copyright_safe_design": "use public metadata and link back to platform",
        },
        {
            "stage": "Transcript analysis",
            "input": "captions/transcripts where API or platform allows access",
            "output": "training notes, tactical themes, player mentions",
            "copyright_safe_design": "store short derived facts and timestamps, not full copyrighted transcripts unless licence permits",
        },
        {
            "stage": "Video analysis",
            "input": "licensed clips or open Kaggle video datasets",
            "output": "action recognition, ball/player tracking, event detection",
            "copyright_safe_design": "only download/process video when dataset licence or rights agreement allows it",
        },
        {
            "stage": "App display",
            "input": "video metadata rows and source URLs",
            "output": "embedded or linked source cards with analysis notes",
            "copyright_safe_design": "embed official player where allowed; do not rehost footage",
        },
    ]
    pd.DataFrame(rows).to_csv(DATA_DIR / "video_analysis_design.csv", index=False)


def main() -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    seeds = pd.read_csv(DATA_DIR / "video_channel_seeds.csv").fillna("")
    channel_results, video_metadata = build_channel_results(seeds)

    channel_results.to_csv(DERIVED_DIR / "video_channel_index.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    if video_metadata.empty:
        video_metadata = pd.DataFrame(
            columns=[
                "source_name",
                "platform",
                "team_or_scope",
                "video_id",
                "title",
                "published",
                "updated",
                "author",
                "url",
                "embed_url",
                "thumbnail_url",
                "matched_keywords",
                "relevance_score",
                "is_mens_world_cup_relevant",
                "exclusion_reason",
                "rights_posture",
                "analysis_allowed_v1",
                "download_allowed",
                "retrieved_on",
            ]
        )
    video_metadata.to_csv(DERIVED_DIR / "video_metadata.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    if "is_mens_world_cup_relevant" not in video_metadata:
        video_metadata["is_mens_world_cup_relevant"] = False
    relevant = video_metadata[
        (video_metadata["relevance_score"].astype(int) > 0) & (video_metadata["is_mens_world_cup_relevant"].astype(bool))
    ].sort_values(
        ["relevance_score", "published"], ascending=[False, False]
    )
    relevant.to_csv(DERIVED_DIR / "video_metadata_relevant.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    write_analysis_design()

    summary = pd.DataFrame(
        [
            {"file": "video_channel_seeds.csv", "rows": len(seeds)},
            {"file": "derived/video_channel_index.csv", "rows": len(channel_results)},
            {"file": "derived/video_metadata.csv", "rows": len(video_metadata)},
            {"file": "derived/video_metadata_relevant.csv", "rows": len(relevant)},
            {"file": "kaggle_video_dataset_catalog.csv", "rows": len(pd.read_csv(DATA_DIR / "kaggle_video_dataset_catalog.csv"))},
            {"file": "video_platform_policy.csv", "rows": len(pd.read_csv(DATA_DIR / "video_platform_policy.csv"))},
            {"file": "video_analysis_design.csv", "rows": 5},
        ]
    )
    summary.to_csv(DATA_DIR / "video_ingestion_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
