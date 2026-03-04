import os
import sys
import random
import uuid
from datetime import datetime, timedelta, timezone

# Add the root directory to sys.path so we can import src and database
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database.database import Database
from src.schemas.inference import (
    AnalyzerResult,
    ABSAOutput,
    SpeakerType,
    SentimentLabel,
    InferenceMetadata,
)


def random_date(start, end):
    return start + timedelta(
        seconds=random.randint(0, int((end - start).total_seconds()))
    )


def main():
    db_path = os.path.join(
        os.path.dirname(__file__), "..", "database", "personalens.db"
    )

    start_date = datetime(2026, 2, 1, tzinfo=timezone.utc)
    end_date = datetime(2026, 3, 4, tzinfo=timezone.utc)

    publishers = [
        "Thai Rath",
        "Matichon",
        "Khaosod",
        "Bangkok Post",
        "The Nation",
        "Channel 3",
        "Thai PBS",
        "Sanook",
        "Thairath Online",
        "Prachatai",
    ]

    langs = ["th"] * 45 + ["en"] * 40 + ["ja"] * 5 + ["zh"] * 5 + ["ko"] * 5

    reasons = [
        "The language used strongly suggests a positive alignment with the core values discussed.",
        "There is a clear negative undertone regarding the recent policy announcements.",
        "The statement presents a balanced view without taking a strong stance either way.",
        "Mixed feelings are expressed, highlighting both successes and significant failures.",
        "The overall sentiment reflects optimism about future developments.",
        "Criticism is evident throughout the narrative regarding the entity's actions.",
    ]

    per_names = [
        "John Doe",
        "Jane Smith",
        "Thaksin Shinawatra",
        "Prayut Chan-o-cha",
        "Anutin Charnvirakul",
    ]
    org_names = [
        "Pheu Thai Party",
        "Move Forward Party",
        "United Thai Nation",
        "Democrat Party",
        "Bhumjaithai Party",
    ]
    loc_names = [
        "Chao Phraya River",
        "Doi Inthanon",
        "Sukhumvit Road",
        "Lumpini Park",
        "Mekong River",
    ]
    gpe_names = ["Bangkok", "Chiang Mai", "Phuket", "Thailand", "Japan"]

    entities = []
    for name in per_names:
        entities.append({"name": name, "surface": name.split()[0], "type": "PER"})
    for name in org_names:
        entities.append({"name": name, "surface": name, "type": "ORG"})
    for name in loc_names:
        entities.append({"name": name, "surface": name, "type": "LOC"})
    for name in gpe_names:
        entities.append({"name": name, "surface": name, "type": "GPE"})

    # Pre-generate random times for our records so we can patch the db
    print("Generating mock data...")

    with Database(db_path) as db:
        # Generate 500 mock entries
        for i in range(500):
            # Patch time
            dt = random_date(start_date, end_date)
            Database._now = staticmethod(lambda: dt.isoformat())

            lang = random.choice(langs)
            pub = random.choice(publishers)
            ent = random.choice(entities)
            sentiment_val = random.choice(list(SentimentLabel))

            ent_id_str = db.upsert_entity(
                canonical_name=ent["name"],
                category=ent["type"],
                lang=lang,
            )
            ent_uuid = uuid.UUID(ent_id_str)

            absa_out = ABSAOutput(
                speaker_type=SpeakerType.REPORTER,
                speaker_name=None,
                is_aimed_at_target=True,
                targeting_keywords=["mock"],
                sentiment=sentiment_val,
                aspects=["general"],
                rationale=random.choice(reasons),
            )

            meta = InferenceMetadata(
                prompt_id="mock-prompt-v1",
                model="mock-slm",
                duration_ms=random.randint(100, 500),
            )

            result = AnalyzerResult(
                surface_form=ent["surface"],
                canonical_name=ent["name"],
                global_id=ent_uuid,  # enforce assigned category
                context_window=f"This is a mock sentence in {lang} mentioning {ent['surface']}.",
                absa=absa_out,
                metadata=meta,
            )

            db.save_analyzer_result(
                result=result,
                source_url=f"https://mocknews.local/articles/{uuid.uuid4()}",
                headline=f"Mock headline {i}",
                publisher=pub,
                lang=lang,
                published_at=dt.isoformat(),
                is_headline=random.choice([True, False]),
                confidence_score=round(random.uniform(0.0, 1.0), 2),
            )

    print("Mock data generated successfully in the database!")


if __name__ == "__main__":
    main()
