"""Investigation test script for Scene 1 Shot 1 sourcing failure.
Simulates:
1. Candidate search with rejected candidates present in approval registry.
2. Demonstrates why passing used_video_ids without rejected candidates returns the rejected candidates.
3. Demonstrates why long sentence in _enrich_search_queries breaks query 1.
4. Demonstrates how excluding rejected IDs and searching broader queries finds replacement candidates.
"""

import os, sys
sys.path.insert(0, os.path.abspath("."))
from app.services.providers.pexels_video_provider import PexelsVideoProvider
from app.services.pipeline_orchestrator import extract_scene_subject

def test_query_enrichment_bug():
    sc = {
        'scene_order': 1,
        'scene_text': 'Water enters the seed, swelling its cells and activating enzymes.',
        'visual_prompt': 'Extreme macro close-up of a dry seed embedded in soil as water saturates the surrounding particles, causing the seed coat to swell and soften visibly.',
        'search_keywords': ['seed swelling water', 'macro seed imbibition', 'dry seed absorbing water', 'seed coat expanding']
    }
    topic = 'How a seed grows into a plant'
    subj = extract_scene_subject(sc, fallback_topic=topic)
    shot_ctx = {
        'topic': topic,
        'scene_order': 1,
        'shot_index': 1,
        'scene_text': sc['scene_text'],
        'visual_prompt': sc['visual_prompt'],
        'required_action': sc['visual_prompt'],
        'primary_subject': subj,
        'required_setting': '',
        'visual_state': '',
        'camera_framing': '',
        'required_objects': [],
        'explicit_exclusions': [],
        'action_timeline': [],
        'search_keywords': sc['search_keywords'],
    }

    prov = PexelsVideoProvider(api_key='dummy')
    ctx = prov._extract_scene_context(shot_ctx, sc['search_keywords'])
    queries = prov._enrich_search_queries(sc['search_keywords'], ctx)
    print("Queries generated:")
    for i, q in enumerate(queries):
        print(f"  {i+1}: {q}")
    
    # Query 1 is a long sentence
    q1_words = len(queries[0].split())
    print(f"Query 1 word count: {q1_words}")
    assert q1_words > 15, f"Expected long sentence bug in query 1, got {q1_words} words"
    print("CONFIRMED: Query 1 is contaminated with full visual_prompt sentence.")

import unittest
from unittest.mock import AsyncMock, patch

async def test_rejected_candidates_not_excluded():
    # Simulate Pexels API returning 38738157, 36508428, and a new candidate 456789
    prov = PexelsVideoProvider(api_key='dummy')
    
    mock_data = {
        "videos": [
            {
                "id": 38738157,
                "width": 1920, "height": 1080, "duration": 10.0, "url": "https://pexels.com/video/38738157",
                "user": {"name": "Test"}, "video_files": [{"link": "http://example.com/1.mp4", "file_type": "video/mp4", "width": 1920, "height": 1080}],
                "tags": [{"name": "seed"}, {"name": "swelling"}, {"name": "water"}]
            },
            {
                "id": 36508428,
                "width": 1920, "height": 1080, "duration": 10.0, "url": "https://pexels.com/video/36508428",
                "user": {"name": "Test"}, "video_files": [{"link": "http://example.com/2.mp4", "file_type": "video/mp4", "width": 1920, "height": 1080}],
                "tags": [{"name": "seed"}, {"name": "water"}]
            },
            {
                "id": 456789,
                "width": 1920, "height": 1080, "duration": 10.0, "url": "https://pexels.com/video/456789",
                "user": {"name": "Test"}, "video_files": [{"link": "http://example.com/3.mp4", "file_type": "video/mp4", "width": 1920, "height": 1080}],
                "tags": [{"name": "seed"}, {"name": "germination"}, {"name": "water"}]
            }
        ]
    }

    with patch.object(prov, "_make_api_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_data
        
        # Current behavior: used_video_ids is empty, rejected candidates are NOT in used_video_ids
        cands_without_rejection = await prov.search_media_candidates(
            ["seed swelling water"],
            target_duration=5.0,
            used_video_ids=set(),
            max_candidates=2
        )
        cand_ids = [c.metadata["pexels_id"] for c in cands_without_rejection]
        print(f"Candidates returned when rejected IDs not excluded (max 2): {cand_ids}")
        assert 38738157 in cand_ids and 36508428 in cand_ids
        print("CONFIRMED: Pexels provider wastes candidate slots on rejected clips if not excluded upfront.")

        # Fixed behavior: used_video_ids includes rejected IDs {38738157, 36508428}
        cands_with_rejection = await prov.search_media_candidates(
            ["seed swelling water"],
            target_duration=5.0,
            used_video_ids={38738157, 36508428},
            max_candidates=2
        )
        cand_ids_fixed = [c.metadata["pexels_id"] for c in cands_with_rejection]
        print(f"Candidates returned when rejected IDs ARE excluded: {cand_ids_fixed}")
        assert 456789 in cand_ids_fixed
        assert 38738157 not in cand_ids_fixed and 36508428 not in cand_ids_fixed
        print("CONFIRMED: Excluding rejected IDs upfront allows replacement candidates (e.g. 456789) to be found!")

import asyncio
if __name__ == "__main__":
    test_query_enrichment_bug()
    asyncio.run(test_rejected_candidates_not_excluded())
    print("All investigation tests passed.")
