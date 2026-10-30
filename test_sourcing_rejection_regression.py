"""Targeted regression test suite for stock footage candidate sourcing and rejected-candidate exclusion.

Tests:
1. Rejection exclusion upfront: Rejected IDs in approval registry must be excluded upfront from search results.
2. Query enrichment hygiene: Descriptive visual prompts (>5 words) must not be prepended as raw Pexels search strings.
3. Query exhaustion: Search pipeline must search all keywords when earlier queries yield only excluded candidates.
4. Broader search fallback: Validates that broader search queries are attempted when specific queries fail.

Run: python test_sourcing_rejection_regression.py
"""

import os
import sys
import uuid
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.abspath("."))

from app.services.providers.pexels_video_provider import PexelsVideoProvider
from app.services.pipeline_orchestrator import VideoPipelineOrchestrator, extract_scene_subject
from app.schemas.media import MediaAssetSchema


class TestSourcingRejectionRegression(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.provider = PexelsVideoProvider(api_key="dummy_pexels_key")
        self.orchestrator = VideoPipelineOrchestrator()

    def test_01_query_enrichment_sentence_contamination(self):
        """Descriptive action prompts (>5 words) must not be concatenated into 20+ word unsearchable queries."""
        sc = {
            "scene_order": 1,
            "scene_text": "Water enters the seed, swelling its cells and activating enzymes.",
            "visual_prompt": "Extreme macro close-up of a dry seed embedded in soil as water saturates the surrounding particles, causing the seed coat to swell and soften visibly.",
            "search_keywords": ["seed swelling water", "macro seed imbibition", "dry seed absorbing water", "seed coat expanding"]
        }
        subj = extract_scene_subject(sc, fallback_topic="How a seed grows into a plant")
        shot_ctx = {
            "topic": "How a seed grows into a plant",
            "scene_order": 1,
            "shot_index": 1,
            "scene_text": sc["scene_text"],
            "visual_prompt": sc["visual_prompt"],
            "required_action": sc["visual_prompt"],
            "primary_subject": subj,
            "required_setting": "soil",
            "visual_state": "",
            "search_keywords": sc["search_keywords"],
        }
        ctx = self.provider._extract_scene_context(shot_ctx, sc["search_keywords"])
        queries = self.provider._enrich_search_queries(sc["search_keywords"], ctx)

        for q in queries:
            word_count = len(q.split())
            self.assertLessEqual(
                word_count, 6,
                f"Search query '{q}' has {word_count} words (>6 words). Long sentence queries cause 0 results in Pexels."
            )

    async def test_02_rejected_candidates_excluded_upfront(self):
        """Candidates listed in used_video_ids/excluded_ids must be skipped before filling candidate quota."""
        mock_api_data = {
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

        with patch.object(self.provider, "_make_api_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_api_data

            # When rejected candidates 38738157 and 36508428 are passed in used_video_ids
            candidates = await self.provider.search_media_candidates(
                keywords=["seed swelling water"],
                target_duration=5.0,
                used_video_ids={38738157, 36508428},
                max_candidates=2
            )

            cand_ids = [c.metadata["pexels_id"] for c in candidates]
            self.assertNotIn(38738157, cand_ids)
            self.assertNotIn(36508428, cand_ids)
            self.assertIn(456789, cand_ids)

    async def test_03_search_continues_when_top_queries_only_have_excluded_candidates(self):
        """If the first query only returns excluded/rejected candidates, search must proceed to subsequent queries."""
        # Query 1 returns only candidate 38738157 (which is rejected)
        data_q1 = {
            "videos": [
                {
                    "id": 38738157,
                    "width": 1920, "height": 1080, "duration": 10.0, "url": "https://pexels.com/video/38738157",
                    "user": {"name": "Test"}, "video_files": [{"link": "http://example.com/1.mp4", "file_type": "video/mp4", "width": 1920, "height": 1080}],
                    "tags": [{"name": "seed"}]
                }
            ]
        }
        # Query 2 returns fresh candidate 999111
        data_q2 = {
            "videos": [
                {
                    "id": 999111,
                    "width": 1920, "height": 1080, "duration": 10.0, "url": "https://pexels.com/video/999111",
                    "user": {"name": "Test"}, "video_files": [{"link": "http://example.com/fresh.mp4", "file_type": "video/mp4", "width": 1920, "height": 1080}],
                    "tags": [{"name": "seed"}, {"name": "germination"}]
                }
            ]
        }

        async def fake_request(client, url, params):
            q = params.get("query")
            if "seed swelling" in q:
                return data_q1
            return data_q2

        with patch.object(self.provider, "_make_api_request", side_effect=fake_request):
            candidates = await self.provider.search_media_candidates(
                keywords=["seed swelling water", "seed germination time lapse"],
                target_duration=5.0,
                used_video_ids={38738157},
                max_candidates=1
            )

            cand_ids = [c.metadata["pexels_id"] for c in candidates]
            self.assertIn(999111, cand_ids)

    async def test_04_orchestrator_passes_rejected_ids_to_search_provider(self):
        """Orchestrator must pass rejected candidate IDs to search_media_candidates to prevent re-fetching."""
        proj_id = uuid.uuid4()
        # Mock approval registry with rejected candidates for scene 1 shot 1
        with patch.object(self.orchestrator, "load_approval_registry") as mock_load_reg:
            mock_load_reg.return_value = {
                "project_id": str(proj_id),
                "approved_candidates": {},
                "rejected_candidates": {
                    "scene_1_shot_1": ["38738157", "36508428"]
                }
            }

            mock_provider = AsyncMock()
            mock_provider.search_media_candidates.return_value = []
            mock_provider.download_media.return_value = "/tmp/dummy.mp4"

            with patch.object(self.orchestrator, "get_stock_media_provider", return_value=mock_provider):
                with patch.object(self.orchestrator, "_save_scene_asset", new_callable=AsyncMock):
                    with patch.object(self.orchestrator, "_record_pipeline_job", new_callable=AsyncMock):
                        with patch.object(self.orchestrator, "_update_project_state", new_callable=AsyncMock):
                            scenes = [{
                                "id": uuid.uuid4(),
                                "scene_order": 1,
                                "duration": 5.0,
                                "scene_text": "Water enters seed.",
                                "search_keywords": ["seed swelling water"],
                                "shot_plan": [{
                                    "shot_index": 1,
                                    "provisional_start_seconds": 0.0,
                                    "provisional_end_seconds": 5.0,
                                    "required_visible_action": "Water absorption",
                                    "primary_subject": "dry seed",
                                    "required_setting": "soil",
                                    "search_keywords": ["seed swelling water", "seed in soil"]
                                }]
                            }]
                            try:
                                await self.orchestrator._stage_5_source_media(proj_id, scenes, "/tmp/clips", topic="Seeds")
                            except Exception:
                                pass

                            # Check what used_video_ids was passed to search_media_candidates
                            self.assertTrue(mock_provider.search_media_candidates.called)
                            passed_used_ids = mock_provider.search_media_candidates.call_args[1].get("used_video_ids", set())
                            passed_used_ids_str = {str(x) for x in passed_used_ids}
                            self.assertIn("38738157", passed_used_ids_str, "Orchestrator failed to pass rejected candidate 38738157 in used_video_ids")
                            self.assertIn("36508428", passed_used_ids_str, "Orchestrator failed to pass rejected candidate 36508428 in used_video_ids")

    def test_05_botanical_fallback_queries_present(self):
        """Natural-language botanical fallback queries must be appended when botanical context is present."""
        ctx = {"primary_subject": "dry seed", "primary_action": "absorbing water", "setting": "soil", "visual_state": ""}
        keywords = ["seed swelling water", "macro seed imbibition"]
        queries = self.provider._enrich_search_queries(keywords, ctx)

        expected_botanical = [
            "seed germination time lapse",
            "watering seed in soil",
            "germinating seed close up",
            "seed sprouting time lapse"
        ]
        for eb in expected_botanical:
            self.assertIn(eb, queries, f"Expected botanical fallback query '{eb}' missing from enriched queries")

    async def test_06_pagination_bounds_and_safety_limit(self):
        """Pagination must be bounded to at most page 2 per query, and total API calls capped at max 8."""
        requested_pages = []

        async def track_request(client, url, params):
            requested_pages.append((params.get("query"), params.get("page")))
            # Return 15 dummy videos to trigger page 2 check
            return {
                "videos": [
                    {
                        "id": 1000 + i + params.get("page", 1) * 100,
                        "width": 1920, "height": 1080, "duration": 10.0,
                        "url": f"https://pexels.com/video/{1000 + i}",
                        "user": {"name": "Test"},
                        "video_files": [{"link": "http://example.com/v.mp4", "file_type": "video/mp4", "width": 1920, "height": 1080}],
                        # Deliberately missing tags/metadata to prevent qualifying candidates
                        "tags": []
                    }
                    for i in range(15)
                ]
            }

        with patch.object(self.provider, "_make_api_request", side_effect=track_request):
            with patch.object(self.provider, "_evaluate_video", return_value=None):
                from app.services.providers.exceptions import NoSuitableMediaFoundError
                with self.assertRaises(NoSuitableMediaFoundError):
                    await self.provider.search_media_candidates(
                        keywords=["seed swelling water", "seed in soil", "sprout timelapse", "seed imbibition", "seed germination"],
                        target_duration=5.0,
                        max_candidates=3
                    )

        # Confirm no page > 2 was ever requested
        for q, p in requested_pages:
            self.assertIn(p, (1, 2), f"Requested page {p} exceeds bounded pagination limit of 2")

        # Confirm total API calls did not exceed 8
        self.assertLessEqual(len(requested_pages), 8, f"Total API requests ({len(requested_pages)}) exceeded safety bound of 8")

    async def test_07_duplicate_candidate_elimination(self):
        """A video returned across multiple search queries must only be evaluated and collected once."""
        duplicate_video = {
            "id": 888777,
            "width": 1920, "height": 1080, "duration": 8.0,
            "url": "https://pexels.com/video/888777",
            "user": {"name": "Botanist"},
            "video_files": [{"link": "http://example.com/888777.mp4", "file_type": "video/mp4", "width": 1920, "height": 1080}],
            "tags": [{"name": "seed"}, {"name": "germination"}, {"name": "soil"}]
        }

        fresh_video = {
            "id": 888778,
            "width": 1920, "height": 1080, "duration": 8.0,
            "url": "https://pexels.com/video/888778",
            "user": {"name": "Botanist"},
            "video_files": [{"link": "http://example.com/888778.mp4", "file_type": "video/mp4", "width": 1920, "height": 1080}],
            "tags": [{"name": "seed"}, {"name": "germination"}, {"name": "soil"}]
        }

        async def fake_response(client, url, params):
            # Both queries return duplicate_video, query 2 also has fresh_video
            if "germination" in params.get("query"):
                return {"videos": [duplicate_video]}
            return {"videos": [duplicate_video, fresh_video]}

        with patch.object(self.provider, "_make_api_request", side_effect=fake_response):
            candidates = await self.provider.search_media_candidates(
                keywords=["seed germination", "soil sprout"],
                target_duration=5.0,
                max_candidates=3
            )

        cand_ids = [c.metadata["pexels_id"] for c in candidates]
        # Video 888777 should only appear once
        self.assertEqual(cand_ids.count(888777), 1, "Duplicate candidate was included more than once")
        self.assertIn(888778, cand_ids)

    async def test_08_no_suitable_candidates_exist_raises_error(self):
        """When no suitable candidates exist (all excluded or failing criteria), raises NoSuitableMediaFoundError."""
        from app.services.providers.exceptions import NoSuitableMediaFoundError

        async def empty_response(client, url, params):
            return {"videos": []}

        with patch.object(self.provider, "_make_api_request", side_effect=empty_response):
            with self.assertRaises(NoSuitableMediaFoundError):
                await self.provider.search_media_candidates(
                    keywords=["unobtainable_scientific_process_keyword"],
                    target_duration=5.0,
                    max_candidates=3
                )

    async def test_09_defense_in_depth_rejection_check_in_orchestrator(self):
        """Defense in depth: If provider returns a rejected ID, orchestrator must still skip it."""
        proj_id = uuid.uuid4()
        with patch.object(self.orchestrator, "load_approval_registry") as mock_load_reg:
            mock_load_reg.return_value = {
                "project_id": str(proj_id),
                "approved_candidates": {},
                "rejected_candidates": {
                    "scene_1_shot_1": ["38738157"]
                }
            }

            rejected_cand = MediaAssetSchema(
                asset_type="video",
                provider="pexels",
                source_url="http://example.com/38738157",
                download_url="http://example.com/38738157.mp4",
                metadata={"pexels_id": 38738157, "width": 1920, "height": 1080, "duration": 5.0}
            )

            mock_provider = AsyncMock()
            # Provider returns the rejected candidate despite exclusion
            mock_provider.search_media_candidates.return_value = [rejected_cand]

            with patch.object(self.orchestrator, "get_stock_media_provider", return_value=mock_provider):
                with patch.object(self.orchestrator, "_save_scene_asset", new_callable=AsyncMock) as mock_save:
                    scenes = [{
                        "id": uuid.uuid4(),
                        "scene_order": 1,
                        "duration": 5.0,
                        "scene_text": "Water enters seed.",
                        "search_keywords": ["seed swelling water"],
                        "shot_plan": [{
                            "shot_index": 1,
                            "provisional_start_seconds": 0.0,
                            "provisional_end_seconds": 5.0,
                            "required_visible_action": "Water absorption",
                            "primary_subject": "dry seed",
                            "required_setting": "soil",
                            "search_keywords": ["seed swelling water"]
                        }]
                    }]
                    try:
                        await self.orchestrator._stage_5_source_media(proj_id, scenes, "/tmp/clips", topic="Seeds")
                    except Exception:
                        pass

                    # Ensure the rejected candidate was never saved as an asset
                    self.assertFalse(mock_save.called, "Rejected candidate was saved as scene asset instead of being skipped")


if __name__ == "__main__":
    unittest.main()
