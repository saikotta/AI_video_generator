import sys
import json
import inspect
import asyncio
import unittest
from unittest.mock import AsyncMock, patch, MagicMock

import httpx

sys.path.insert(0, r"C:\AiVideoGenerator")

from app.services.providers.groq_script_provider import GroqScriptProvider
from app.services.providers.exceptions import (
    ScriptProviderError,
    ScriptGenerationTimeoutError,
    InvalidScriptResponseError
)


class TestStep11D1GroqJsonRecovery(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.provider = GroqScriptProvider(
            api_key="gsk_mocked_test_key_1234567890abcdef",
            model="openai/gpt-oss-120b",
            base_delay=0.01,
            timeout_seconds=5.0
        )

    # 1. Successful JSON-mode response
    async def test_01_successful_json_mode_response(self):
        mock_resp = httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"content": '{"scenes": [{"scene_order": 1, "scene_text": "Hello world"}]}'}}]},
            request=httpx.Request("POST", "https://api.groq.com")
        )
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp

        with patch.object(self.provider, "_get_client", return_value=mock_client):
            res = await self.provider._post_groq("test prompt", is_json=True, operation="test_op")
            self.assertIn("Hello world", res)
            self.assertFalse(self.provider.last_json_recovery_attempted)
            self.assertFalse(self.provider.last_json_recovery_failed)
            self.assertEqual(mock_client.post.call_count, 1)
            # Verify response_format was present
            call_payload = mock_client.post.call_args[1]["json"]
            self.assertEqual(call_payload["response_format"], {"type": "json_object"})

    # 2. HTTP 400 json_validate_failed followed by valid JSON fallback
    async def test_02_http400_json_validate_failed_recovery_success(self):
        err_resp = httpx.Response(
            status_code=400,
            json={"error": {"code": "json_validate_failed", "message": "Failed to validate JSON", "type": "invalid_request_error"}},
            request=httpx.Request("POST", "https://api.groq.com")
        )
        rec_resp = httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"content": '{"scenes": [{"scene_order": 1, "scene_text": "Recovered scene"}]}'}}]},
            request=httpx.Request("POST", "https://api.groq.com")
        )
        mock_client = AsyncMock()
        mock_client.post.side_effect = [err_resp, rec_resp]

        with patch.object(self.provider, "_get_client", return_value=mock_client):
            res = await self.provider._post_groq("test prompt", is_json=True, operation="test_recovery")
            self.assertIn("Recovered scene", res)
            self.assertTrue(self.provider.last_json_recovery_attempted)
            self.assertFalse(self.provider.last_json_recovery_failed)
            self.assertEqual(mock_client.post.call_count, 2)

            # First call had response_format
            first_payload = mock_client.post.call_args_list[0][1]["json"]
            self.assertIn("response_format", first_payload)

            # Recovery call omitted response_format
            rec_payload = mock_client.post.call_args_list[1][1]["json"]
            self.assertNotIn("response_format", rec_payload)
            self.assertEqual(rec_payload["model"], "openai/gpt-oss-120b")

    # 3. HTTP 400 json_validate_failed followed by malformed JSON
    async def test_03_http400_recovery_malformed_json_caught_by_parser(self):
        err_resp = httpx.Response(
            status_code=400,
            json={"error": {"code": "json_validate_failed", "message": "Failed to validate JSON"}},
            request=httpx.Request("POST", "https://api.groq.com")
        )
        rec_resp = httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"content": "This is plain conversational text without any JSON."}}]},
            request=httpx.Request("POST", "https://api.groq.com")
        )
        mock_client = AsyncMock()
        mock_client.post.side_effect = [err_resp, rec_resp]

        with patch.object(self.provider, "_get_client", return_value=mock_client):
            # split_into_scenes calls _post_groq then _extract_json_payload
            with self.assertRaises(InvalidScriptResponseError) as ctx:
                await self.provider.split_into_scenes("Hello world this is a test script.", 20)
            self.assertIn("Groq returned malformed JSON", str(ctx.exception))
            self.assertTrue(self.provider.last_json_recovery_attempted)

    # 4. HTTP 400 with unrelated error code (e.g. context_length_exceeded)
    async def test_04_http400_unrelated_error_code_does_not_recover(self):
        err_resp = httpx.Response(
            status_code=400,
            json={"error": {"code": "context_length_exceeded", "message": "Prompt is too long"}},
            request=httpx.Request("POST", "https://api.groq.com")
        )
        mock_client = AsyncMock()
        mock_client.post.return_value = err_resp

        with patch.object(self.provider, "_get_client", return_value=mock_client):
            with self.assertRaises(ScriptProviderError) as ctx:
                await self.provider._post_groq("test prompt", is_json=True, operation="unrelated_test")
            self.assertIn("context_length_exceeded", str(ctx.exception))
            self.assertIn("json_recovery_attempted: False", str(ctx.exception))
            self.assertFalse(self.provider.last_json_recovery_attempted)
            # Must NOT have made a second recovery call
            self.assertEqual(mock_client.post.call_count, 1)

    # 5. HTTP 401 authentication error
    async def test_05_http401_auth_error_no_recovery(self):
        err_resp = httpx.Response(
            status_code=401,
            json={"error": {"code": "invalid_api_key", "message": "Invalid API key provided"}},
            request=httpx.Request("POST", "https://api.groq.com")
        )
        mock_client = AsyncMock()
        mock_client.post.return_value = err_resp

        with patch.object(self.provider, "_get_client", return_value=mock_client):
            with self.assertRaises(ScriptProviderError) as ctx:
                await self.provider._post_groq("test prompt", is_json=True, operation="auth_op")
            self.assertIn("HTTP error status 401", str(ctx.exception))
            self.assertIn("json_recovery_attempted: False", str(ctx.exception))
            self.assertFalse(self.provider.last_json_recovery_attempted)
            self.assertEqual(mock_client.post.call_count, 1)

    # 6. HTTP 429 existing retry behavior
    async def test_06_http429_retry_behavior_preserved(self):
        rate_resp = httpx.Response(
            status_code=429,
            headers={"Retry-After": "0.01"},
            json={"error": {"message": "Rate limit reached"}},
            request=httpx.Request("POST", "https://api.groq.com")
        )
        ok_resp = httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"content": "Script content after rate limit"}}]},
            request=httpx.Request("POST", "https://api.groq.com")
        )
        mock_client = AsyncMock()
        mock_client.post.side_effect = [rate_resp, ok_resp]

        with patch.object(self.provider, "_get_client", return_value=mock_client):
            res = await self.provider._post_groq("test prompt", is_json=False)
            self.assertEqual(res, "Script content after rate limit")
            self.assertEqual(self.provider.last_retry_count, 1)
            self.assertEqual(mock_client.post.call_count, 2)

    # 7. HTTP 500 server error behavior
    async def test_07_http500_server_error_no_recovery(self):
        err_resp = httpx.Response(
            status_code=500,
            json={"error": {"code": "internal_server_error", "message": "Server crashed"}},
            request=httpx.Request("POST", "https://api.groq.com")
        )
        mock_client = AsyncMock()
        mock_client.post.return_value = err_resp

        with patch.object(self.provider, "_get_client", return_value=mock_client):
            with self.assertRaises(ScriptProviderError) as ctx:
                await self.provider._post_groq("test prompt", is_json=True, operation="server_err")
            self.assertIn("HTTP error status 500", str(ctx.exception))
            self.assertIn("internal_server_error", str(ctx.exception))
            self.assertIn("json_recovery_attempted: False", str(ctx.exception))
            self.assertFalse(self.provider.last_json_recovery_attempted)
            self.assertEqual(mock_client.post.call_count, 1)

    # 8. Network timeout handling
    async def test_08_network_timeout_handling(self):
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.TimeoutException("Connection timed out")

        with patch.object(self.provider, "_get_client", return_value=mock_client):
            with self.assertRaises(ScriptGenerationTimeoutError) as ctx:
                await self.provider._post_groq("test prompt", is_json=True, operation="timeout_op")
            self.assertIn("Groq request timed out", str(ctx.exception))
            self.assertIn("timeout_op", str(ctx.exception))

    # 9. JSON recovery attempted exactly once
    async def test_09_recovery_attempted_exactly_once_on_double_400(self):
        err_resp = httpx.Response(
            status_code=400,
            json={"error": {"code": "json_validate_failed", "message": "Validation failed"}},
            request=httpx.Request("POST", "https://api.groq.com")
        )
        rec_err_resp = httpx.Response(
            status_code=400,
            json={"error": {"code": "another_bad_request", "message": "Still bad"}},
            request=httpx.Request("POST", "https://api.groq.com")
        )
        mock_client = AsyncMock()
        mock_client.post.side_effect = [err_resp, rec_err_resp]

        with patch.object(self.provider, "_get_client", return_value=mock_client):
            with self.assertRaises(ScriptProviderError) as ctx:
                await self.provider._post_groq("test prompt", is_json=True, operation="bounded_op")
            self.assertIn("json_recovery_attempted: True", str(ctx.exception))
            self.assertIn("fallback_failed: True", str(ctx.exception))
            self.assertTrue(self.provider.last_json_recovery_attempted)
            self.assertTrue(self.provider.last_json_recovery_failed)
            # Exactly 2 calls: 1 initial + 1 recovery
            self.assertEqual(mock_client.post.call_count, 2)

    # 10. Recovery request omits response_format
    async def test_10_recovery_request_omits_response_format(self):
        err_resp = httpx.Response(
            status_code=400,
            json={"error": {"code": "json_validate_failed", "message": "Validation failed"}},
            request=httpx.Request("POST", "https://api.groq.com")
        )
        rec_resp = httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"content": '{"result": "ok"}'}}]},
            request=httpx.Request("POST", "https://api.groq.com")
        )
        mock_client = AsyncMock()
        mock_client.post.side_effect = [err_resp, rec_resp]

        with patch.object(self.provider, "_get_client", return_value=mock_client):
            await self.provider._post_groq("test prompt", is_json=True)
            self.assertEqual(mock_client.post.call_count, 2)
            first_body = mock_client.post.call_args_list[0][1]["json"]
            second_body = mock_client.post.call_args_list[1][1]["json"]
            self.assertEqual(first_body["response_format"], {"type": "json_object"})
            self.assertNotIn("response_format", second_body)

    # 11. Original model preserved in recovery
    async def test_11_original_model_preserved_in_recovery(self):
        self.provider.model = "openai/gpt-oss-120b"
        err_resp = httpx.Response(
            status_code=400,
            json={"error": {"code": "json_validate_failed"}},
            request=httpx.Request("POST", "https://api.groq.com")
        )
        rec_resp = httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"content": '{"ok": true}'}}]},
            request=httpx.Request("POST", "https://api.groq.com")
        )
        mock_client = AsyncMock()
        mock_client.post.side_effect = [err_resp, rec_resp]

        with patch.object(self.provider, "_get_client", return_value=mock_client):
            await self.provider._post_groq("test", is_json=True)
            rec_body = mock_client.post.call_args_list[1][1]["json"]
            self.assertEqual(rec_body["model"], "openai/gpt-oss-120b")

    # 12. Original prompt and messages preserved in recovery
    async def test_12_original_prompt_preserved_in_recovery(self):
        prompt_text = "Unique verbatim prompt for video generation 12345"
        err_resp = httpx.Response(
            status_code=400,
            json={"error": {"code": "json_validate_failed"}},
            request=httpx.Request("POST", "https://api.groq.com")
        )
        rec_resp = httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"content": '{"ok": true}'}}]},
            request=httpx.Request("POST", "https://api.groq.com")
        )
        mock_client = AsyncMock()
        mock_client.post.side_effect = [err_resp, rec_resp]

        with patch.object(self.provider, "_get_client", return_value=mock_client):
            await self.provider._post_groq(prompt_text, is_json=True, temperature=0.1)
            rec_body = mock_client.post.call_args_list[1][1]["json"]
            self.assertEqual(rec_body["messages"], [{"role": "user", "content": prompt_text}])
            self.assertEqual(rec_body["temperature"], 0.1)

    # 13. No credentials leaked in errors
    async def test_13_no_credentials_leaked_in_errors(self):
        secret_key = "gsk_SUPER_SECRET_KEY_NEVER_LOG_9999"
        self.provider.api_key = secret_key
        err_resp = httpx.Response(
            status_code=400,
            json={"error": {"code": "some_error", "message": "Detailed error"}},
            request=httpx.Request("POST", "https://api.groq.com")
        )
        mock_client = AsyncMock()
        mock_client.post.return_value = err_resp

        with patch.object(self.provider, "_get_client", return_value=mock_client):
            with self.assertRaises(ScriptProviderError) as ctx:
                await self.provider._post_groq("test prompt", is_json=True)
            err_msg = str(ctx.exception)
            self.assertNotIn(secret_key, err_msg)
            self.assertNotIn("Authorization", err_msg)
            self.assertNotIn("Bearer", err_msg)

    # 14. Scene splitting works with recovered JSON
    async def test_14_split_into_scenes_with_recovered_json(self):
        script = "Heat a pan over medium flame, then add butter and stir gently."
        scenes_json = json.dumps({
            "scenes": [
                {"scene_order": 1, "scene_text": "Heat a pan over medium flame,"},
                {"scene_order": 2, "scene_text": "then add butter and stir gently."}
            ]
        })
        err_resp = httpx.Response(
            status_code=400,
            json={"error": {"code": "json_validate_failed", "message": "Failed to validate JSON"}},
            request=httpx.Request("POST", "https://api.groq.com")
        )
        # Even wrapped in markdown fences in fallback
        rec_resp = httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"content": f"```json\n{scenes_json}\n```"}}]},
            request=httpx.Request("POST", "https://api.groq.com")
        )
        mock_client = AsyncMock()
        mock_client.post.side_effect = [err_resp, rec_resp]

        with patch.object(self.provider, "_get_client", return_value=mock_client):
            scenes = await self.provider.split_into_scenes(script, 10)
            self.assertEqual(len(scenes), 2)
            self.assertEqual(scenes[0]["scene_order"], 1)
            self.assertEqual(scenes[0]["scene_text"], "Heat a pan over medium flame,")
            self.assertEqual(scenes[1]["scene_order"], 2)
            self.assertTrue(self.provider.last_json_recovery_attempted)
            self.assertFalse(self.provider.last_json_recovery_failed)

    # 15. Visual specification works with recovered JSON
    async def test_15_generate_visual_specification_with_recovered_json(self):
        spec_json = json.dumps({
            "visual_prompt": "Close-up of butter melting in a hot stainless pan.",
            "search_keywords": ["melting butter pan", "butter sizzling pan"],
            "action_timeline": [
                {
                    "action_description": "Butter melts and foams",
                    "visual_start_seconds": 0.0,
                    "visual_end_seconds": 3.0,
                    "sound_type": "action",
                    "sfx_query": "butter sizzling",
                    "sfx_start_seconds": 0.5,
                    "sfx_duration_seconds": 2.0,
                    "sfx_volume": 0.3
                }
            ],
            "shot_plan": None
        })
        err_resp = httpx.Response(
            status_code=400,
            json={"error": {"code": "json_validate_failed", "message": "Validation failed"}},
            request=httpx.Request("POST", "https://api.groq.com")
        )
        rec_resp = httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"content": spec_json}}]},
            request=httpx.Request("POST", "https://api.groq.com")
        )
        mock_client = AsyncMock()
        mock_client.post.side_effect = [err_resp, rec_resp]

        with patch.object(self.provider, "_get_client", return_value=mock_client):
            spec = await self.provider.generate_visual_specification("Add butter to pan.")
            self.assertIn("melting butter pan", spec["search_keywords"])
            self.assertEqual(len(spec["action_timeline"]), 1)
            self.assertTrue(self.provider.last_json_recovery_attempted)

    # 16. Invalid schema is still rejected after recovery
    async def test_16_invalid_schema_rejected_after_recovery(self):
        # Valid JSON syntax, but missing required visual_prompt schema field
        broken_schema_json = json.dumps({"search_keywords": ["tag1"]})
        err_resp = httpx.Response(
            status_code=400,
            json={"error": {"code": "json_validate_failed"}},
            request=httpx.Request("POST", "https://api.groq.com")
        )
        rec_resp = httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"content": broken_schema_json}}]},
            request=httpx.Request("POST", "https://api.groq.com")
        )
        mock_client = AsyncMock()
        mock_client.post.side_effect = [err_resp, rec_resp]

        with patch.object(self.provider, "_get_client", return_value=mock_client):
            with self.assertRaises(InvalidScriptResponseError) as ctx:
                await self.provider.generate_visual_specification("Add butter.")
            self.assertIn("Pydantic visual specification validation failed", str(ctx.exception))

    # 17. Existing shot-plan fallback remains functional with recovered JSON
    async def test_17_shot_plan_fallback_functional_with_recovered_json(self):
        # Valid base spec, but invalid shot_plan (shot_index 5 instead of 1)
        spec_with_invalid_shot = json.dumps({
            "visual_prompt": "Close-up of butter melting in hot pan.",
            "search_keywords": ["butter melting pan", "sizzling butter skillet"],
            "action_timeline": [],
            "shot_plan": [
                {
                    "shot_index": 5,  # Invalid: should start at 1
                    "required_visible_action": "Butter melting",
                    "primary_subject": "Butter",
                    "required_setting": "Kitchen pan",
                    "visual_state": "melting",
                    "camera_framing": "close-up",
                    "search_keywords": ["melting butter", "butter in pan"],
                    "provisional_start_seconds": 0.0,
                    "provisional_end_seconds": 2.0,
                    "expected_duration": 2.0
                }
            ]
        })
        err_resp = httpx.Response(
            status_code=400,
            json={"error": {"code": "json_validate_failed"}},
            request=httpx.Request("POST", "https://api.groq.com")
        )
        rec_resp = httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"content": spec_with_invalid_shot}}]},
            request=httpx.Request("POST", "https://api.groq.com")
        )
        mock_client = AsyncMock()
        mock_client.post.side_effect = [err_resp, rec_resp]

        with patch.object(self.provider, "_get_client", return_value=mock_client):
            spec = await self.provider.generate_visual_specification("Butter melting.")
            # Verify it fell back to single-shot visual specification (shot_plan=None)
            self.assertIsNone(spec["shot_plan"])
            self.assertTrue(self.provider.last_shot_plan_fallback_occurred)
            self.assertTrue(self.provider.last_json_recovery_attempted)

    # 18. No public signature changes
    def test_18_no_public_signature_changes(self):
        sig_gen_script = inspect.signature(GroqScriptProvider.generate_script)
        self.assertEqual(list(sig_gen_script.parameters.keys()), ["self", "topic", "desired_duration_seconds"])

        sig_split = inspect.signature(GroqScriptProvider.split_into_scenes)
        self.assertEqual(list(sig_split.parameters.keys()), ["self", "script", "desired_duration_seconds"])

        sig_vspec = inspect.signature(GroqScriptProvider.generate_visual_specification)
        self.assertEqual(list(sig_vspec.parameters.keys()), ["self", "scene_text"])

        sig_words = inspect.signature(GroqScriptProvider.calculate_target_words)
        self.assertEqual(list(sig_words.parameters.keys()), ["self", "duration_seconds"])

        sig_clean = inspect.signature(GroqScriptProvider.clean_script_text)
        self.assertEqual(list(sig_clean.parameters.keys()), ["self", "text"])


if __name__ == "__main__":
    unittest.main()
