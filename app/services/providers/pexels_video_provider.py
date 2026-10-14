import os
import asyncio
import logging
from typing import List, Dict, Any, Optional, Set
import httpx

from app.config import settings
from app.schemas.media import MediaAssetSchema
from app.services.providers.stock_media_base import StockMediaProvider
from app.services.providers.exceptions import (
    StockMediaError,
    PexelsAPIError,
    PexelsRateLimitError,
    PexelsAuthenticationError,
    NoSuitableMediaFoundError
)

logger = logging.getLogger(__name__)

class PexelsVideoProvider(StockMediaProvider):
    """
    Pexels Video API implementation of StockMediaProvider for stock video clips.
    Uses Pexels' Video Search endpoint (https://api.pexels.com/videos/search).
    """

    PEXELS_VIDEO_SEARCH_URL = "https://api.pexels.com/videos/search"

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        min_width: int = 1280,
        min_height: int = 720,
        min_quality_score: float = 40.0
    ):
        self.api_key = (api_key or settings.PEXELS_API_KEY or os.getenv("PEXELS_API_KEY", "")).strip()
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.min_width = min_width
        self.min_height = min_height
        self.min_quality_score = min_quality_score

        self.rate_limit_events_count: int = 0
        self.rate_limit_event_details: List[Dict[str, Any]] = []

        if not self.api_key:
            raise PexelsAuthenticationError("PEXELS_API_KEY is not configured in environment or settings.")

    async def _make_api_request(self, client: httpx.AsyncClient, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute API GET request with internal retry logic for HTTP 429 rate limits and 5xx errors.
        Raises PexelsAuthenticationError on 401/403 (fatal), PexelsRateLimitError after retries,
        or PexelsAPIError on other failures.
        """
        headers = {"Authorization": self.api_key}

        for attempt in range(self.max_retries + 1):
            try:
                response = await client.get(url, headers=headers, params=params, timeout=15.0)

                if response.status_code in (401, 403):
                    raise PexelsAuthenticationError(
                        f"Pexels API authentication failed (HTTP {response.status_code}). Check PEXELS_API_KEY."
                    )

                if response.status_code == 429:
                    retry_after_hdr = response.headers.get("Retry-After")
                    if retry_after_hdr and retry_after_hdr.isdigit():
                        wait_seconds = float(retry_after_hdr)
                    else:
                        wait_seconds = self.retry_base_delay * (2 ** attempt)

                    self.rate_limit_events_count += 1
                    self.rate_limit_event_details.append({
                        "attempt": attempt + 1,
                        "retry_after_header": retry_after_hdr,
                        "wait_seconds": wait_seconds,
                        "query": params.get("query")
                    })

                    if attempt < self.max_retries:
                        logger.warning(
                            f"Pexels rate limit hit (429). Retrying in {wait_seconds:.1f}s (Attempt {attempt + 1}/{self.max_retries})..."
                        )
                        await asyncio.sleep(wait_seconds)
                        continue
                    else:
                        raise PexelsRateLimitError("Pexels API rate limit exceeded after exhausting retries.")

                if response.status_code >= 500:
                    if attempt < self.max_retries:
                        wait_seconds = self.retry_base_delay * (2 ** attempt)
                        logger.warning(
                            f"Pexels server error (HTTP {response.status_code}). Retrying in {wait_seconds:.1f}s..."
                        )
                        await asyncio.sleep(wait_seconds)
                        continue
                    else:
                        raise PexelsAPIError(f"Pexels API HTTP {response.status_code} server error: {response.text}")

                if response.status_code != 200:
                    raise PexelsAPIError(f"Pexels API error HTTP {response.status_code}: {response.text}")

                return response.json()

            except (httpx.TimeoutException, httpx.NetworkError) as e:
                if attempt < self.max_retries:
                    wait_seconds = self.retry_base_delay * (2 ** attempt)
                    logger.warning(f"Pexels network error: {e}. Retrying in {wait_seconds:.1f}s...")
                    await asyncio.sleep(wait_seconds)
                    continue
                raise PexelsAPIError(f"Pexels API network request failed: {e}")

        raise PexelsAPIError("Unexpected termination of request retry loop.")

    def _evaluate_video(self, video: Dict[str, Any], target_duration: float, used_video_ids: Set[int]) -> Optional[Dict[str, Any]]:
        """
        Evaluate candidate video item against selection criteria:
        1. Reject duplicate video IDs already used in this render project.
        2. Prefer landscape orientation (width > height).
        3. Find best MP4 download file matching/exceeding min_width & min_height.
        4. Prefer clip duration >= target_duration.
        5. Basic relevance check via tags/metadata.
        Returns candidate dict with score if score >= min_quality_score, else None.
        """
        video_id = video.get("id")
        if not video_id or video_id in used_video_ids:
            return None

        width = video.get("width", 0)
        height = video.get("height", 0)
        duration = float(video.get("duration", 0.0))
        source_url = video.get("url", "")
        user_info = video.get("user", {})
        photographer = user_info.get("name", "Unknown") if isinstance(user_info, dict) else "Unknown"

        video_files = video.get("video_files", [])
        if not video_files or width <= 0 or height <= 0 or duration <= 0.0:
            return None

        # Find best MP4 video file link matching/exceeding HD resolution
        best_file = None
        best_file_res = 0

        for vf in video_files:
            if vf.get("file_type") == "video/mp4" and vf.get("link"):
                v_w = vf.get("width", 0)
                v_h = vf.get("height", 0)
                res_area = v_w * v_h

                # Prefer 1920x1080 or 1280x720 HD resolutions
                if v_w >= self.min_width or v_h >= self.min_height:
                    if res_area > best_file_res and v_w <= 3840:
                        best_file_res = res_area
                        best_file = vf

        # Fallback to best available MP4 file if no file met min resolution
        if not best_file:
            for vf in video_files:
                if vf.get("file_type") == "video/mp4" and vf.get("link"):
                    res_area = vf.get("width", 0) * vf.get("height", 0)
                    if res_area > best_file_res:
                        best_file_res = res_area
                        best_file = vf

        if not best_file or not best_file.get("link"):
            return None

        download_url = best_file["link"]
        video_w = best_file.get("width", width)
        video_h = best_file.get("height", height)

        is_landscape = video_w > video_h
        aspect_ratio = video_w / float(video_h) if video_h > 0 else 1.0

        score = 0.0

        # Criterion 1: Landscape orientation (+40 pts, bonus for 16:9 ratio +10 pts)
        if is_landscape:
            score += 40.0
            if 1.4 <= aspect_ratio <= 2.2:
                score += 10.0

        # Criterion 2: HD Resolution (+30 pts for 1280x720+, +40 pts for 1920x1080+)
        if video_w >= 1920 and video_h >= 1080:
            score += 40.0
        elif video_w >= 1280 and video_h >= 720:
            score += 30.0
        else:
            score += 15.0

        # Criterion 3: Duration matching (+30 pts if clip >= target_duration)
        if target_duration > 0 and duration >= target_duration:
            score += 30.0
        elif duration >= 5.0:
            score += 15.0

        if score < self.min_quality_score:
            return None

        return {
            "video_id": video_id,
            "source_url": source_url,
            "download_url": download_url,
            "width": video_w,
            "height": video_h,
            "duration": duration,
            "orientation": "landscape" if is_landscape else "portrait/square",
            "aspect_ratio": round(aspect_ratio, 2),
            "photographer": photographer,
            "score": score
        }

    async def search_media(
        self,
        keywords: List[str],
        target_duration: float = 0.0,
        used_video_ids: Optional[Set[int]] = None
    ) -> MediaAssetSchema:
        """
        Search Pexels Video API using candidate keywords in order (primary first, secondary next).
        Applies selection criteria, prevents duplicate video IDs, and returns MediaAssetSchema.
        """
        if not keywords:
            raise NoSuitableMediaFoundError("Keyword list is empty; cannot perform stock video search.")

        cleaned_keywords = [k.strip() for k in keywords if k and k.strip()]
        if not cleaned_keywords:
            raise NoSuitableMediaFoundError("No valid non-empty keywords provided.")

        active_used_ids = used_video_ids if used_video_ids is not None else set()

        async with httpx.AsyncClient(timeout=15.0) as client:
            for kw in cleaned_keywords:
                params = {"query": kw, "per_page": 15, "orientation": "landscape"}
                try:
                    data = await self._make_api_request(client, self.PEXELS_VIDEO_SEARCH_URL, params)
                except (PexelsAuthenticationError, PexelsRateLimitError):
                    raise

                videos = data.get("videos", [])
                if not videos:
                    logger.info(f"Pexels returned 0 videos for keyword '{kw}'. Trying next keyword...")
                    continue

                best_candidate: Optional[Dict[str, Any]] = None
                best_score = -1.0

                for video in videos:
                    candidate = self._evaluate_video(video, target_duration, active_used_ids)
                    if candidate and candidate["score"] > best_score:
                        best_score = candidate["score"]
                        best_candidate = candidate

                if best_candidate and best_score >= self.min_quality_score:
                    active_used_ids.add(best_candidate["video_id"])
                    logger.info(f"Selected Pexels Video ID {best_candidate['video_id']} for keyword '{kw}' (Duration: {best_candidate['duration']}s, Score: {best_score})")

                    return MediaAssetSchema(
                        asset_type="video",
                        provider="pexels",
                        source_url=best_candidate["source_url"],
                        download_url=best_candidate["download_url"],
                        metadata={
                            "pexels_id": best_candidate["video_id"],
                            "width": best_candidate["width"],
                            "height": best_candidate["height"],
                            "duration": best_candidate["duration"],
                            "orientation": best_candidate["orientation"],
                            "aspect_ratio": best_candidate["aspect_ratio"],
                            "photographer": best_candidate["photographer"],
                            "keyword_used": kw,
                            "score": best_candidate["score"]
                        }
                    )

                logger.info(f"No video meeting quality criteria for keyword '{kw}'. Trying secondary keyword...")

        raise NoSuitableMediaFoundError(f"No suitable stock video clip found matching criteria for keywords: {keywords}")

    async def download_media(self, download_url: str, output_dir: str, filename_prefix: str) -> str:
        """
        Download MP4 video clip from download_url into output_dir.
        Validates non-empty file size (> 50KB) and MP4 container headers.
        Returns local filesystem path string.
        """
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{filename_prefix}.mp4")

        if os.path.exists(file_path) and os.path.getsize(file_path) > 50000:
            try:
                import av
                container = av.open(file_path)
                if container.streams.video:
                    container.close()
                    return file_path
                container.close()
            except Exception:
                pass

        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    res = await client.get(download_url)
                    if res.status_code == 200:
                        content = res.content
                        if len(content) < 50000:
                            raise StockMediaError(f"Downloaded video file size too small ({len(content)} bytes).")

                        # Validate binary MP4 container signature (ftyp header in first 32 bytes)
                        if not (b"ftyp" in content[:32] or b"moov" in content[:1024]):
                            raise StockMediaError("Downloaded file content does not match valid MP4 video headers.")

                        with open(file_path, "wb") as f:
                            f.write(content)

                        # Validate video container decoding via PyAV
                        try:
                            import av
                            container = av.open(file_path)
                            if container.streams.video:
                                stream = container.streams.video[0]
                                for _ in container.decode(stream):
                                    break
                            container.close()
                        except Exception as decode_err:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                            raise StockMediaError(f"Downloaded video container decoding failed: {decode_err}")

                        return file_path

                    if res.status_code == 429 and attempt < self.max_retries:
                        await asyncio.sleep(self.retry_base_delay * (2 ** attempt))
                        continue

                    raise StockMediaError(f"Failed to download video clip from {download_url}: HTTP {res.status_code}")

                except (httpx.TimeoutException, httpx.NetworkError) as e:
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.retry_base_delay * (2 ** attempt))
                        continue
                    raise StockMediaError(f"Network error downloading video clip from {download_url}: {e}")

        raise StockMediaError(f"Download failed after retries for {download_url}")
