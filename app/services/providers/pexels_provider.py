import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
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

# NOTE: This is the MVP visual implementation and covers stock IMAGES only.
# Pexels video-clip search and AI-generated visuals are future providers
# behind the same StockMediaProvider interface, not built in this step.

logger = logging.getLogger(__name__)

class PexelsImageProvider(StockMediaProvider):
    """
    Pexels API implementation of StockMediaProvider for stock images.
    """

    PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        min_shorter_side_px: int = 1280,
        min_quality_score: float = 50.0
    ):
        self.api_key = (api_key or settings.PEXELS_API_KEY or os.getenv("PEXELS_API_KEY", "")).strip()
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.min_shorter_side_px = min_shorter_side_px
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

    def _evaluate_photo(self, photo: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Evaluate photo against selection criteria:
        1. Must contain valid src dictionary and download URL, width > 0, height > 0 (Criterion 4)
        2. Calculate score: prefer landscape orientation (width > height) and resolution >= min_shorter_side_px (Criteria 2 & 3)
        Returns evaluated candidate dict with score if score >= min_quality_score, else None.
        """
        photo_id = photo.get("id")
        src = photo.get("src", {})
        width = photo.get("width", 0)
        height = photo.get("height", 0)
        source_url = photo.get("url", "")
        photographer = photo.get("photographer", "Unknown")
        alt = photo.get("alt", "")

        # 4. Reject results with no usable image data
        download_url = src.get("large2x") or src.get("original") or src.get("large")
        if not photo_id or not download_url or not source_url or width <= 0 or height <= 0:
            return None

        is_landscape = width > height
        shorter_side = min(width, height)
        aspect_ratio = width / float(height)

        score = 0.0

        # Criterion 2: Prefer landscape orientation (+40 pts, bonus for ~16:9 widescreen format +10 pts)
        if is_landscape:
            score += 40.0
            if 1.3 <= aspect_ratio <= 2.2:
                score += 10.0

        # Criterion 3: Prefer resolution above threshold (at least 1280px on shorter side +40 pts)
        if shorter_side >= self.min_shorter_side_px:
            score += 40.0
        elif shorter_side >= 720:
            score += 20.0
        elif shorter_side < 600:
            score -= 30.0  # Penalty for very small resolution

        if score < self.min_quality_score:
            return None

        return {
            "photo_id": photo_id,
            "source_url": source_url,
            "download_url": download_url,
            "width": width,
            "height": height,
            "orientation": "landscape" if is_landscape else "portrait/square",
            "aspect_ratio": round(aspect_ratio, 2),
            "photographer": photographer,
            "alt": alt,
            "score": score
        }

    async def search_media(self, keywords: List[str]) -> MediaAssetSchema:
        """
        Search Pexels API using candidate keywords in order (primary first, secondary next).
        Applies concrete selection criteria. Returns MediaAssetSchema or raises NoSuitableMediaFoundError.
        """
        if not keywords:
            raise NoSuitableMediaFoundError("Keyword list is empty; cannot perform stock media search.")

        cleaned_keywords = [k.strip() for k in keywords if k and k.strip()]
        if not cleaned_keywords:
            raise NoSuitableMediaFoundError("No valid non-empty keywords provided.")

        async with httpx.AsyncClient(timeout=15.0) as client:
            for kw in cleaned_keywords:
                params = {"query": kw, "per_page": 15}
                try:
                    data = await self._make_api_request(client, self.PEXELS_SEARCH_URL, params)
                except (PexelsAuthenticationError, PexelsRateLimitError):
                    # Re-raise authentication and rate limit errors immediately
                    raise

                photos = data.get("photos", [])
                if not photos:
                    logger.info(f"Pexels returned 0 photos for keyword '{kw}'. Trying next keyword...")
                    continue

                best_candidate: Optional[Dict[str, Any]] = None
                best_score = -1.0

                for photo in photos:
                    candidate = self._evaluate_photo(photo)
                    if candidate and candidate["score"] > best_score:
                        best_score = candidate["score"]
                        best_candidate = candidate

                if best_candidate and best_score >= self.min_quality_score:
                    logger.info(f"Selected Pexels image ID {best_candidate['photo_id']} using keyword '{kw}' (Score: {best_score})")
                    return MediaAssetSchema(
                        asset_type="image",
                        provider="pexels",
                        source_url=best_candidate["source_url"],
                        download_url=best_candidate["download_url"],
                        metadata={
                            "pexels_id": best_candidate["photo_id"],
                            "width": best_candidate["width"],
                            "height": best_candidate["height"],
                            "orientation": best_candidate["orientation"],
                            "aspect_ratio": best_candidate["aspect_ratio"],
                            "photographer": best_candidate["photographer"],
                            "alt": best_candidate["alt"],
                            "keyword_used": kw,
                            "score": best_candidate["score"]
                        }
                    )

                logger.info(f"No photo meeting quality criteria for keyword '{kw}'. Trying secondary keyword...")

        # Criterion 5: Return clear "no suitable result" error if nothing acceptable is found after trying all keywords
        raise NoSuitableMediaFoundError(f"No suitable stock image found matching selection criteria for keywords: {keywords}")

    async def download_media(self, download_url: str, output_dir: str, filename_prefix: str) -> str:
        """
        Download image from download_url into output_dir.
        Validates non-empty file size and image file magic headers (JPEG, PNG, WebP).
        Returns local file path string.
        """
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{filename_prefix}.jpg")

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    res = await client.get(download_url)
                    if res.status_code == 200:
                        content = res.content
                        if len(content) < 5000:
                            raise StockMediaError(f"Downloaded image size too small ({len(content)} bytes).")

                        # Validate binary image headers
                        is_jpeg = content.startswith(b'\xff\xd8\xff')
                        is_png = content.startswith(b'\x89PNG\r\n\x1a\n')
                        is_webp = content.startswith(b'RIFF') and b'WEBP' in content[8:16]

                        if not (is_jpeg or is_png or is_webp):
                            raise StockMediaError("Downloaded file content does not match valid image headers (JPEG/PNG/WebP).")

                        with open(file_path, "wb") as f:
                            f.write(content)

                        return file_path

                    if res.status_code == 429 and attempt < self.max_retries:
                        await asyncio.sleep(self.retry_base_delay * (2 ** attempt))
                        continue

                    raise StockMediaError(f"Failed to download image from {download_url}: HTTP {res.status_code}")

                except (httpx.TimeoutException, httpx.NetworkError) as e:
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.retry_base_delay * (2 ** attempt))
                        continue
                    raise StockMediaError(f"Network error downloading image from {download_url}: {e}")

        raise StockMediaError(f"Download failed after retries for {download_url}")
