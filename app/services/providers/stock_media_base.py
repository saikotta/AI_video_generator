from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from app.schemas.media import MediaAssetSchema

# NOTE: This is the MVP visual implementation and covers stock IMAGES only.
# Pexels video-clip search and AI-generated visuals are future providers
# behind the same StockMediaProvider interface, not built in this step.

class StockMediaProvider(ABC):
    """
    Abstract Base Class interface for stock media providers.
    
    All media providers (stock images, stock videos, AI generation) must implement
    this interface to supply structured media assets for scene visuals.
    """

    @abstractmethod
    async def search_media(self, keywords: List[str]) -> MediaAssetSchema:
        """
        Search for stock media using candidate search keywords (ordered primary to secondary).
        
        Selection criteria must prioritize high quality, landscape orientation, and resolution.
        Returns a structured MediaAssetSchema instance.
        """
        pass

    @abstractmethod
    async def download_media(self, download_url: str, output_dir: str, filename_prefix: str) -> str:
        """
        Download media file from download_url into output_dir.
        
        Returns the absolute or relative local filesystem path to the downloaded media file.
        """
        pass
