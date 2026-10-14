from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class MediaAssetSchema(BaseModel):
    asset_type: str = Field("image", description="Type of media asset, e.g. image, video")
    provider: str = Field("pexels", description="Name of stock media provider, e.g. pexels")
    source_url: str = Field(..., description="Attribution / source page URL for media asset")
    download_url: str = Field(..., description="Direct URL to download full-resolution image media asset")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Detailed metadata including dimensions, photographer, keyword, asset ID, etc.")
