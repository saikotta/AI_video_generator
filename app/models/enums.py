import enum

class ProjectStatus(str, enum.Enum):
    QUEUED = "queued"
    GENERATING_SCRIPT = "generating_script"
    SPLITTING_SCENES = "splitting_scenes"
    GENERATING_VISUALS = "generating_visuals"
    GENERATING_VOICE = "generating_voice"
    ASSEMBLING = "assembling"
    DONE = "done"
    FAILED = "failed"

class AssetType(str, enum.Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"

class ProviderStage(str, enum.Enum):
    SCRIPT = "script"
    SCENE_SPLIT = "scene_split"
    PROMPT_GEN = "prompt_gen"
    VISUAL = "visual"
    VOICE = "voice"
    ASSEMBLY = "assembly"

class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
