import math
from typing import List, Optional, Literal, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class ActionEventSchema(BaseModel):
    """
    Structured action event for a visual beat, carrying physical action description
    and optional synchronized sound effect cues.
    """
    action_description: str = Field(..., description="Concrete description of observable physical action")
    visual_start_seconds: float = Field(0.0, description="Provisional or measured start time in seconds relative to scene start")
    visual_end_seconds: float = Field(..., description="Provisional or measured end time in seconds relative to scene start")
    sfx_query: Optional[str] = Field(None, description="Action sound effect query (e.g. 'water droplets falling'), or None if naturally silent")
    sfx_start_seconds: Optional[float] = Field(None, description="Start offset in seconds for SFX playback relative to scene start")
    sfx_duration_seconds: Optional[float] = Field(None, description="Duration in seconds of the sound effect")
    sfx_volume: Optional[float] = Field(None, description="Relative volume level between 0.0 and 1.0")
    sound_type: Literal["action", "ambience", "none"] = Field("none", description="Classification of sound: 'action', 'ambience', or 'none'")

    @field_validator('action_description')
    @classmethod
    def validate_action_description(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("action_description cannot be empty")
        return cleaned

    @field_validator('visual_start_seconds')
    @classmethod
    def validate_start_non_negative(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError(f"visual_start_seconds must be non-negative, got {value}")
        return round(float(value), 3)

    @field_validator('visual_end_seconds')
    @classmethod
    def validate_end_positive(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError(f"visual_end_seconds must be greater than 0, got {value}")
        return round(float(value), 3)

    @field_validator('sfx_volume')
    @classmethod
    def validate_volume(cls, value: Optional[float]) -> Optional[float]:
        if value is not None:
            if value < 0.0 or value > 1.0:
                raise ValueError(f"sfx_volume must be between 0.0 and 1.0, got {value}")
            return round(float(value), 3)
        return None

    @field_validator('sfx_start_seconds')
    @classmethod
    def validate_sfx_start(cls, value: Optional[float]) -> Optional[float]:
        if value is not None:
            if value < 0.0:
                raise ValueError(f"sfx_start_seconds must be non-negative, got {value}")
            return round(float(value), 3)
        return None

    @field_validator('sfx_duration_seconds')
    @classmethod
    def validate_sfx_duration(cls, value: Optional[float]) -> Optional[float]:
        if value is not None:
            if value <= 0.0:
                raise ValueError(f"sfx_duration_seconds must be greater than 0, got {value}")
            return round(float(value), 3)
        return None

    @model_validator(mode='after')
    def validate_event_logic(self) -> 'ActionEventSchema':
        if self.visual_start_seconds >= self.visual_end_seconds:
            raise ValueError(
                f"visual_start_seconds ({self.visual_start_seconds}) must be strictly less than "
                f"visual_end_seconds ({self.visual_end_seconds})"
            )
        if self.sound_type in ("action", "ambience"):
            if not self.sfx_query or not self.sfx_query.strip():
                raise ValueError(f"sfx_query cannot be empty when sound_type is '{self.sound_type}'")
        return self


class ShotSpecificationSchema(BaseModel):
    """
    Structured camera shot specification within a multi-action scene beat.
    Defines one concrete physical action, primary subject, setting, and search terms.
    """
    shot_index: int = Field(..., ge=1, description="1-indexed sequence number of the shot within the scene beat")
    required_visible_action: str = Field(..., description="Concrete description of observable physical action on screen")
    primary_subject: str = Field(..., description="Primary entity or object being filmed")
    required_setting: str = Field(..., description="Environment or setting of the shot")
    visual_state: str = Field("active", description="Visual state (e.g. 'empty', 'sizzling', 'raw', 'clean')")
    camera_framing: str = Field("close-up", description="Camera perspective (e.g. 'macro close-up', 'extreme close-up', 'overhead')")
    search_keywords: List[str] = Field(..., description="List of 2 to 5 specific stock search queries for this shot")
    provisional_start_seconds: float = Field(0.0, ge=0.0, description="Provisional AI start time in seconds relative to scene start")
    provisional_end_seconds: float = Field(..., gt=0.0, description="Provisional AI end time in seconds relative to scene start")
    expected_duration: Optional[float] = Field(None, description="Expected duration in seconds")
    action_event_index: Optional[int] = Field(None, ge=0, description="Optional index into action_timeline events if linked")
    required_objects: Optional[List[str]] = Field(default=None, description="Optional concrete physical objects that must be visible on screen")
    explicit_exclusions: Optional[List[str]] = Field(default=None, description="Optional explicit objects, subjects, settings, or states that must NOT be present (contradictions)")

    @field_validator('required_visible_action')
    @classmethod
    def validate_action(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("required_visible_action cannot be empty")
        return cleaned

    @field_validator('primary_subject')
    @classmethod
    def validate_subject(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("primary_subject cannot be empty")
        return cleaned

    @field_validator('required_setting')
    @classmethod
    def validate_setting(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("required_setting cannot be empty")
        return cleaned

    @field_validator('visual_state')
    @classmethod
    def validate_state(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("visual_state cannot be empty")
        return cleaned

    @field_validator('camera_framing')
    @classmethod
    def validate_framing(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("camera_framing cannot be empty")
        return cleaned

    @field_validator('search_keywords')
    @classmethod
    def validate_keywords(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("search_keywords list cannot be empty")
        cleaned = [k.strip() for k in value if k and k.strip()]
        if len(cleaned) < 2:
            raise ValueError(f"search_keywords must contain at least 2 useful keywords, got {len(cleaned)}")
        return cleaned

    @field_validator('provisional_start_seconds')
    @classmethod
    def validate_start(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError(f"provisional_start_seconds must be non-negative, got {value}")
        return round(float(value), 4)

    @field_validator('provisional_end_seconds')
    @classmethod
    def validate_end(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError(f"provisional_end_seconds must be greater than zero, got {value}")
        return round(float(value), 4)

    @field_validator('required_objects')
    @classmethod
    def validate_objects(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return None
        cleaned = [o.strip() for o in value if o and o.strip()]
        return cleaned if cleaned else None

    @field_validator('explicit_exclusions')
    @classmethod
    def validate_exclusions(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return None
        cleaned = [e.strip() for e in value if e and e.strip()]
        return cleaned if cleaned else None

    @model_validator(mode='after')
    def validate_shot_logic(self) -> 'ShotSpecificationSchema':
        if self.provisional_start_seconds >= self.provisional_end_seconds:
            raise ValueError(
                f"provisional_start_seconds ({self.provisional_start_seconds}) must be strictly less than "
                f"provisional_end_seconds ({self.provisional_end_seconds})"
            )
        calc_duration = round(self.provisional_end_seconds - self.provisional_start_seconds, 4)
        if self.expected_duration is not None:
            if self.expected_duration <= 0.0:
                raise ValueError(f"expected_duration must be positive, got {self.expected_duration}")
        else:
            self.expected_duration = calc_duration
        return self


class VisualSpecificationSchema(BaseModel):
    visual_prompt: str = Field(..., description="Short concrete visual description of observable scene elements")
    search_keywords: List[str] = Field(..., description="List of concrete stock media search terms")
    action_timeline: Optional[List[ActionEventSchema]] = Field(
        default=None,
        description="Optional list of structured chronological action and sound events"
    )
    shot_plan: Optional[List[ShotSpecificationSchema]] = Field(
        default=None,
        description="Optional list of sequential sub-scene shots for multi-action scenes"
    )

    @field_validator('visual_prompt')
    @classmethod
    def validate_visual_prompt(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("visual_prompt cannot be empty")
        return cleaned

    @field_validator('search_keywords')
    @classmethod
    def validate_search_keywords(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("search_keywords list cannot be empty")
        cleaned_keywords = [k.strip() for k in value if k and k.strip()]
        if len(cleaned_keywords) < 2:
            raise ValueError(f"search_keywords list must contain at least 2 useful keywords, got {len(cleaned_keywords)}")
        return cleaned_keywords

    @field_validator('action_timeline')
    @classmethod
    def validate_action_timeline_ordering(cls, timeline: Optional[List[ActionEventSchema]]) -> Optional[List[ActionEventSchema]]:
        if not timeline:
            return timeline
        for i in range(len(timeline) - 1):
            if timeline[i].visual_start_seconds > timeline[i + 1].visual_start_seconds:
                raise ValueError(
                    f"Action timeline events must be chronologically ordered: "
                    f"event {i} starts at {timeline[i].visual_start_seconds}s which is after "
                    f"event {i+1} starting at {timeline[i+1].visual_start_seconds}s"
                )
        return timeline

    @field_validator('shot_plan')
    @classmethod
    def validate_shot_plan_structure(cls, shots: Optional[List[ShotSpecificationSchema]]) -> Optional[List[ShotSpecificationSchema]]:
        if not shots:
            return shots

        # Maximum supported shot count per scene beat
        MAX_SHOTS = 4
        if len(shots) > MAX_SHOTS:
            raise ValueError(f"shot_plan exceeds maximum supported shot count ({MAX_SHOTS} shots maximum, got {len(shots)})")

        for i, s in enumerate(shots):
            expected_index = i + 1
            if s.shot_index != expected_index:
                raise ValueError(
                    f"Shot index at position {i} must be {expected_index}, got {s.shot_index}. "
                    f"Shot indices must be 1-indexed and strictly sequential."
                )
            if i < len(shots) - 1:
                curr_end = s.provisional_end_seconds
                next_start = shots[i + 1].provisional_start_seconds
                if curr_end > next_start:
                    raise ValueError(
                        f"Shot ranges cannot overlap: shot {s.shot_index} ends at {curr_end}s, "
                        f"which is after shot {shots[i + 1].shot_index} starts at {next_start}s."
                    )

        return shots

    @model_validator(mode='after')
    def validate_shot_action_linkage(self) -> 'VisualSpecificationSchema':
        if self.shot_plan:
            max_action_idx = (len(self.action_timeline) - 1) if self.action_timeline else -1
            for s in self.shot_plan:
                if s.action_event_index is not None:
                    if self.action_timeline is None or len(self.action_timeline) == 0:
                        raise ValueError(
                            f"Shot {s.shot_index} specifies action_event_index {s.action_event_index}, "
                            f"but no action_timeline is defined."
                        )
                    if s.action_event_index < 0 or s.action_event_index > max_action_idx:
                        raise ValueError(
                            f"Shot {s.shot_index} specifies action_event_index {s.action_event_index}, "
                            f"which is out of range for action_timeline with {len(self.action_timeline)} events."
                        )
        return self


class ReconciledShot(dict):
    """
    Structured dictionary representing a reconciled camera shot with frame-exact timing.
    Accessible via both dictionary key syntax (shot["start_seconds"]) and attribute syntax (shot.start_seconds).
    Serializes directly to JSON.
    """
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'ReconciledShot' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def to_dict(self) -> Dict[str, Any]:
        return dict(self)


def reconcile_shot_plan_timing(
    shot_plan: Optional[List[Union[ShotSpecificationSchema, Dict[str, Any]]]],
    measured_scene_duration_seconds: float,
    fps: int = 24
) -> List[ReconciledShot]:
    """
    Pure helper function to reconcile provisional AI shot boundaries with actual measured
    voiceover audio duration using deterministic frame-exact proportional allocation.

    Guarantees:
    1. Total integer frame budget is computed via ceiling: ceil(measured_duration * fps).
    2. Proportional integer frame allocation based on provisional durations.
    3. Every shot receives at least one frame.
    4. Sum of allocated frames strictly equals total scene frame budget.
    5. Contiguous boundaries with zero gaps and zero overlaps.
    6. Exact start_seconds and end_seconds derived directly from integer frame numbers.
    7. Input shot_plan is never modified in place.
    8. Deterministic remainder distribution (largest remainder method).
    """
    if not shot_plan:
        return []

    if measured_scene_duration_seconds <= 0.0:
        raise ValueError(f"measured_scene_duration_seconds must be positive, got {measured_scene_duration_seconds}")

    if fps <= 0:
        raise ValueError(f"fps must be a positive integer, got {fps}")

    total_frames = int(math.ceil(round(measured_scene_duration_seconds * fps, 6)))
    num_shots = len(shot_plan)

    if total_frames < num_shots:
        raise ValueError(
            f"Total frame budget ({total_frames}) is smaller than number of shots ({num_shots}). "
            f"Every shot must receive at least one frame."
        )

    # Validate items without mutating input data
    validated_shots: List[ShotSpecificationSchema] = []
    for item in shot_plan:
        if isinstance(item, ShotSpecificationSchema):
            validated_shots.append(item.model_copy())
        elif isinstance(item, dict):
            validated_shots.append(ShotSpecificationSchema.model_validate(dict(item)))
        else:
            raise ValueError(f"Invalid shot item type: {type(item)}")

    for i, s in enumerate(validated_shots):
        if s.provisional_end_seconds <= s.provisional_start_seconds:
            raise ValueError(
                f"Shot {s.shot_index} has non-positive duration: provisional_start_seconds ({s.provisional_start_seconds}) "
                f"must be strictly less than provisional_end_seconds ({s.provisional_end_seconds})."
            )
        expected_idx = i + 1
        if s.shot_index != expected_idx:
            raise ValueError(
                f"Shot index at position {i} must be {expected_idx}, got {s.shot_index}. "
                f"Shot indices must be 1-indexed and strictly sequential."
            )
        if i < num_shots - 1:
            curr_end = s.provisional_end_seconds
            next_start = validated_shots[i + 1].provisional_start_seconds
            if curr_end > next_start:
                raise ValueError(
                    f"Shot ranges cannot overlap: shot {s.shot_index} ends at {curr_end}s, "
                    f"which is after shot {validated_shots[i + 1].shot_index} starts at {next_start}s."
                )

    durations = [s.provisional_end_seconds - s.provisional_start_seconds for s in validated_shots]
    total_provisional = sum(durations)
    if total_provisional <= 0.0:
        raise ValueError("Total provisional duration across shots must be greater than zero.")

    # 1. Allocate minimum 1 frame per shot
    allocated = [1] * num_shots
    remaining_frames = total_frames - num_shots

    # 2. Proportional distribution of remaining frames
    if remaining_frames > 0:
        ideal_frames = [(d / total_provisional) * total_frames for d in durations]
        excess_ideal = [max(0.0, f - 1.0) for f in ideal_frames]
        sum_excess = sum(excess_ideal)

        if sum_excess > 0.0:
            raw_additions = [(remaining_frames * (e / sum_excess)) for e in excess_ideal]
            base_additions = [int(math.floor(r)) for r in raw_additions]
            remainders = [r - b for r, b in zip(raw_additions, base_additions)]
        else:
            base_additions = [0] * num_shots
            remainders = [f - 1.0 for f in ideal_frames]

        for i in range(num_shots):
            allocated[i] += base_additions[i]

        leftover = remaining_frames - sum(base_additions)
        if leftover > 0:
            # Deterministic tie-breaking: largest remainder first, tie-break by shot_index ascending
            sorted_indices = sorted(range(num_shots), key=lambda idx: (-remainders[idx], idx))
            for k in range(leftover):
                target_idx = sorted_indices[k % num_shots]
                allocated[target_idx] += 1

    assert sum(allocated) == total_frames, f"Allocated frame sum {sum(allocated)} != total budget {total_frames}"

    # 3. Derive contiguous frame boundaries and exact timestamps
    reconciled: List[ReconciledShot] = []
    curr_frame = 0

    for i, shot in enumerate(validated_shots):
        frames_i = allocated[i]
        start_frame = curr_frame
        end_frame = curr_frame + frames_i
        curr_frame = end_frame

        start_sec = round(start_frame / float(fps), 4)
        end_sec = round(end_frame / float(fps), 4)
        dur_sec = round(frames_i / float(fps), 4)

        r = ReconciledShot({
            "shot_index": shot.shot_index,
            "required_visible_action": shot.required_visible_action,
            "primary_subject": shot.primary_subject,
            "required_setting": shot.required_setting,
            "visual_state": shot.visual_state,
            "camera_framing": shot.camera_framing,
            "search_keywords": list(shot.search_keywords),
            "allocated_frames": frames_i,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_seconds": start_sec,
            "end_seconds": end_sec,
            "duration_seconds": dur_sec,
            "provisional_start_seconds": shot.provisional_start_seconds,
            "provisional_end_seconds": shot.provisional_end_seconds,
            "provisional_duration_seconds": round(shot.provisional_end_seconds - shot.provisional_start_seconds, 4),
            "action_event_index": shot.action_event_index,
            "required_objects": list(shot.required_objects) if shot.required_objects else [],
            "explicit_exclusions": list(shot.explicit_exclusions) if shot.explicit_exclusions else [],
        })
        reconciled.append(r)

    return reconciled


def validate_action_timeline_timing(
    timeline: Optional[List[ActionEventSchema]],
    max_duration_seconds: Optional[float] = None
) -> Optional[List[ActionEventSchema]]:
    """
    Deterministic timing validation function to reconcile provisional AI timestamps
    with final measured TTS audio duration.
    - Clamps visual_end_seconds to max_duration_seconds if provided.
    - Disables SFX that start beyond scene duration.
    """
    if not timeline:
        return timeline

    validated: List[ActionEventSchema] = []
    for ev in timeline:
        ev_dict = ev.model_dump()
        if max_duration_seconds is not None and max_duration_seconds > 0:
            if ev_dict["visual_start_seconds"] >= max_duration_seconds:
                continue
            if ev_dict["visual_end_seconds"] > max_duration_seconds:
                ev_dict["visual_end_seconds"] = max_duration_seconds
            if ev_dict.get("sfx_start_seconds") is not None and ev_dict["sfx_start_seconds"] >= max_duration_seconds:
                ev_dict["sfx_start_seconds"] = None
                ev_dict["sfx_duration_seconds"] = None
                ev_dict["sfx_query"] = None
                ev_dict["sound_type"] = "none"
        validated.append(ActionEventSchema.model_validate(ev_dict))

    return validated

