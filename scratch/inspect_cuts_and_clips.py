import os
import sys
import av
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\AiVideoGenerator")

project_id = "2143bbc8-8061-4639-b974-475f6071336d"
clips_dir = rf"C:\AiVideoGenerator\media\projects\{project_id}\clips"
final_video = rf"C:\AiVideoGenerator\media\projects\{project_id}\video_{project_id}.mp4"
keyframes_dir = rf"C:\AiVideoGenerator\media\projects\{project_id}\keyframes"
os.makedirs(keyframes_dir, exist_ok=True)

# 1. Inspect each sourced clip
clips = sorted([f for f in os.listdir(clips_dir) if f.endswith(".mp4")])
print(f"Sourced Clips in clips_dir ({len(clips)} clips):")
for c in clips:
    path = os.path.join(clips_dir, c)
    container = av.open(path)
    vs = container.streams.video[0]
    fps = float(vs.average_rate or 24.0)
    frames = vs.frames or 0
    dur = float(container.duration / av.time_base) if container.duration else 0.0
    
    # Grab middle frame of the clip
    mid_frame = None
    f_idx = 0
    for frame in container.decode(video=0):
        if f_idx == frames // 2 or mid_frame is None:
            mid_frame = frame.to_ndarray(format="rgb24")
        f_idx += 1
    container.close()

    clip_thumb = os.path.join(keyframes_dir, f"source_clip_{c}.jpg")
    if mid_frame is not None:
        Image.fromarray(mid_frame).save(clip_thumb, quality=85)
        lum = float(np.mean(mid_frame))
        print(f"  • {c} ({dur:.2f}s, {f_idx} frames, {vs.codec_context.width}x{vs.codec_context.height}): Lum={lum:.1f} -> {os.path.basename(clip_thumb)}")

# 2. Inspect boundaries in final video
print("\nInspecting Final Video Cuts & Boundaries...")
container = av.open(final_video)
all_frames = []
for frame in container.decode(video=0):
    all_frames.append(frame.to_ndarray(format="rgb24"))
container.close()
print(f"Total Decoded Frames in Final Video: {len(all_frames)}")

# Timeline frame boundaries from narration durations & reconciled shots:
# Scene 1: 4.416s (106 frames: Shot 1 = 51 frames [0..50], Shot 2 = 55 frames [51..105])
# Scene 2: 5.088s (123 frames: Shot 1 = 60 frames [106..165], Shot 2 = 63 frames [166..228])
# Scene 3: 5.424s (131 frames: Shot 1 = 63 frames [229..291], Shot 2 = 68 frames [292..359])
# Scene 4: 5.376s (129 frames: Single Shot [360..488])
# Scene 5: 3.936s (95 frames: Shot 1 = 51 frames [489..539], Shot 2 = 44 frames [540..583])

cuts = [
    ("Scene 1 Shot 1 -> Shot 2 (Intra-scene cut)", 50, 51),
    ("Scene 1 -> Scene 2 (Inter-scene transition)", 105, 106),
    ("Scene 2 Shot 1 -> Shot 2 (Intra-scene cut)", 165, 166),
    ("Scene 2 -> Scene 3 (Inter-scene transition)", 228, 229),
    ("Scene 3 Shot 1 -> Shot 2 (Intra-scene cut)", 291, 292),
    ("Scene 3 -> Scene 4 (Inter-scene transition)", 359, 360),
    ("Scene 4 -> Scene 5 (Inter-scene transition)", 488, 489),
    ("Scene 5 Shot 1 -> Shot 2 (Intra-scene cut)", 539, 540)
]

for label, f1, f2 in cuts:
    if f1 < len(all_frames) and f2 < len(all_frames):
        arr1 = all_frames[f1]
        arr2 = all_frames[f2]
        lum1 = float(np.mean(arr1))
        lum2 = float(np.mean(arr2))
        diff = float(np.mean(np.abs(arr1.astype(np.float32) - arr2.astype(np.float32))))
        print(f"\n  Cut [{label}]:")
        print(f"    Frame {f1}: Lum={lum1:.1f} | RGB center: {arr1[360, 640].tolist()}")
        print(f"    Frame {f2}: Lum={lum2:.1f} | RGB center: {arr2[360, 640].tolist()}")
        print(f"    Pixel Delta: {diff:.2f}")

        # Save boundary frames
        Image.fromarray(arr1).save(os.path.join(keyframes_dir, f"boundary_{f1}.jpg"), quality=85)
        Image.fromarray(arr2).save(os.path.join(keyframes_dir, f"boundary_{f2}.jpg"), quality=85)

print("\nAll boundary inspections complete.")
