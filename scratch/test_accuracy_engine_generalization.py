"""
General-Purpose Accuracy Engine — Multi-Topic Generalization Test Suite

Comprehensive offline test suite verifying the visual verification, candidate selection,
bounded recovery, and audio/Foley relevance across 6 distinct domains:
1. Vegetable Stir-Fry (Cooking/Culinary)
2. Seed Germination (Botany/Natural Growth)
3. Car Engine Operation (Mechanical Engineering/Internal Mechanics)
4. Photosynthesis (Biochemistry/Cellular & Molecular Processes)
5. Ocean Waves (Geophysical/Fluid Dynamics)
6. Construction Equipment (Heavy Industry/Civil Engineering)

Tests:
- Correct candidate accepted (ACCEPTED)
- Wrong subject rejected (REJECTED / Exclusion)
- Wrong physical action rejected (Action Mismatch / Static footage)
- Wrong setting rejected (Cross-domain / Setting contradiction)
- Ambiguous candidate marked uncertain (UNCERTAIN)
- Conceptual feasibility classification (CONCEPTUAL_ABSTRACT -> diagram/motion graphic)
- Cross-domain contextuality (e.g. bathtub inappropriate for food, appropriate for bathing)
- Audio/Foley relevance and violence/gore exclusion
- Bounded recovery and unresolved shot handling (never silently substitute)
"""

import os
import sys
import asyncio
import tempfile
import unittest
import numpy as np
import av
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas.visual_spec import (
    ShotSpecificationSchema,
    VisualSpecificationSchema,
    ReconciledShot,
)
from app.services.visual_verification import (
    VisualVerifier,
    HeuristicVisualVerifier,
    ConceptClassifier,
    FrameExtractor,
    VerificationStatus,
    ConceptFeasibility,
    VisualVerificationResult,
)
from app.services.providers.pexels_video_provider import PexelsVideoProvider
from app.services.providers.freesound_provider import FreesoundAudioProvider
from app.schemas.media import MediaAssetSchema


def create_test_video(
    path: str,
    base_color: tuple,
    duration: float = 1.0,
    fps: int = 10,
    size: tuple = (320, 240),
    is_dynamic: bool = True
) -> str:
    """Creates a real lightweight H.264 MP4 with either dynamic motion or static frames."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    container = av.open(path, mode='w')
    stream = container.add_stream('libx264', rate=fps)
    stream.width = size[0]
    stream.height = size[1]
    stream.pix_fmt = 'yuv420p'
    num_frames = int(round(duration * fps))

    for i in range(num_frames):
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        if is_dynamic:
            # Shift color and moving bar across frames to produce motion delta
            shift = (i * 15) % 256
            c = ((base_color[0] + shift) % 256, (base_color[1] + shift) % 256, (base_color[2] + shift) % 256)
            img[:, :] = c
            # Moving rectangle
            x_start = (i * 20) % (size[0] - 40)
            img[50:150, x_start:x_start + 40] = 255
        else:
            # Perfectly static
            img[:, :] = base_color

        frame = av.VideoFrame.from_ndarray(img, format='rgb24')
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


class TestAccuracyEngineVegetableStirFry(unittest.IsolatedAsyncioTestCase):
    """Domain 1: Vegetable Stir-Fry (Culinary / Food Preparation)"""

    async def asyncSetUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="test_veg_")
        self.verifier = VisualVerifier()
        self.shot_reqs = {
            "primary_subject": "vegetables",
            "required_visible_action": "stir frying vigorously in a hot wok pan",
            "required_setting": "kitchen stovetop",
            "visual_state": "sizzling vibrant vegetables in hot pan",
            "required_objects": ["wok", "spatula", "vegetables"],
            "explicit_exclusions": ["meat", "beef", "chicken", "pork", "seafood", "bathtub", "bathroom"],
        }

    async def asyncTearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    async def test_01_correct_candidate_accepted(self):
        vid_path = os.path.join(self.tmp_dir, "correct_stir_fry.mp4")
        create_test_video(vid_path, base_color=(40, 180, 50), is_dynamic=True)

        meta = {
            "title": "Cooking stir fry with vegetables in kitchen wok",
            "tags": "cooking vegetables stir fry wok kitchen spatula food",
            "semantic_reasons": ["vegetables in kitchen wok"],
        }
        res = await self.verifier.verify_candidate_clip(vid_path, self.shot_reqs, meta)
        self.assertEqual(res.status, VerificationStatus.ACCEPTED)
        self.assertGreaterEqual(res.confidence, 0.70)
        self.assertEqual(len(res.contradictions), 0)
        self.assertTrue(any("Subject terms matched" in m for m in res.confirmed_matches))

    async def test_02_wrong_subject_meat_rejected(self):
        vid_path = os.path.join(self.tmp_dir, "beef_steak.mp4")
        create_test_video(vid_path, base_color=(180, 50, 40), is_dynamic=True)

        meta = {
            "title": "Grilling thick raw beef steak with butter in kitchen",
            "tags": "cooking beef steak meat grill kitchen pan",
            "semantic_reasons": ["beef meat in pan"],
        }
        res = await self.verifier.verify_candidate_clip(vid_path, self.shot_reqs, meta)
        self.assertEqual(res.status, VerificationStatus.REJECTED)
        self.assertTrue(any("meat" in c or "beef" in c for c in res.contradictions))

    async def test_03_wrong_physical_action_static_pan_penalized(self):
        vid_path = os.path.join(self.tmp_dir, "static_veggies.mp4")
        # Generate completely motionless video
        create_test_video(vid_path, base_color=(50, 150, 50), is_dynamic=False)

        meta = {
            "title": "Vegetables sitting still inside unheated cookware",
            "tags": "vegetables kitchen pan",
        }
        res = await self.verifier.verify_candidate_clip(vid_path, self.shot_reqs, meta)
        # Should be rejected due to static frozen footage contradiction on active physical action
        self.assertEqual(res.status, VerificationStatus.REJECTED)
        self.assertTrue(any("Static frozen footage" in c for c in res.contradictions))

    async def test_04_wrong_setting_bathtub_rejected(self):
        vid_path = os.path.join(self.tmp_dir, "bathtub.mp4")
        create_test_video(vid_path, base_color=(200, 220, 240), is_dynamic=True)

        meta = {
            "title": "Pouring bath salts into running warm water in a white bathtub",
            "tags": "bathroom bathtub water bath salt relax",
        }
        res = await self.verifier.verify_candidate_clip(vid_path, self.shot_reqs, meta)
        self.assertEqual(res.status, VerificationStatus.REJECTED)
        self.assertTrue(any("bathtub" in c or "bathroom" in c for c in res.contradictions))

    async def test_05_ambiguous_candidate_marked_uncertain(self):
        vid_path = os.path.join(self.tmp_dir, "cutting_board.mp4")
        create_test_video(vid_path, base_color=(100, 160, 60), is_dynamic=True)

        meta = {
            "title": "Green peppers resting on cutting board",
            "tags": "peppers food salad",
        }
        res = await self.verifier.verify_candidate_clip(vid_path, self.shot_reqs, meta)
        self.assertEqual(res.status, VerificationStatus.UNCERTAIN)
        self.assertEqual(len(res.contradictions), 0)


class TestAccuracyEngineSeedGermination(unittest.IsolatedAsyncioTestCase):
    """Domain 2: Seed Germination (Botany / Biological Growth)"""

    async def asyncSetUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="test_seed_")
        self.verifier = VisualVerifier()
        self.shot_reqs = {
            "primary_subject": "seed sprout",
            "required_visible_action": "seed coat splitting and sprout emerging downward into dark soil",
            "required_setting": "underground moist dark soil",
            "visual_state": "emerging white root and green shoot from seed",
            "required_objects": ["seed", "soil", "sprout"],
            "explicit_exclusions": ["mature tree", "full grown plant", "kitchen", "cooking", "salad", "grocery"],
        }

    async def asyncTearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    async def test_01_correct_candidate_accepted(self):
        vid_path = os.path.join(self.tmp_dir, "seed_sprout.mp4")
        create_test_video(vid_path, base_color=(70, 50, 20), is_dynamic=True)

        meta = {
            "title": "Time-lapse of bean seed germinating in soil as root sprout emerges and grows",
            "tags": "seed sprout germination soil root growing",
        }
        res = await self.verifier.verify_candidate_clip(vid_path, self.shot_reqs, meta)
        self.assertEqual(res.status, VerificationStatus.ACCEPTED)
        self.assertGreaterEqual(res.confidence, 0.70)
        self.assertTrue(any("Subject terms matched" in m for m in res.confirmed_matches))

    async def test_02_wrong_subject_mature_tree_rejected(self):
        vid_path = os.path.join(self.tmp_dir, "mature_tree.mp4")
        create_test_video(vid_path, base_color=(20, 120, 30), is_dynamic=True)

        meta = {
            "title": "Huge mature tree with thick branches swaying in the summer breeze",
            "tags": "nature mature tree forest woodland oak leaves",
        }
        res = await self.verifier.verify_candidate_clip(vid_path, self.shot_reqs, meta)
        self.assertEqual(res.status, VerificationStatus.REJECTED)
        self.assertTrue(any("mature tree" in c for c in res.contradictions))

    async def test_03_wrong_physical_action_seeds_in_jar_mismatch(self):
        vid_path = os.path.join(self.tmp_dir, "seeds_in_jar.mp4")
        create_test_video(vid_path, base_color=(120, 100, 70), is_dynamic=False)

        meta = {
            "title": "Dry chia seeds stored motionless in a clear glass container",
            "tags": "seeds glass jar storage dry",
        }
        res = await self.verifier.verify_candidate_clip(vid_path, self.shot_reqs, meta)
        self.assertEqual(res.status, VerificationStatus.REJECTED)
        self.assertTrue(any("Static frozen footage" in c for c in res.contradictions))

    async def test_04_wrong_setting_kitchen_cutting_board_rejected(self):
        vid_path = os.path.join(self.tmp_dir, "chopping_seeds.mp4")
        create_test_video(vid_path, base_color=(150, 140, 90), is_dynamic=True)

        meta = {
            "title": "Chef preparing sunflower seeds in kitchen cooking salad",
            "tags": "kitchen cooking salad sunflower seeds cutting board chef",
        }
        res = await self.verifier.verify_candidate_clip(vid_path, self.shot_reqs, meta)
        self.assertEqual(res.status, VerificationStatus.REJECTED)
        self.assertTrue(any("kitchen" in c or "cooking" in c or "salad" in c for c in res.contradictions))

    async def test_05_ambiguous_candidate_marked_uncertain(self):
        vid_path = os.path.join(self.tmp_dir, "potting_soil.mp4")
        create_test_video(vid_path, base_color=(60, 45, 25), is_dynamic=True)

        meta = {
            "title": "Moist black potting soil in a garden bed with sunlight reflections",
            "tags": "earth dirt garden background",
        }
        res = await self.verifier.verify_candidate_clip(vid_path, self.shot_reqs, meta)
        self.assertEqual(res.status, VerificationStatus.UNCERTAIN)
        self.assertEqual(len(res.contradictions), 0)


class TestAccuracyEngineCarEngineOperation(unittest.IsolatedAsyncioTestCase):
    """Domain 3: Car Engine Operation (Mechanical Engineering / Internal Mechanism)"""

    async def asyncSetUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="test_engine_")
        self.verifier = VisualVerifier()
        self.shot_reqs = {
            "primary_subject": "engine piston",
            "required_visible_action": "piston reciprocating inside cylinder block during four stroke cycle",
            "required_setting": "internal engine cylinder chamber",
            "visual_state": "mechanical stroke of piston and opening intake valve",
            "required_objects": ["piston", "cylinder", "engine"],
            "explicit_exclusions": ["highway driving", "car exterior", "traffic", "car wash", "dealership"],
        }

    async def asyncTearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_conceptual_feasibility_classification(self):
        feasibility, rationale = ConceptClassifier.classify_feasibility(
            action="four stroke cycle with piston reciprocating inside cylinder",
            subject="engine piston cylinder combustion",
            visual_prompt="Internal combustion chamber mechanism"
        )
        self.assertEqual(feasibility, ConceptFeasibility.CONCEPTUAL_ABSTRACT)
        self.assertIn("internal", rationale.lower())

    async def test_02_correct_candidate_cutaway_accepted(self):
        vid_path = os.path.join(self.tmp_dir, "engine_piston_cutaway.mp4")
        create_test_video(vid_path, base_color=(100, 100, 110), is_dynamic=True)

        meta = {
            "title": "Mechanical cutaway of engine piston moving inside cylinder",
            "tags": "engine piston cylinder mechanics combustion",
        }
        res = await self.verifier.verify_candidate_clip(vid_path, self.shot_reqs, meta)
        self.assertEqual(res.status, VerificationStatus.ACCEPTED)
        self.assertGreaterEqual(res.confidence, 0.70)
        self.assertTrue(any("Subject terms matched" in m for m in res.confirmed_matches))

    async def test_03_wrong_subject_highway_driving_rejected(self):
        vid_path = os.path.join(self.tmp_dir, "highway_driving.mp4")
        create_test_video(vid_path, base_color=(80, 120, 160), is_dynamic=True)

        meta = {
            "title": "Modern blue sedan driving smoothly on highway at sunset",
            "tags": "car driving highway driving road trip traffic vehicle",
        }
        res = await self.verifier.verify_candidate_clip(vid_path, self.shot_reqs, meta)
        self.assertEqual(res.status, VerificationStatus.REJECTED)
        self.assertTrue(any("highway driving" in c or "driving" in c for c in res.contradictions))

    async def test_04_wrong_physical_action_parked_car_mismatch(self):
        vid_path = os.path.join(self.tmp_dir, "parked_car.mp4")
        create_test_video(vid_path, base_color=(70, 70, 70), is_dynamic=False)

        meta = {
            "title": "Parked car standing motionless in garage driveway",
            "tags": "car engine bay garage",
        }
        res = await self.verifier.verify_candidate_clip(vid_path, self.shot_reqs, meta)
        self.assertEqual(res.status, VerificationStatus.REJECTED)
        self.assertTrue(any("Static frozen footage" in c for c in res.contradictions))

    async def test_05_wrong_setting_car_dealership_rejected(self):
        vid_path = os.path.join(self.tmp_dir, "dealership.mp4")
        create_test_video(vid_path, base_color=(200, 200, 210), is_dynamic=True)

        meta = {
            "title": "Customer browsing new cars in dealership showroom",
            "tags": "dealership showroom sales luxury cars interior",
        }
        res = await self.verifier.verify_candidate_clip(vid_path, self.shot_reqs, meta)
        self.assertEqual(res.status, VerificationStatus.REJECTED)
        self.assertTrue(any("dealership" in c for c in res.contradictions))

    async def test_06_ambiguous_candidate_engine_bay_marked_uncertain(self):
        vid_path = os.path.join(self.tmp_dir, "engine_bay.mp4")
        create_test_video(vid_path, base_color=(90, 90, 95), is_dynamic=True)

        meta = {
            "title": "Mechanic looking at open car hood showing engine cover and battery",
            "tags": "car motor hood mechanic workshop garage",
        }
        res = await self.verifier.verify_candidate_clip(vid_path, self.shot_reqs, meta)
        self.assertEqual(res.status, VerificationStatus.UNCERTAIN)
        self.assertEqual(len(res.contradictions), 0)


class TestAccuracyEnginePhotosynthesis(unittest.IsolatedAsyncioTestCase):
    """Domain 4: Photosynthesis (Biochemistry / Cellular & Molecular Processes)"""

    async def asyncSetUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="test_photo_")
        self.verifier = VisualVerifier()
        self.shot_reqs = {
            "primary_subject": "chloroplast",
            "required_visible_action": "chloroplast moving inside plant cell capturing light",
            "required_setting": "microscopic plant cell",
            "visual_state": "green chloroplasts under microscope",
            "required_objects": ["chloroplast", "plant cell"],
            "explicit_exclusions": ["human", "sunbathing", "beach", "vacation", "sunglasses", "supermarket", "kitchen"],
        }

    async def asyncTearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_conceptual_feasibility_classification(self):
        feasibility, rationale = ConceptClassifier.classify_feasibility(
            action="chlorophyll synthesizing energy from sunlight in plant",
            subject="chloroplast cellular organelle photosynthesis",
            visual_prompt="Microscopic view of plant cell thylakoid"
        )
        self.assertEqual(feasibility, ConceptFeasibility.CONCEPTUAL_ABSTRACT)
        self.assertIn("microscopic", rationale.lower())

    async def test_02_correct_candidate_microscopic_chloroplast_accepted(self):
        vid_path = os.path.join(self.tmp_dir, "chloroplast_microscopy.mp4")
        create_test_video(vid_path, base_color=(20, 160, 40), is_dynamic=True)

        meta = {
            "title": "Microscopic view of chloroplast moving inside plant cell",
            "tags": "microscopic plant cell chloroplast chlorophyll biology science",
        }
        res = await self.verifier.verify_candidate_clip(vid_path, self.shot_reqs, meta)
        self.assertEqual(res.status, VerificationStatus.ACCEPTED)
        self.assertGreaterEqual(res.confidence, 0.70)
        self.assertTrue(any("Subject terms matched" in m for m in res.confirmed_matches))

    async def test_03_wrong_subject_beach_sunbathing_rejected(self):
        vid_path = os.path.join(self.tmp_dir, "sunbathing.mp4")
        create_test_video(vid_path, base_color=(230, 180, 120), is_dynamic=True)

        meta = {
            "title": "Woman in sunglasses sunbathing on sandy tropical beach in summer sun",
            "tags": "beach sunbathing vacation human sunshine summer ocean",
        }
        res = await self.verifier.verify_candidate_clip(vid_path, self.shot_reqs, meta)
        self.assertEqual(res.status, VerificationStatus.REJECTED)
        self.assertTrue(any("sunbathing" in c or "beach" in c or "human" in c for c in res.contradictions))

    async def test_04_wrong_setting_supermarket_produce_rejected(self):
        vid_path = os.path.join(self.tmp_dir, "supermarket.mp4")
        create_test_video(vid_path, base_color=(140, 150, 130), is_dynamic=True)

        meta = {
            "title": "Shopper selecting green vegetables in supermarket produce aisle",
            "tags": "supermarket grocery produce aisle vegetables store",
        }
        res = await self.verifier.verify_candidate_clip(vid_path, self.shot_reqs, meta)
        self.assertEqual(res.status, VerificationStatus.REJECTED)
        self.assertTrue(any("supermarket" in c for c in res.contradictions))

    async def test_05_ambiguous_candidate_forest_canopy_marked_uncertain(self):
        vid_path = os.path.join(self.tmp_dir, "forest_canopy.mp4")
        create_test_video(vid_path, base_color=(30, 100, 30), is_dynamic=True)

        meta = {
            "title": "Golden sunlight shining through dense green forest tree leaves",
            "tags": "green leaves forest sunlight trees nature",
        }
        res = await self.verifier.verify_candidate_clip(vid_path, self.shot_reqs, meta)
        self.assertEqual(res.status, VerificationStatus.UNCERTAIN)
        self.assertEqual(len(res.contradictions), 0)


class TestAccuracyEngineOceanWaves(unittest.IsolatedAsyncioTestCase):
    """Domain 5: Ocean Waves (Geophysical / Fluid Dynamics)"""

    async def asyncSetUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="test_ocean_")
        self.verifier = VisualVerifier()
        self.shot_reqs = {
            "primary_subject": "ocean waves",
            "required_visible_action": "powerful ocean waves crashing onto rocky coastline with spray",
            "required_setting": "rocky ocean coast",
            "visual_state": "foaming white surf and turbulent sea waves",
            "required_objects": ["waves", "ocean", "surf"],
            "explicit_exclusions": ["swimming pool", "bathtub", "water park", "indoor pool"],
        }

    async def asyncTearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    async def test_01_correct_candidate_accepted(self):
        vid_path = os.path.join(self.tmp_dir, "crashing_waves.mp4")
        create_test_video(vid_path, base_color=(30, 90, 160), is_dynamic=True)

        meta = {
            "title": "Ocean waves crashing against rocky coastline with sea surf",
            "tags": "ocean waves surf coast sea rocky shore foam",
        }
        res = await self.verifier.verify_candidate_clip(vid_path, self.shot_reqs, meta)
        self.assertEqual(res.status, VerificationStatus.ACCEPTED)
        self.assertGreaterEqual(res.confidence, 0.70)
        self.assertTrue(any("Subject terms matched" in m for m in res.confirmed_matches))

    async def test_02_wrong_subject_swimming_pool_rejected(self):
        vid_path = os.path.join(self.tmp_dir, "swimming_pool.mp4")
        create_test_video(vid_path, base_color=(50, 180, 220), is_dynamic=True)

        meta = {
            "title": "Sunny outdoor swimming pool with clean calm blue water and tile lanes",
            "tags": "swimming pool clear water blue tiles resort swimmer lanes",
        }
        res = await self.verifier.verify_candidate_clip(vid_path, self.shot_reqs, meta)
        self.assertEqual(res.status, VerificationStatus.REJECTED)
        self.assertTrue(any("swimming pool" in c for c in res.contradictions))

    async def test_03_wrong_physical_action_static_water_penalized(self):
        vid_path = os.path.join(self.tmp_dir, "flat_puddle.mp4")
        create_test_video(vid_path, base_color=(70, 90, 110), is_dynamic=False)

        meta = {
            "title": "Motionless calm puddle of rainwater resting on pavement",
            "tags": "water calm flat puddle",
        }
        res = await self.verifier.verify_candidate_clip(vid_path, self.shot_reqs, meta)
        self.assertEqual(res.status, VerificationStatus.REJECTED)
        self.assertTrue(any("Static frozen footage" in c for c in res.contradictions))

    async def test_04_wrong_setting_indoor_bathtub_rejected(self):
        vid_path = os.path.join(self.tmp_dir, "bathtub_ocean.mp4")
        create_test_video(vid_path, base_color=(120, 160, 200), is_dynamic=True)

        meta = {
            "title": "Indoor bathroom bathtub with bubbling warm water",
            "tags": "bathtub whirlpool bathroom indoor pool spa",
        }
        res = await self.verifier.verify_candidate_clip(vid_path, self.shot_reqs, meta)
        self.assertEqual(res.status, VerificationStatus.REJECTED)
        self.assertTrue(any("bathtub" in c or "indoor pool" in c for c in res.contradictions))

    async def test_05_ambiguous_candidate_calm_horizon_marked_uncertain(self):
        vid_path = os.path.join(self.tmp_dir, "calm_sea.mp4")
        create_test_video(vid_path, base_color=(60, 100, 140), is_dynamic=True)

        meta = {
            "title": "Gentle ripples on open water surface stretching toward hazy horizon",
            "tags": "blue horizon water seascape",
        }
        res = await self.verifier.verify_candidate_clip(vid_path, self.shot_reqs, meta)
        self.assertEqual(res.status, VerificationStatus.UNCERTAIN)
        self.assertEqual(len(res.contradictions), 0)


class TestAccuracyEngineConstructionEquipment(unittest.IsolatedAsyncioTestCase):
    """Domain 6: Construction Equipment (Heavy Industry / Civil Engineering)"""

    async def asyncSetUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="test_const_")
        self.verifier = VisualVerifier()
        self.shot_reqs = {
            "primary_subject": "excavator",
            "required_visible_action": "excavator bucket digging deep trench into earth",
            "required_setting": "outdoor active construction site",
            "visual_state": "yellow excavator moving dirt",
            "required_objects": ["excavator", "dirt", "construction site"],
            "explicit_exclusions": ["toy", "miniature", "playroom", "child", "living room", "indoor"],
        }

    async def asyncTearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    async def test_01_correct_candidate_accepted(self):
        vid_path = os.path.join(self.tmp_dir, "excavator_digging.mp4")
        create_test_video(vid_path, base_color=(200, 170, 30), is_dynamic=True)

        meta = {
            "title": "Excavator digging trench at construction site",
            "tags": "excavator heavy machinery construction site digging earth dirt",
        }
        res = await self.verifier.verify_candidate_clip(vid_path, self.shot_reqs, meta)
        self.assertEqual(res.status, VerificationStatus.ACCEPTED)
        self.assertGreaterEqual(res.confidence, 0.70)
        self.assertTrue(any("Subject terms matched" in m for m in res.confirmed_matches))

    async def test_02_wrong_subject_toy_truck_rejected(self):
        vid_path = os.path.join(self.tmp_dir, "toy_truck.mp4")
        create_test_video(vid_path, base_color=(220, 190, 50), is_dynamic=True)

        meta = {
            "title": "Small yellow plastic toy excavator on carpet in kids playroom",
            "tags": "toy miniature child playroom yellow truck plastic",
        }
        res = await self.verifier.verify_candidate_clip(vid_path, self.shot_reqs, meta)
        self.assertEqual(res.status, VerificationStatus.REJECTED)
        self.assertTrue(any("toy" in c or "miniature" in c or "playroom" in c for c in res.contradictions))

    async def test_03_wrong_physical_action_parked_machinery_mismatch(self):
        vid_path = os.path.join(self.tmp_dir, "parked_machinery.mp4")
        create_test_video(vid_path, base_color=(180, 150, 40), is_dynamic=False)

        meta = {
            "title": "Machinery parked still in empty industrial lot overnight",
            "tags": "machinery construction equipment parking lot",
        }
        res = await self.verifier.verify_candidate_clip(vid_path, self.shot_reqs, meta)
        self.assertEqual(res.status, VerificationStatus.REJECTED)
        self.assertTrue(any("Static frozen footage" in c for c in res.contradictions))

    async def test_04_wrong_setting_indoor_living_room_rejected(self):
        vid_path = os.path.join(self.tmp_dir, "indoor_living_room.mp4")
        create_test_video(vid_path, base_color=(180, 140, 110), is_dynamic=True)

        meta = {
            "title": "Indoor living room setting with beige carpet and wooden coffee table",
            "tags": "living room indoor home carpet sofa",
        }
        res = await self.verifier.verify_candidate_clip(vid_path, self.shot_reqs, meta)
        self.assertEqual(res.status, VerificationStatus.REJECTED)
        self.assertTrue(any("living room" in c or "indoor" in c for c in res.contradictions))

    async def test_05_ambiguous_candidate_traffic_cones_marked_uncertain(self):
        vid_path = os.path.join(self.tmp_dir, "traffic_cones.mp4")
        create_test_video(vid_path, base_color=(200, 100, 30), is_dynamic=True)

        meta = {
            "title": "Orange safety cones lined up along asphalt road",
            "tags": "traffic cones roadwork safety orange",
        }
        res = await self.verifier.verify_candidate_clip(vid_path, self.shot_reqs, meta)
        self.assertEqual(res.status, VerificationStatus.UNCERTAIN)
        self.assertEqual(len(res.contradictions), 0)


class TestCrossDomainContextualityAndAudioRelevance(unittest.IsolatedAsyncioTestCase):
    """Cross-Domain Context Sensitivity and Audio/Foley Relevance Verification"""

    async def asyncSetUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="test_cross_")
        self.verifier = VisualVerifier()

    async def asyncTearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    async def test_01_bathtub_rejected_for_cooking_accepted_for_bathing(self):
        vid_path = os.path.join(self.tmp_dir, "bathtub_clip.mp4")
        create_test_video(vid_path, base_color=(200, 220, 240), is_dynamic=True)

        meta = {
            "title": "Running warm water filling a clean white bathtub with soap bubbles",
            "tags": "bathtub warm water bubbles bathroom washing",
        }

        # Context A: Cooking Stir-Fry
        cooking_reqs = {
            "primary_subject": "vegetables",
            "required_visible_action": "stir frying in wok",
            "required_setting": "kitchen stove",
            "explicit_exclusions": ["bathtub", "bathroom", "soap"],
        }
        res_cooking = await self.verifier.verify_candidate_clip(vid_path, cooking_reqs, meta)
        self.assertEqual(res_cooking.status, VerificationStatus.REJECTED)

        # Context B: Infant Bathing Tutorial
        bathing_reqs = {
            "primary_subject": "warm water bath",
            "required_visible_action": "filling bathtub with warm water and bubbles",
            "required_setting": "clean bathroom",
            "required_objects": ["bathtub", "water"],
            "explicit_exclusions": ["kitchen", "cooking", "knife", "stove"],
        }
        res_bathing = await self.verifier.verify_candidate_clip(vid_path, bathing_reqs, meta)
        self.assertEqual(res_bathing.status, VerificationStatus.ACCEPTED)

    async def test_02_meat_rejected_for_vegan_accepted_for_steak(self):
        vid_path = os.path.join(self.tmp_dir, "beef_clip.mp4")
        create_test_video(vid_path, base_color=(190, 60, 50), is_dynamic=True)

        meta = {
            "title": "Searing prime ribeye beef steak with garlic butter in cast iron skillet",
            "tags": "beef steak meat searing grill cast iron pan",
        }

        # Context A: Vegan Stir Fry
        vegan_reqs = {
            "primary_subject": "vegetables",
            "required_visible_action": "stir frying plant ingredients",
            "required_setting": "kitchen",
            "explicit_exclusions": ["meat", "beef", "chicken", "pork", "fish"],
        }
        res_vegan = await self.verifier.verify_candidate_clip(vid_path, vegan_reqs, meta)
        self.assertEqual(res_vegan.status, VerificationStatus.REJECTED)

        # Context B: Steak Grilling Guide
        steak_reqs = {
            "primary_subject": "beef steak",
            "required_visible_action": "searing meat in hot skillet",
            "required_setting": "kitchen stove or grill",
            "required_objects": ["steak", "pan"],
            "explicit_exclusions": ["sweet dessert", "ice cream", "cake"],
        }
        res_steak = await self.verifier.verify_candidate_clip(vid_path, steak_reqs, meta)
        self.assertEqual(res_steak.status, VerificationStatus.ACCEPTED)

    def test_03_freesound_violence_gore_disqualification(self):
        audio_provider = FreesoundAudioProvider()

        # Check violence/gore hard exclusion
        bad_cand = {
            "id": 12345,
            "name": "Human being hit with stick - Flesh Gore Impact.wav",
            "tags": ["gore", "hit", "flesh", "impact", "violence"],
            "description": "Heavy gore flesh impact sound effect",
            "license": "http://creativecommons.org/publicdomain/zero/1.0/"
        }
        rejected, reason = audio_provider._is_hard_rejected(bad_cand)
        self.assertTrue(rejected)
        self.assertTrue(any(term in reason.lower() for term in ("gore", "flesh", "body hit", "human being hit", "horror")))

        good_cand = {
            "id": 67890,
            "name": "Sizzling vegetables in frying pan.wav",
            "tags": ["sizzle", "cooking", "frying", "kitchen", "food"],
            "description": "Crisp sizzle of vegetables cooking in hot skillet",
            "license": "http://creativecommons.org/publicdomain/zero/1.0/"
        }
        rejected_good, _ = audio_provider._is_hard_rejected(good_cand)
        self.assertFalse(rejected_good)

    def test_04_freesound_topic_domain_mismatch_rejection(self):
        audio_provider = FreesoundAudioProvider()
        bathtub_cand = {
            "id": 99111,
            "name": "Bathtub splashing and soap bubbles.wav",
            "tags": ["bathroom", "bathtub", "wash", "splash", "water"],
            "description": "Playing in bubble bath tub water",
            "license": "http://creativecommons.org/publicdomain/zero/1.0/"
        }
        # In a cooking topic, bathtub sound is rejected due to topic conflict
        conflict_cooking, reason_cooking = audio_provider._check_topic_conflict(bathtub_cand, topic="How to prepare vegetable stir-fry")
        self.assertTrue(conflict_cooking)
        self.assertIn("cooking", reason_cooking)

        # In a bathing topic, bathtub sound is accepted (no topic conflict)
        conflict_bathing, _ = audio_provider._check_topic_conflict(bathtub_cand, topic="How to give a baby a gentle bath")
        self.assertFalse(conflict_bathing)


class TestBoundedRecoveryAndCandidateRanking(unittest.IsolatedAsyncioTestCase):
    """Candidate Ranking & Bounded Multi-Candidate Recovery Verification"""

    async def asyncSetUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="test_bound_")
        self.verifier = VisualVerifier()

    async def asyncTearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_candidate_ranking_applies_penalties_and_bonuses(self):
        provider = PexelsVideoProvider()

        scene_ctx = {
            "primary_subject": "fresh vegetables",
            "required_setting": "kitchen stove",
            "required_action": "stir frying",
            "explicit_exclusions": ["bathtub", "bathroom", "meat", "beef"],
            "required_objects": ["wok", "spatula", "vegetables"],
        }
        query_analysis = provider._analyze_query("stir fry cooking vegetables kitchen")

        # Cand A: Contains bathtub (exclusion violation)
        score_a, reasons_a = provider._compute_semantic_score(
            video_text="warm water pouring in white bathroom bathtub with soap bubbles",
            query_analysis=query_analysis,
            raw_query="stir fry cooking",
            scene_context=scene_ctx
        )
        # Explicit exclusion causes -100 penalty, clamped to 0.0 (fails 20.0 threshold)
        self.assertEqual(score_a, 0.0)
        self.assertTrue(any("explicit exclusion disqualification" in r for r in reasons_a))

        # Cand B: Correct kitchen stir fry with required objects
        score_b, reasons_b = provider._compute_semantic_score(
            video_text="chef stir frying fresh vegetables in kitchen wok stove using spatula",
            query_analysis=query_analysis,
            raw_query="stir fry cooking",
            scene_context=scene_ctx
        )
        self.assertGreaterEqual(score_b, 50.0)
        self.assertGreater(score_b, score_a + 40.0)

    async def test_02_bounded_recovery_flow_rejects_contradiction_accepts_alternative(self):
        """Simulates Stage 5 candidate evaluation loop."""
        # Candidate 1: Contradiction (bathtub)
        vid_1 = os.path.join(self.tmp_dir, "cand_1_bath.mp4")
        create_test_video(vid_1, base_color=(190, 210, 230), is_dynamic=True)
        meta_1 = {"title": "Bath salts pouring into bathtub", "tags": "bathtub bathroom"}

        # Candidate 2: Confirmed match (wok vegetables)
        vid_2 = os.path.join(self.tmp_dir, "cand_2_wok.mp4")
        create_test_video(vid_2, base_color=(40, 170, 50), is_dynamic=True)
        meta_2 = {"title": "Stir frying fresh vegetables in hot wok on kitchen stove", "tags": "kitchen cooking vegetables wok stir fry"}

        shot_reqs = {
            "primary_subject": "vegetables",
            "required_visible_action": "stir frying in wok",
            "required_setting": "kitchen stove",
            "explicit_exclusions": ["bathtub", "bathroom", "meat"],
        }

        candidates = [
            (vid_1, meta_1),
            (vid_2, meta_2),
        ]

        accepted_clip = None
        for path, meta in candidates:
            res = await self.verifier.verify_candidate_clip(path, shot_reqs, meta)
            if res.status == VerificationStatus.REJECTED:
                continue
            elif res.status in (VerificationStatus.ACCEPTED, VerificationStatus.UNCERTAIN):
                accepted_clip = (path, res)
                break

        self.assertIsNotNone(accepted_clip)
        self.assertEqual(accepted_clip[0], vid_2)
        self.assertEqual(accepted_clip[1].status, VerificationStatus.ACCEPTED)

    async def test_03_all_candidates_contradict_marked_unresolved_never_substitute(self):
        """When all candidate clips contradict the requirement, none are accepted."""
        vid_1 = os.path.join(self.tmp_dir, "bad_1.mp4")
        create_test_video(vid_1, base_color=(200, 50, 50), is_dynamic=True)
        meta_1 = {"title": "Raw beef steak", "tags": "beef meat"}

        vid_2 = os.path.join(self.tmp_dir, "bad_2.mp4")
        create_test_video(vid_2, base_color=(200, 200, 220), is_dynamic=True)
        meta_2 = {"title": "Bathtub soap", "tags": "bathtub bathroom"}

        shot_reqs = {
            "primary_subject": "vegetables",
            "required_visible_action": "cooking vegetables",
            "required_setting": "kitchen",
            "explicit_exclusions": ["meat", "beef", "bathtub", "bathroom"],
        }

        candidates = [(vid_1, meta_1), (vid_2, meta_2)]
        accepted_clip = None
        for path, meta in candidates:
            res = await self.verifier.verify_candidate_clip(path, shot_reqs, meta)
            if res.status != VerificationStatus.REJECTED:
                accepted_clip = (path, res)
                break

        # Must not silently accept any contradicted clip
        self.assertIsNone(accepted_clip)


if __name__ == "__main__":
    unittest.main()
