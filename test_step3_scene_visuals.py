import asyncio
import sys
from app.services.providers.ollama_script_provider import OllamaScriptProvider
from app.schemas.scene import SceneListSchema
from app.schemas.visual_spec import VisualSpecificationSchema

async def run_step3_test_case(case_name: str, topic: str, duration_seconds: int, wpm: int = 150):
    provider = OllamaScriptProvider(words_per_minute=wpm)
    target_words = provider.calculate_target_words(duration_seconds)

    print(f"\n==================================================")
    print(f"STEP 3 CASE {case_name}: '{topic}' -- {duration_seconds}s @ {wpm} WPM")
    print(f"==================================================")

    # 1. Generate narration using EXISTING Step 2 generate_script()
    print("[1/5] Generating narration script...")
    script = await provider.generate_script(topic, duration_seconds)
    original_script_words = len(script.split())
    print(f"-> Script generated: {original_script_words} words (Target: {target_words})")

    # 2. Split script into scenes
    print("[2/5] Splitting script into duration-aware scenes...")
    scenes = await provider.split_into_scenes(script, duration_seconds)
    scene_count = len(scenes)
    print(f"-> Generated {scene_count} scenes")

    # 3. Validate scene structure with Pydantic
    print("[3/5] Validating scene list structure with Pydantic...")
    scene_list_data = {"scenes": scenes}
    
    pydantic_valid = False
    scene_order_valid = False
    dup_empty_valid = False

    try:
        validated_scene_list = SceneListSchema.model_validate(scene_list_data)
        pydantic_valid = True
    except Exception as e:
        print(f"FAILED Pydantic Validation: {e}")
        raise

    # Verify sequential order, non-empty, and uniqueness
    for idx, s in enumerate(validated_scene_list.scenes, start=1):
        if s.scene_order != idx:
            print(f"FAILED Scene Order Validation: Scene {idx} has order {s.scene_order}")
            sys.exit(1)
        if not s.scene_text.strip():
            print(f"FAILED Empty Scene Validation: Scene {idx} text is empty")
            sys.exit(1)

    scene_order_valid = True

    seen_texts = set()
    for s in validated_scene_list.scenes:
        norm = s.scene_text.strip().lower()
        if norm in seen_texts:
            print(f"FAILED Duplicate Scene Validation: Duplicate text '{s.scene_text[:30]}...'")
            sys.exit(1)
        seen_texts.add(norm)

    dup_empty_valid = True

    # 4. Programmatic Narration Content Preservation Check (Combined word count >= 90% of original)
    combined_scene_text = " ".join(s.scene_text for s in validated_scene_list.scenes)
    combined_scene_words = len(combined_scene_text.split())
    preservation_percentage = (combined_scene_words / max(1, original_script_words)) * 100
    preservation_valid = combined_scene_words >= (original_script_words * 0.90)

    if not preservation_valid:
        print(f"FAILED Narration Preservation: {combined_scene_words}/{original_script_words} words ({preservation_percentage:.1f}%) < 90.0%")
        sys.exit(1)

    # 5. Generate and validate visual specifications for every scene
    print(f"[4/5] Generating and validating visual specifications for all {scene_count} scenes...")
    visual_specs = []
    visual_spec_valid = True
    import httpx
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        for scene in validated_scene_list.scenes:
            spec_dict = await provider.generate_visual_specification(scene.scene_text, client=client)
            try:
                validated_spec = VisualSpecificationSchema.model_validate(spec_dict)
                visual_specs.append(validated_spec)
            except Exception as e:
                print(f"FAILED Visual Spec Validation for Scene {scene.scene_order}: {e}")
                visual_spec_valid = False
                raise

    # 6. Print complete structured result for EVERY scene
    print("\n--------------------------------------------------")
    print(f"CASE {case_name} STRUCTURED SCENE & VISUAL SPECIFICATIONS")
    print("--------------------------------------------------")
    for scene, spec in zip(validated_scene_list.scenes, visual_specs):
        print(f"Scene Order: {scene.scene_order}")
        print(f"Scene Text: {scene.scene_text}")
        print(f"Visual Prompt: {spec.visual_prompt}")
        print(f"Search Keywords: {spec.search_keywords}")
        print("- - - - - - - - - - - - - - - - - - - - - - - - -")

    avg_words_per_scene = round(original_script_words / max(1, scene_count), 1)
    avg_seconds_per_scene = round(duration_seconds / max(1, scene_count), 1)
    overall_valid = (
        pydantic_valid and scene_order_valid and dup_empty_valid and
        preservation_valid and visual_spec_valid
    )

    print("\n--------------------------------------------------")
    print(f"CASE {case_name} METRICS & VALIDATION SUMMARY")
    print("--------------------------------------------------")
    print(f"Target Duration: {duration_seconds} seconds")
    print(f"Original Script Word Count: {original_script_words} words")
    print(f"Scene Count: {scene_count} scenes")
    print(f"Combined Scene Word Count: {combined_scene_words} words")
    print(f"Narration Preservation Percentage: {preservation_percentage:.1f}%")
    print(f"Average Words Per Scene: {avg_words_per_scene} words/scene")
    print(f"Estimated Average Scene Duration: {avg_seconds_per_scene} seconds/scene")
    print(f"Pydantic Validation Result: {'PASSED' if pydantic_valid else 'FAILED'}")
    print(f"Scene-Order Validation Result: {'PASSED' if scene_order_valid else 'FAILED'}")
    print(f"Duplicate/Empty-Scene Validation Result: {'PASSED' if dup_empty_valid else 'FAILED'}")
    print(f"Narration Preservation Result: {'PASSED' if preservation_valid else 'FAILED'}")
    print(f"Visual-Spec Validation Result: {'PASSED' if visual_spec_valid else 'FAILED'}")
    print(f"Overall Step 3 Validation Result: {'PASSED' if overall_valid else 'FAILED'}")

    return {
        "case_name": case_name,
        "duration_seconds": duration_seconds,
        "original_script_words": original_script_words,
        "combined_scene_words": combined_scene_words,
        "scene_count": scene_count,
        "preservation_percentage": preservation_percentage,
        "avg_words_per_scene": avg_words_per_scene,
        "avg_seconds_per_scene": avg_seconds_per_scene,
        "overall_valid": overall_valid
    }

async def main():
    print("==================================================")
    print("STARTING STANDALONE STEP 3 VERIFICATION SUITE")
    print("==================================================")

    res_a = await run_step3_test_case("A", "the history of coffee", 120, 150)
    res_b = await run_step3_test_case("B", "the history of coffee", 300, 150)

    scenes_a = res_a['scene_count']
    scenes_b = res_b['scene_count']
    more_scenes_check = scenes_b > scenes_a

    print("\n==================================================")
    print("STEP 3 VERIFICATION COMPARATIVE SUMMARY")
    print("==================================================")
    print(f"Case A (120s): {scenes_a} scenes ({res_a['avg_seconds_per_scene']}s/scene) - Result: {'PASSED' if res_a['overall_valid'] else 'FAILED'}")
    print(f"Case B (300s): {scenes_b} scenes ({res_b['avg_seconds_per_scene']}s/scene) - Result: {'PASSED' if res_b['overall_valid'] else 'FAILED'}")
    print(f"Scene Count Scaling Check (300s > 120s): {scenes_b} > {scenes_a} -> {'PASSED' if more_scenes_check else 'FAILED'}")

    if res_a['overall_valid'] and res_b['overall_valid'] and more_scenes_check:
        print("\n==================================================")
        print("SUCCESS: BOTH STEP 3 TEST CASES VERIFIED CLEANLY!")
        print("==================================================")
    else:
        print("\n==================================================")
        print("FAILURE: STEP 3 VERIFICATION FAILED.")
        print("==================================================")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
