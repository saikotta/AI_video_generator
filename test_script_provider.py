import asyncio
import sys
from app.services.providers.ollama_script_provider import OllamaScriptProvider

async def run_test_case(case_name: str, topic: str, duration_seconds: int, wpm: int = 150):
    provider = OllamaScriptProvider(words_per_minute=wpm)
    target_words = provider.calculate_target_words(duration_seconds)
    min_pass_words = int(target_words * 0.90)
    max_pass_words = int(target_words * 1.15)

    print(f"\n==========================================")
    print(f"CASE {case_name}: {topic} -- {duration_seconds} seconds -- {wpm} WPM")
    print(f"==========================================")

    script = await provider.generate_script(topic, duration_seconds)
    actual_word_count = len(script.split())
    estimated_duration = round((actual_word_count / wpm) * 60, 2)
    validation_passed = min_pass_words <= actual_word_count <= max_pass_words

    expansion_occurred = getattr(provider, 'last_expansion_occurred', False)
    trimming_occurred = getattr(provider, 'last_trimming_occurred', False)

    print(f"Target Words: {target_words}")
    print(f"90% Floor (Minimum): {min_pass_words}")
    print(f"115% Ceiling (Maximum): {max_pass_words}")
    print(f"Actual Final Word Count: {actual_word_count}")
    print(f"Estimated Narration Duration: {estimated_duration} seconds")
    print(f"Expansion Occurred: {'YES' if expansion_occurred else 'NO'}")
    print(f"Trimming Occurred: {'YES' if trimming_occurred else 'NO'}")
    print(f"Symmetric Duration Validated: {'YES PASSED' if validation_passed else 'NO FAILED'}")
    print(f"Validation Result: {'PASSED' if validation_passed else 'FAILED'}")

    return validation_passed, target_words, min_pass_words, max_pass_words, actual_word_count, estimated_duration, expansion_occurred, trimming_occurred

async def main():
    print("Executing Script Provider Duration Validation Suite...\n")
    
    pass_a, target_a, min_a, max_a, actual_a, est_a, exp_a, trim_a = await run_test_case(
        "A", "the history of coffee", 120, 150
    )
    pass_b, target_b, min_b, max_b, actual_b, est_b, exp_b, trim_b = await run_test_case(
        "B", "the history of coffee", 300, 150
    )

    print("\n==========================================")
    print("FINAL SUMMARY REPORT")
    print("==========================================")
    print(f"Case A (120s): target={target_a}, 90% floor={min_a}, 115% ceiling={max_a}, actual={actual_a}, est_dur={est_a}s, expanded={exp_a}, trimmed={trim_a}, result={'PASSED' if pass_a else 'FAILED'}")
    print(f"Case B (300s): target={target_b}, 90% floor={min_b}, 115% ceiling={max_b}, actual={actual_b}, est_dur={est_b}s, expanded={exp_b}, trimmed={trim_b}, result={'PASSED' if pass_b else 'FAILED'}")

    if pass_a and pass_b:
        print("\nSUCCESS: BOTH Case A and Case B satisfy 90% <= actual words <= 115%. Step 2 verification PASSED.")
    else:
        print("\nFAILURE: One or more test cases did not satisfy 90% <= actual words <= 115%. Step 2 verification FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
