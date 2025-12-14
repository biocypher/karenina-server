#!/usr/bin/env python
"""Test script to verify database rubric save/load flow."""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add karenina to path
sys.path.insert(0, str(Path(__file__).parent.parent / "karenina" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from karenina.benchmark import Benchmark
from karenina.storage import DBConfig, init_database, load_benchmark, save_benchmark


def load_checkpoint_file(checkpoint_path: str) -> dict:
    """Load and parse a JSON-LD checkpoint file."""
    with open(checkpoint_path) as f:
        return json.load(f)


def extract_global_rubric_from_jsonld(checkpoint_data: dict) -> dict | None:
    """Extract global rubric from JSON-LD checkpoint format."""
    from karenina.schemas.domain import LLMRubricTrait, RegexTrait

    ratings = checkpoint_data.get("rating", [])
    if not ratings:
        return None

    llm_traits = []
    regex_traits = []
    callable_traits = []
    metric_traits = []

    for rating in ratings:
        additional_type = rating.get("additionalType", "")
        name = rating.get("name", "")
        description = rating.get("description", "")

        if additional_type == "GlobalRubricTrait":
            # LLM trait - determine kind from bestRating/worstRating
            best = rating.get("bestRating", 5)
            worst = rating.get("worstRating", 1)
            is_boolean = (best == 1 and worst == 0) or (best == 0 and worst == 1)

            trait = LLMRubricTrait(
                name=name,
                description=description,
                kind="boolean" if is_boolean else "score",
                min_score=worst if not is_boolean else None,
                max_score=best if not is_boolean else None,
            )
            llm_traits.append(trait)

        elif additional_type == "GlobalRegexTrait":
            # Extract pattern from additionalProperty
            pattern = ".*"
            case_sensitive = True
            invert_result = False

            for prop in rating.get("additionalProperty", []):
                prop_name = prop.get("name", "")
                prop_value = prop.get("value")
                if prop_name == "pattern":
                    pattern = prop_value
                elif prop_name == "case_sensitive":
                    case_sensitive = prop_value
                elif prop_name == "invert_result":
                    invert_result = prop_value

            trait = RegexTrait(
                name=name,
                description=description,
                pattern=pattern,
                case_sensitive=case_sensitive,
                invert_result=invert_result,
            )
            regex_traits.append(trait)

    if not (llm_traits or regex_traits or callable_traits or metric_traits):
        return None

    return {
        "llm_traits": llm_traits,
        "regex_traits": regex_traits,
        "callable_traits": callable_traits,
        "metric_traits": metric_traits,
    }


def main():
    checkpoint_path = (
        "/Users/carli/Projects/karenina-monorepo/local_data/data/checkpoints/karenina_checkpoint_ot_bench_v5.4.jsonld"
    )

    # Create temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    print("=" * 60)
    print("TEST: Database Rubric Save/Load Flow")
    print("=" * 60)
    print(f"\n📁 Checkpoint: {checkpoint_path}")
    print(f"💾 Test DB: {db_path}")

    try:
        # Step 1: Load checkpoint file
        print(f"\n{'=' * 60}")
        print("STEP 1: Load checkpoint file")
        print(f"{'=' * 60}")

        checkpoint_data = load_checkpoint_file(checkpoint_path)
        print(f"✅ Loaded checkpoint: {checkpoint_data.get('name', 'Unknown')}")
        print(f"   Version: {checkpoint_data.get('version', 'Unknown')}")
        print(f"   Questions: {len(checkpoint_data.get('dataFeedElement', []))}")

        # Extract global rubric
        global_rubric_data = extract_global_rubric_from_jsonld(checkpoint_data)
        if global_rubric_data:
            print("   Global rubric traits:")
            print(f"     - LLM traits: {len(global_rubric_data['llm_traits'])}")
            print(f"     - Regex traits: {len(global_rubric_data['regex_traits'])}")
            print(f"     - Callable traits: {len(global_rubric_data['callable_traits'])}")
            print(f"     - Metric traits: {len(global_rubric_data['metric_traits'])}")
        else:
            print("   ⚠️ No global rubric found in checkpoint")

        # Step 2: Initialize database
        print(f"\n{'=' * 60}")
        print("STEP 2: Initialize database")
        print(f"{'=' * 60}")

        db_config = DBConfig(storage_url=f"sqlite:///{db_path}")
        init_database(db_config)
        print("✅ Database initialized")

        # Step 3: Create benchmark from checkpoint
        print(f"\n{'=' * 60}")
        print("STEP 3: Create benchmark and add questions")
        print(f"{'=' * 60}")

        benchmark = Benchmark.create(
            name="test_rubric_benchmark",
            description="Test benchmark for rubric save/load",
            version="1.0.0",
            creator="Test Script",
        )
        print(f"✅ Created benchmark: {benchmark.name}")

        # Add questions from checkpoint (limit to first 5 for testing)
        from karenina.schemas.domain import Question

        questions_added = 0
        for item in checkpoint_data.get("dataFeedElement", [])[:5]:
            question_item = item.get("item", {})
            question_text = question_item.get("text", "")
            answer = question_item.get("acceptedAnswer", {}).get("text", "")
            template = question_item.get("hasPart", {}).get("text", "")

            # Get finished status
            finished = False
            for prop in question_item.get("additionalProperty", []):
                if prop.get("name") == "finished":
                    finished = prop.get("value", False)

            question = Question(
                question=question_text,
                raw_answer=answer,
            )

            benchmark.add_question(
                question=question,
                answer_template=template,
                finished=finished,
            )
            questions_added += 1

        print(f"✅ Added {questions_added} questions")

        # Step 4: Set global rubric
        print(f"\n{'=' * 60}")
        print("STEP 4: Set global rubric")
        print(f"{'=' * 60}")

        if global_rubric_data:
            from karenina.schemas.domain import Rubric

            global_rubric = Rubric(
                llm_traits=global_rubric_data["llm_traits"],
                regex_traits=global_rubric_data["regex_traits"],
                callable_traits=global_rubric_data["callable_traits"],
                metric_traits=global_rubric_data["metric_traits"],
            )
            benchmark.set_global_rubric(global_rubric)
            print("✅ Set global rubric with:")
            print(f"   - {len(global_rubric.llm_traits)} LLM traits")
            print(f"   - {len(global_rubric.regex_traits)} regex traits")
            print(f"   - {len(global_rubric.callable_traits)} callable traits")
            print(f"   - {len(global_rubric.metric_traits)} metric traits")
        else:
            print("⚠️ No global rubric to set")

        # Step 4b: Set question-specific rubric on first question
        print(f"\n{'=' * 60}")
        print("STEP 4b: Set question-specific rubric")
        print(f"{'=' * 60}")

        from karenina.schemas.domain import LLMRubricTrait, Rubric

        question_ids = benchmark.get_question_ids()
        if question_ids:
            first_q_id = question_ids[0]
            question_rubric = Rubric(
                llm_traits=[
                    LLMRubricTrait(
                        name="Test Question Trait",
                        description="A test trait for this specific question",
                        kind="boolean",
                    )
                ],
                regex_traits=[],
                callable_traits=[],
                metric_traits=[],
            )
            benchmark.set_question_rubric(first_q_id, question_rubric)
            print(f"✅ Set question-specific rubric on question: {first_q_id[:20]}...")
            print("   - 1 LLM trait: 'Test Question Trait'")

        # Step 5: Save to database
        print(f"\n{'=' * 60}")
        print("STEP 5: Save benchmark to database")
        print(f"{'=' * 60}")

        save_benchmark(benchmark, db_config)
        print("✅ Saved benchmark to database")

        # Step 6: Verify database contents
        print(f"\n{'=' * 60}")
        print("STEP 6: Verify database contents (raw SQL)")
        print(f"{'=' * 60}")

        import sqlite3

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check benchmarks table
        cursor.execute("SELECT name, metadata_json FROM benchmarks")
        row = cursor.fetchone()
        if row:
            print(f"✅ Benchmark in DB: {row[0]}")
            metadata = json.loads(row[1]) if row[1] else {}
            if metadata.get("global_rubric"):
                gr = metadata["global_rubric"]
                print("   Global rubric in metadata_json:")
                print(f"     - traits (LLM): {len(gr.get('traits', []))}")
                print(f"     - regex_traits: {len(gr.get('regex_traits', []))}")
                print(f"     - callable_traits: {len(gr.get('callable_traits', []))}")
                print(f"     - metric_traits: {len(gr.get('metric_traits', []))}")
            else:
                print("   ❌ No global_rubric in metadata_json!")

        # Check questions
        cursor.execute("SELECT COUNT(*) FROM benchmark_questions")
        q_count = cursor.fetchone()[0]
        print(f"✅ Questions in DB: {q_count}")

        conn.close()

        # Step 7: Load benchmark back and verify
        print(f"\n{'=' * 60}")
        print("STEP 7: Load benchmark back and verify")
        print(f"{'=' * 60}")

        loaded_benchmark = load_benchmark("test_rubric_benchmark", db_config)
        print(f"✅ Loaded benchmark: {loaded_benchmark.name}")
        print(f"   Questions: {len(loaded_benchmark.get_question_ids())}")

        # Check global rubric
        loaded_rubric = loaded_benchmark.get_global_rubric()
        if loaded_rubric:
            print("✅ Global rubric loaded:")
            print(f"   - LLM traits: {len(loaded_rubric.llm_traits)}")
            print(f"   - Regex traits: {len(loaded_rubric.regex_traits)}")
            print(f"   - Callable traits: {len(loaded_rubric.callable_traits)}")
            print(f"   - Metric traits: {len(loaded_rubric.metric_traits)}")

            # Print trait names
            if loaded_rubric.llm_traits:
                print(f"   LLM trait names: {[t.name for t in loaded_rubric.llm_traits]}")
            if loaded_rubric.regex_traits:
                print(f"   Regex trait names: {[t.name for t in loaded_rubric.regex_traits]}")
        else:
            print("❌ No global rubric loaded!")

        # Check question-specific rubric
        loaded_q_ids = loaded_benchmark.get_question_ids()
        if loaded_q_ids:
            first_loaded_q_id = loaded_q_ids[0]
            # Access question rubric via the internal cache
            q_data = loaded_benchmark.get_question(first_loaded_q_id)
            loaded_q_rubric = q_data.get("question_rubric")
            if loaded_q_rubric:
                print(f"✅ Question-specific rubric loaded for {first_loaded_q_id[:20]}...")
                # The rubric is stored as a dict with llm_traits, regex_traits, etc.
                if isinstance(loaded_q_rubric, dict):
                    llm_count = len(loaded_q_rubric.get("llm_traits", []))
                    print(f"   - LLM traits: {llm_count}")
                elif isinstance(loaded_q_rubric, list):
                    print(f"   - Traits (list): {len(loaded_q_rubric)}")
            else:
                print(f"❌ No question-specific rubric loaded for {first_loaded_q_id[:20]}...")

        # Step 8: Test API endpoint simulation
        print(f"\n{'=' * 60}")
        print("STEP 8: Test API endpoint format (simulated)")
        print(f"{'=' * 60}")

        # Simulate rubric serialization (same logic as _serialize_rubric_to_dict)
        if loaded_rubric:
            serialized = {
                "llm_traits": [t.model_dump() for t in loaded_rubric.llm_traits],
                "regex_traits": [t.model_dump() for t in loaded_rubric.regex_traits],
                "callable_traits": [t.model_dump() for t in loaded_rubric.callable_traits],
                "metric_traits": [t.model_dump() for t in loaded_rubric.metric_traits],
            }
            print("✅ Rubric serialized to API format:")
            print(f"   Keys: {list(serialized.keys())}")
            print(f"   llm_traits count: {len(serialized.get('llm_traits', []))}")
            print(f"   regex_traits count: {len(serialized.get('regex_traits', []))}")

            # Verify the format matches what the GUI expects
            if "llm_traits" in serialized:
                print("✅ Uses 'llm_traits' key (correct)")
            if "traits" in serialized:
                print("⚠️ Also has 'traits' key (old format)")
        else:
            print("❌ Failed to serialize rubric")

        # Final summary
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")

        original_llm = len(global_rubric_data["llm_traits"]) if global_rubric_data else 0
        original_regex = len(global_rubric_data["regex_traits"]) if global_rubric_data else 0
        loaded_llm = len(loaded_rubric.llm_traits) if loaded_rubric else 0
        loaded_regex = len(loaded_rubric.regex_traits) if loaded_rubric else 0

        all_passed = True

        if original_llm == loaded_llm:
            print(f"✅ LLM traits: {original_llm} saved, {loaded_llm} loaded")
        else:
            print(f"❌ LLM traits MISMATCH: {original_llm} saved, {loaded_llm} loaded")
            all_passed = False

        if original_regex == loaded_regex:
            print(f"✅ Regex traits: {original_regex} saved, {loaded_regex} loaded")
        else:
            print(f"❌ Regex traits MISMATCH: {original_regex} saved, {loaded_regex} loaded")
            all_passed = False

        if all_passed:
            print("\n🎉 ALL TESTS PASSED!")
        else:
            print("\n💥 SOME TESTS FAILED!")
            sys.exit(1)

    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.remove(db_path)
            print("\n🧹 Cleaned up temp database")


if __name__ == "__main__":
    main()
