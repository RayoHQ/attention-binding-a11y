"""Tests for behavioral evaluation scoring."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scoring import score_recognition, score_generation, evaluate_output, KEYWORDS, KEYWORD_THRESHOLD


def test_recognition_scoring():
    """Test recognition scoring with various output formats."""
    # Direct letter match
    assert score_recognition("A", "A") is True
    assert score_recognition("B", "A") is False

    # Explicit answer patterns
    assert score_recognition("The answer is A", "A") is True
    assert score_recognition("Option A", "A") is True
    assert score_recognition("choice: B", "B") is True

    # Letter with punctuation
    assert score_recognition("A)", "A") is True
    assert score_recognition("A.", "A") is True

    # Wrong answer
    assert score_recognition("C", "A") is False
    assert score_recognition("The answer is D", "A") is False

    # Empty / gibberish
    assert score_recognition("", "A") is False
    assert score_recognition("lorem ipsum", "A") is False

    print("✅ Recognition scoring tests passed")


def test_generation_scoring():
    """Test generation scoring with keyword rubric."""
    # High relevance for screen reader
    score1 = score_generation(
        "A screen reader is software that reads text aloud for blind users",
        "screen reader",
    )
    assert score1 > 0.5, f"Expected high score, got {score1}"

    # Irrelevant text → low score
    score2 = score_generation(
        "The weather is nice today",
        "screen reader",
    )
    assert score2 < 0.1, f"Expected low score, got {score2}"

    # Skip link keywords
    score3 = score_generation(
        "helps keyboard users jump to main content for better navigation",
        "skip link",
    )
    assert score3 > 0.5, f"Expected high score, got {score3}"

    # Alt text keywords
    score4 = score_generation(
        "a description of an image for blind users using a screen reader",
        "alt text",
    )
    assert score4 > 0.5, f"Expected high score, got {score4}"

    # Unknown term → 0
    score5 = score_generation("anything", "unknown_term")
    assert score5 == 0.0, f"Expected 0.0 for unknown term, got {score5}"

    # Word-boundary: "bread" should NOT match "read"
    score6 = score_generation("bread and butter", "screen reader")
    assert score6 == 0.0, f"Expected 0.0 for false substring match, got {score6}"

    # Natural base-model output should score well
    score7 = score_generation(
        "a program that reads text from a screen and displays it",
        "screen reader",
    )
    assert score7 > 0.3, f"Expected decent score for natural output, got {score7}"

    print("✅ Generation scoring tests passed")


def test_generation_contradiction_penalty():
    """Test that contradictions reduce score."""
    # Screen reader text mentioning 'deaf' should get penalized
    score_clean = score_generation(
        "software for blind users to read aloud",
        "screen reader",
    )
    score_penalty = score_generation(
        "software for deaf and blind users to read aloud",
        "screen reader",
    )
    assert score_penalty < score_clean, (
        f"Contradiction penalty not applied: {score_penalty} >= {score_clean}"
    )

    print("✅ Contradiction penalty tests passed")


def test_evaluate_output_recognition():
    """Test main evaluation function for recognition tasks (legacy fallback)."""
    # Without model → falls back to legacy regex
    result = evaluate_output("A", "recognition", "screen reader", "A")
    assert result["is_correct"] is True
    assert result["score"] == 1.0
    assert result["method"] == "exact_match"

    result = evaluate_output("B", "recognition", "screen reader", "A")
    assert result["is_correct"] is False
    assert result["score"] == 0.0

    print("✅ evaluate_output recognition tests passed")


def test_evaluate_output_generation():
    """Test main evaluation function for generation tasks."""
    result = evaluate_output(
        "software for blind users to hear text",
        "generation",
        "screen reader",
    )
    assert result["score"] > 0.0
    assert result["method"] == "keyword_rubric"

    print("✅ evaluate_output generation tests passed")


def test_evaluate_output_invalid_task():
    """Test that invalid task raises ValueError."""
    try:
        evaluate_output("text", "invalid_task", "screen reader")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown task" in str(e)

    print("✅ Invalid task error tests passed")


if __name__ == "__main__":
    test_recognition_scoring()
    test_generation_scoring()
    test_generation_contradiction_penalty()
    test_evaluate_output_recognition()
    test_evaluate_output_generation()
    test_evaluate_output_invalid_task()
    print("\n✅ All behavioral tests passed!")
