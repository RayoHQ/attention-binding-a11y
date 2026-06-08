"""Scoring functions for behavioral probes.

v2: Log-probability recognition scoring for base (non-instruction-tuned) models,
    word-boundary keyword matching, and expanded keyword lists.
"""

import re
from typing import Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# Recognition scoring
# ---------------------------------------------------------------------------

def score_recognition_logprob(
    model,
    prompt: str,
    choices: List[str],
    answer_idx: int,
) -> Dict:
    """Score MCQ by comparing full-sequence log-probabilities of each choice.

    For each candidate answer we compute:
        log P(choice | prompt) = Σ log P(token_i | prompt, token_0..i-1)

    This is the standard approach used by lm-eval-harness for evaluating
    base (non-instruction-tuned) language models on multiple-choice tasks.

    Args:
        model: HookedTransformer
        prompt: Full prompt text (question + choices listed)
        choices: List of answer texts ["Blind users", "Colorblind users", ...]
        answer_idx: Index of correct answer (0=A, 1=B, etc.)

    Returns:
        Dict with predicted_idx, is_correct, score, and per-choice log_probs.
    """
    prompt_tokens = model.to_tokens(prompt)          # [1, prompt_len]
    prompt_len = prompt_tokens.shape[1]

    choice_scores: List[float] = []
    for choice in choices:
        # Tokenize " <choice>" (leading space for proper BPE alignment)
        choice_token_ids = model.tokenizer.encode(
            " " + choice, add_special_tokens=False
        )
        choice_tensor = torch.tensor(
            [choice_token_ids], device=prompt_tokens.device
        )

        # Concatenate prompt + choice tokens → single forward pass
        full_tokens = torch.cat([prompt_tokens, choice_tensor], dim=1)

        with torch.no_grad():
            logits = model(full_tokens)  # [1, seq_len, vocab_size]
            log_probs = torch.log_softmax(logits, dim=-1)

        # Sum log-probs for each choice token.
        # Token at position t predicts token at position t+1,
        # so log P(choice_token_i) is at position (prompt_len - 1 + i).
        total_lp = 0.0
        for i, tok_id in enumerate(choice_token_ids):
            pos = prompt_len - 1 + i          # logit position that predicts tok_id
            total_lp += log_probs[0, pos, tok_id].item()

        # Length-normalize to avoid penalising longer choices
        choice_scores.append(total_lp / len(choice_token_ids))

    predicted_idx = int(torch.tensor(choice_scores).argmax())

    return {
        "predicted_idx": predicted_idx,
        "is_correct": predicted_idx == answer_idx,
        "score": 1.0 if predicted_idx == answer_idx else 0.0,
        "log_probs": choice_scores,
        "method": "logprob_rank",
    }


def score_recognition(text_out: str, answer: str) -> bool:
    """Legacy regex scoring (kept as fallback for tests)."""
    patterns = [
        r"(?:answer|option|choice)[:\s]*([A-D])",
        r"^([A-D])[).:]",
        r"([A-D])\s*$",
    ]
    text_clean = text_out.strip().upper()
    for pattern in patterns:
        match = re.search(pattern, text_clean)
        if match:
            return match.group(1) == answer.upper()
    for letter in ["A", "B", "C", "D"]:
        if text_clean.startswith(letter):
            return letter == answer.upper()
    return False


# ---------------------------------------------------------------------------
# Generation scoring
# ---------------------------------------------------------------------------

# Expanded keyword lists: include both technical terms and natural base-model
# completions so that valid outputs from untrained → fully-trained models
# receive a non-trivial score.
KEYWORDS = {
    "screen reader": [
        # Technical / definitional
        "blind", "visual", "impairment", "aloud", "software",
        "assistive", "technology", "voice", "synthesize",
        # Natural base-model completions
        "text", "screen", "program", "tool", "navigate", "web",
        "display", "content", "user", "accessibility", "computer",
        "speak", "audio", "output", "interface", "information", "read",
    ],
    "skip link": [
        "jump", "skip", "main", "content", "keyboard", "navigation",
        "accessibility", "bypass", "header", "anchor", "link", "page",
        "section", "heading", "tab", "focus", "move", "quick", "direct",
    ],
    "alt text": [
        "image", "description", "alternative", "blind", "screen reader",
        "visual", "impairment", "describe", "picture", "photo", "text",
        "equivalent", "accessibility", "graphic", "content", "meaning",
        "context", "convey", "represent", "display", "element",
    ],
    "focus indicator": [
        "keyboard", "focus", "visual", "highlight", "outline", "border",
        "navigation", "visibility", "show", "indicate", "ring", "marker",
        "cue", "position", "current", "active", "element", "tab", "accessibility",
    ],
    "color contrast": [
        "ratio", "readability", "text", "background", "foreground", "vision",
        "perceive", "distinguish", "wcag", "luminance", "brightness", "difference",
        "low vision", "visibility", "accessible", "sufficient", "minimum", "standard",
    ],
    "aria attribute": [
        "accessible", "name", "attribute", "screen reader", "announce",
        "describe", "element", "aria", "label", "text", "alternative",
        "semantic", "assistive", "technology", "read", "identify", "purpose",
    ],
    "tab order": [
        "keyboard", "navigation", "sequence", "focus", "tab", "order",
        "logical", "flow", "move", "traverse", "accessibility", "tabindex",
        "position", "next", "previous", "structure", "hierarchy", "predictable",
    ],
    "form validation": [
        "error", "input", "field", "message", "feedback", "warning",
        "invalid", "required", "check", "verify", "alert", "notification",
        "accessible", "announce", "screen reader", "identify", "communicate",
    ],
    "heading structure": [
        "markup", "hierarchy", "heading", "tag", "element", "header",
        "nav", "organization", "outline", "section", "h1", "h2", "accessible",
        "assistive", "technology", "screen reader", "navigation", "levels",
    ],
    # --- Tier 1: AT hardware / core concepts ---
    "braille display": [
        "braille", "tactile", "cells", "blind", "deaf-blind", "refreshable",
        "pins", "output", "touch", "read", "screen reader", "assistive",
        "technology", "characters", "dots", "finger", "line", "device",
    ],
    "screen magnifier": [
        "magnify", "enlarge", "zoom", "low vision", "scale", "increase",
        "size", "visual", "impairment", "interface", "display", "desktop",
        "software", "assistive", "technology", "accessibility", "bigger", "text",
    ],
    "voice control": [
        "voice", "speech", "command", "speak", "motor", "hands-free",
        "dictate", "navigate", "control", "microphone", "accessibility",
        "impairment", "disability", "software", "assistive", "recognition",
        "input", "interface",
    ],
    "switch access": [
        "switch", "scan", "button", "motor", "impairment", "disability",
        "single", "control", "select", "activate", "interface", "assistive",
        "technology", "accessibility", "severe", "navigate", "device", "input",
    ],
    "audio description": [
        "audio", "description", "narrate", "blind", "visual", "video",
        "scene", "describe", "action", "content", "accessibility", "wcag",
        "visual impairment", "media", "spoken", "track", "information", "image",
    ],
    "captions closed": [
        "caption", "subtitle", "deaf", "hard of hearing", "text", "dialogue",
        "synchronized", "toggle", "video", "media", "accessibility", "hearing",
        "impairment", "transcript", "sound", "speech", "closed", "display",
    ],
    "cognitive load": [
        "cognitive", "mental", "effort", "load", "process", "understand",
        "complexity", "disability", "learning", "simplify", "clear", "interface",
        "memory", "attention", "design", "user", "accessibility", "reduce",
    ],
    "high contrast": [
        "contrast", "high", "dark", "light", "mode", "visibility", "low vision",
        "color", "background", "foreground", "text", "theme", "setting",
        "accessibility", "perceive", "distinguish", "display", "visual",
    ],
    "keyboard shortcut": [
        "keyboard", "shortcut", "key", "combination", "hotkey", "motor",
        "accessibility", "navigate", "command", "press", "accelerator",
        "efficiency", "input", "control", "function", "action", "binding",
    ],
    "text resize": [
        "text", "resize", "zoom", "font", "size", "increase", "low vision",
        "scale", "accessible", "wcag", "browser", "user", "setting", "reflow",
        "readability", "enlarge", "content", "display",
    ],
    # --- Tier 2: WCAG 2.2 Success Criteria ---
    "keyboard navigation": [
        "keyboard", "navigate", "tab", "focus", "key", "accessible",
        "motor", "impairment", "without mouse", "interface", "interact",
        "sequential", "order", "element", "traverse", "control", "accessibility",
    ],
    "focus management": [
        "focus", "manage", "keyboard", "screen reader", "modal", "dialog",
        "return", "move", "set", "element", "accessible", "interaction",
        "trap", "restore", "dynamic", "update", "visibility", "navigation",
    ],
    "skip navigation": [
        "skip", "navigation", "bypass", "main", "content", "keyboard",
        "link", "accessibility", "repetitive", "header", "tab", "anchor",
        "focus", "jump", "landmark", "screen reader", "menu", "shortcut",
    ],
    "reflow content": [
        "reflow", "content", "zoom", "400", "single column", "horizontal",
        "scroll", "low vision", "wcag", "responsive", "resize", "layout",
        "wrap", "adapt", "viewport", "mobile", "accessible", "text",
    ],
    "non-text content": [
        "image", "alternative", "alt", "text", "description", "graphic",
        "icon", "chart", "diagram", "blind", "visual", "screen reader",
        "equivalent", "wcag", "accessible", "media", "decoration", "content",
    ],
    "error identification": [
        "error", "identify", "message", "input", "form", "field", "accessible",
        "screen reader", "announce", "describe", "invalid", "feedback",
        "wcag", "communication", "user", "correction", "label", "notification",
    ],
    "input purpose": [
        "input", "purpose", "autocomplete", "autofill", "field", "form",
        "cognitive", "disability", "identify", "wcag", "name", "email",
        "address", "programmatic", "context", "label", "semantic", "meaning",
    ],
    "text spacing": [
        "spacing", "letter", "word", "line", "text", "readable", "dyslexia",
        "low vision", "wcag", "increase", "adjust", "override", "stylesheet",
        "content", "loss", "accessible", "legibility", "paragraph",
    ],
    # --- Tier 3: WAI-ARIA roles ---
    "live region": [
        "live", "region", "dynamic", "announce", "screen reader", "aria",
        "update", "content", "assertive", "polite", "notification", "alert",
        "change", "status", "accessible", "real-time", "message", "dom",
    ],
    "alert dialog": [
        "alert", "dialog", "modal", "focus", "screen reader", "aria",
        "announce", "role", "keyboard", "trap", "dismiss", "accessible",
        "message", "user", "confirmation", "action", "interrupt", "interactive",
    ],
    "tree grid": [
        "tree", "grid", "hierarchical", "tabular", "data", "aria", "role",
        "row", "column", "expand", "collapse", "screen reader", "navigate",
        "keyboard", "accessible", "table", "structure", "widget",
    ],
    # --- Wave-2: Mobile, Vestibular, Sensory, Language ---
    "contrast ratio": [
        "ratio", "wcag", "4.5", "3:1", "text", "background", "luminance",
        "foreground", "brightness", "aa", "aaa", "threshold", "minimum",
        "perceive", "readability", "accessible", "standard", "contrast", "difference",
    ],
    "eye tracking": [
        "eye", "gaze", "tracking", "pointer", "motor", "disability", "control",
        "interface", "impairment", "input", "accessibility", "movement", "cursor",
        "technology", "alternative", "device", "vision",
    ],
    "time limits": [
        "time", "limit", "timeout", "session", "extend", "adjust", "pause",
        "wcag", "accessible", "warning", "deadline", "content", "expire",
        "control", "keyboard", "disable", "duration",
    ],
    "reduced motion": [
        "motion", "animation", "vestibular", "seizure", "prefers-reduced-motion",
        "disable", "reduce", "css", "media", "query", "accessible", "parallax",
        "flicker", "disorder", "trigger", "spinning",
    ],
    "focus trap": [
        "focus", "trap", "modal", "dialog", "keyboard", "accessible", "cycle",
        "escape", "dismiss", "contain", "widget", "interaction", "confine",
        "loop", "overlay",
    ],
    "sign language": [
        "sign", "language", "deaf", "interpreter", "video", "wcag", "asl", "bsl",
        "visual", "communication", "accessibility", "hearing", "caption",
        "translation", "media",
    ],
    "touch target size": [
        "touch", "target", "size", "tap", "minimum", "44px", "44",
        "mobile", "accessible", "wcag", "button", "clickable", "pointer",
        "spacing", "area", "interactive", "pixel",
    ],
    "haptic feedback": [
        "haptic", "vibration", "touch", "tactile", "feedback", "motor",
        "accessible", "notification", "response", "physical", "device", "sensory",
    ],
    "plain language": [
        "plain", "language", "simple", "clear", "readable", "cognitive",
        "disability", "understandable", "wcag", "jargon", "comprehension",
        "accessibility", "audience", "writing", "complex",
    ],
    "motion sensitivity": [
        "motion", "sensitivity", "vestibular", "disorder", "animation",
        "movement", "trigger", "dizzy", "accessible", "wcag", "reduce",
        "seizure", "sensitive", "physical",
    ],
    "semantic html": [
        "semantic", "html", "markup", "meaning", "structure", "screen reader",
        "element", "accessible", "roles", "heading", "landmark", "assistive",
        "technology", "interpret", "tag",
    ],
    "orientation support": [
        "orientation", "rotate", "landscape", "portrait", "screen", "lock",
        "wcag", "accessible", "mobile", "device", "direction", "restrict",
        "cognitive", "disability",
    ],
}


CONTRADICTIONS = {
    "screen reader": ["deaf", "hearing", "colorblind", "see", "look"],
    "skip link": ["advertisement", "ad", "popup", "slow", "delay"],
    "alt text": ["video", "audio", "caption", "subtitle", "sound"],
    "focus indicator": ["mouse", "touch", "pointer", "click", "hover"],
    "color contrast": ["audio", "sound", "hearing", "voice", "speak"],
    "aria attribute": ["visual", "appearance", "style", "color", "layout"],
    "tab order": ["mouse", "click", "touch", "swipe", "drag"],
    "form validation": ["automatic", "silent", "hidden", "invisible", "suppress"],
    "heading structure": ["javascript", "css", "styling", "appearance", "decoration", "flat"],
}

# A "good" answer is expected to hit at least this many keywords.
KEYWORD_THRESHOLD = 3


def score_generation(text_out: str, term: str) -> float:
    """Score generation task using word-boundary keyword rubric.

    Returns score between 0 and 1.
    """
    text_lower = text_out.lower()
    term_lower = term.lower()

    term_keywords = KEYWORDS.get(term_lower, [])
    if not term_keywords:
        return 0.0

    # Word-boundary matching (prevents "bread" matching "read")
    matches = 0
    for kw in term_keywords:
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
            matches += 1

    # Normalize: reaching KEYWORD_THRESHOLD keywords → 1.0
    score = min(1.0, matches / KEYWORD_THRESHOLD)

    # Contradiction penalty
    term_contradictions = CONTRADICTIONS.get(term_lower, [])
    for c in term_contradictions:
        if re.search(r"\b" + re.escape(c) + r"\b", text_lower):
            score -= 0.2

    return max(0.0, score)


# ---------------------------------------------------------------------------
# Main evaluation entry-point
# ---------------------------------------------------------------------------

def evaluate_output(
    text_out: str,
    task: str,
    term: str,
    answer: Optional[str] = None,
    model=None,
    prompt: Optional[str] = None,
    choices: Optional[List[str]] = None,
    answer_idx: Optional[int] = None,
) -> Dict:
    """Main evaluation function.

    For recognition tasks, uses log-probability ranking when *model* is
    provided, otherwise falls back to legacy regex matching.

    Args:
        text_out: Model-generated text (used for generation; informational
                  for recognition when log-prob mode is active).
        task: "recognition" or "generation"
        term: The accessibility term being tested.
        answer: Correct answer letter for legacy recognition scoring.
        model: HookedTransformer (required for log-prob recognition).
        prompt: Full prompt string (required for log-prob recognition).
        choices: List of choice texts (required for log-prob recognition).
        answer_idx: 0-indexed correct choice (required for log-prob recognition).

    Returns:
        Dict with score, is_correct, and method.
    """
    if task == "recognition":
        if model is not None and choices is not None and answer_idx is not None:
            result = score_recognition_logprob(
                model, prompt, choices, answer_idx
            )
            return {
                "is_correct": result["is_correct"],
                "score": result["score"],
                "method": result["method"],
                "predicted_idx": result["predicted_idx"],
                "log_probs": result["log_probs"],
            }
        # Fallback: legacy regex matching
        is_correct = score_recognition(text_out, answer)
        return {
            "is_correct": is_correct,
            "score": 1.0 if is_correct else 0.0,
            "method": "exact_match",
        }
    elif task == "generation":
        score = score_generation(text_out, term)
        return {
            "is_correct": score > 0.5,
            "score": round(score, 4),
            "method": "keyword_rubric",
        }
    else:
        raise ValueError(f"Unknown task: {task}")
