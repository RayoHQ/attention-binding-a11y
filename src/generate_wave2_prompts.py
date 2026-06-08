"""Generate 11 prompts per term for 12 Wave-2 accessibility terms.

Wave-2 terms fill coverage gaps identified in the term audit:
- UPGRADE:    contrast ratio (replaces color contrast for C1/C4)
- STRONG ADD: eye tracking, time limits, reduced motion, focus trap,
              sign language, touch target size
- GOOD ADD:   haptic feedback, plain language, motion sensitivity,
              semantic html, orientation support

New categories covered: Vestibular, Mobile, Sensory, Hearing/Sign.
"""

import json
from pathlib import Path

OUTPUT = Path("data/prompts/expanded_terms_wave2.jsonl")

# (term, definition_context, user_group, contrast_pair)
TERMS = {
    "contrast ratio": (
        "measures the luminance difference between foreground and background colors",
        "users with low vision and color perception difficulties",
        "color contrast adjustment",
    ),
    "eye tracking": (
        "lets users control devices using gaze direction and eye movement",
        "users with severe motor impairments who cannot use hands or voice",
        "voice control",
    ),
    "time limits": (
        "restricts how long users have to complete a task before a session timeout",
        "users with cognitive disabilities, slow readers, and motor impairments",
        "session persistence",
    ),
    "reduced motion": (
        "limits or eliminates animations and transitions to prevent vestibular discomfort",
        "users with vestibular disorders and motion sensitivity",
        "animation effects",
    ),
    "focus trap": (
        "restricts keyboard focus to within a component such as a modal dialog",
        "keyboard and screen reader users navigating modal dialogs",
        "focus management",
    ),
    "sign language": (
        "uses hand gestures and visual signals to communicate for deaf users",
        "deaf users whose primary language is a signed language",
        "closed captions",
    ),
    "touch target size": (
        "defines the minimum physical size of interactive elements for reliable touch input",
        "users with motor impairments using touch screens on mobile devices",
        "mouse click target",
    ),
    "haptic feedback": (
        "provides tactile vibration responses to confirm user interactions",
        "users with visual or hearing impairments who need non-visual confirmation",
        "audio feedback",
    ),
    "plain language": (
        "uses simple vocabulary and sentence structure for easy comprehension",
        "users with cognitive disabilities and low literacy levels",
        "technical jargon",
    ),
    "motion sensitivity": (
        "refers to discomfort or nausea triggered by screen animations or movement",
        "users with vestibular disorders or migraines triggered by motion",
        "reduced motion",
    ),
    "semantic html": (
        "uses meaningful HTML elements that convey content structure and purpose",
        "screen reader users and assistive technology that relies on document structure",
        "generic div layout",
    ),
    "orientation support": (
        "ensures content displays correctly in both portrait and landscape screen modes",
        "users who cannot physically rotate their device due to disability or mounting",
        "fixed orientation design",
    ),
}


def make_prompts(term, ctx, users, contrast):
    entries = []

    rec_templates = [
        (
            f"A {term} is primarily used for: "
            f"A) {ctx.capitalize()} "
            f"B) Increasing page loading speed "
            f"C) Enhancing visual aesthetics "
            f"D) Managing server requests",
            "A",
            [ctx.capitalize(), "Increasing page loading speed", "Enhancing visual aesthetics", "Managing server requests"],
        ),
        (
            f"A {term} primarily benefits: "
            f"A) {users.capitalize()} "
            f"B) Network administrators "
            f"C) Database engineers "
            f"D) Graphic designers only",
            "A",
            [users.capitalize(), "Network administrators", "Database engineers", "Graphic designers only"],
        ),
        (
            f"True or False: A {term} is a recognized web accessibility technique that benefits disabled users.",
            "True",
            ["True", "False"],
        ),
        (
            f"Which practice improves {term} compatibility: "
            f"A) Following WCAG guidelines and semantic markup "
            f"B) Using proprietary frameworks only "
            f"C) Disabling assistive technology hooks "
            f"D) Minimizing HTML structure",
            "A",
            ["Following WCAG guidelines and semantic markup", "Using proprietary frameworks only", "Disabling assistive technology hooks", "Minimizing HTML structure"],
        ),
        (
            f"A {term} differs from a {contrast} because: "
            f"A) It addresses a distinct accessibility need through a different mechanism "
            f"B) They are identical tools "
            f"C) It only works on mobile devices "
            f"D) It requires no configuration",
            "A",
            ["It addresses a distinct accessibility need through a different mechanism", "They are identical tools", "It only works on mobile devices", "It requires no configuration"],
        ),
    ]

    for i, (tmpl, ans, choices) in enumerate(rec_templates, 1):
        entries.append({
            "term": term,
            "task": "recognition",
            "template": tmpl,
            "answer": ans,
            "choices": choices,
            "answer_idx": 0,
            "prompt_id": f"rec_{i:03d}",
        })

    gen_templates = [
        f"In web accessibility, a {term} is",
        f"For {users.split(' who')[0].split(' and')[0]}, a {term}",
        f"To support {term} compatibility, developers should",
        f"Without proper {term} support, affected users may",
        f"An accessibility audit found the site lacks {term} support. This means",
        f"When building accessible products, {term} ensures",
    ]

    for i, tmpl in enumerate(gen_templates, 1):
        entries.append({
            "term": term,
            "task": "generation",
            "template": tmpl,
            "max_tokens": 20 if i <= 2 else 25,
            "prompt_id": f"gen_{i:03d}",
        })

    return entries


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    all_entries = []
    for term, (ctx, users, contrast) in TERMS.items():
        entries = make_prompts(term, ctx, users, contrast)
        all_entries.extend(entries)
        print(f"  {term}: {len(entries)} prompts")

    with open(OUTPUT, "w") as f:
        for e in all_entries:
            f.write(json.dumps(e) + "\n")

    print(f"\nTotal: {len(all_entries)} prompts across {len(TERMS)} terms")
    print(f"Saved to {OUTPUT}")
