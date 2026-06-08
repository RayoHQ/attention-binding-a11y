"""Generate 11 prompts per term for 21 new Tier 1/2/3 accessibility terms (A3)."""

import json
from pathlib import Path

OUTPUT = Path("data/prompts/expanded_terms_tier123.jsonl")

TERMS = {
    # (term, definition_context, user_group, contrast_pair)
    "braille display": ("converts digital text to tactile braille cells", "deaf-blind or blind users who prefer tactile reading", "screen magnifier"),
    "screen magnifier": ("enlarges on-screen content for low vision users", "users with low vision", "screen reader"),
    "voice control": ("lets users navigate devices using spoken commands", "users with motor impairments who cannot use keyboard or mouse", "speech-to-text software"),
    "switch access": ("lets users control devices using one or more physical switches", "users with severe motor impairments", "keyboard navigation"),
    "audio description": ("provides verbal narration of visual content in video", "blind or low vision users watching video", "closed captions"),
    "captions closed": ("provides toggleable synchronized text of spoken dialogue and sounds", "deaf and hard of hearing users", "open captions"),
    "cognitive load": ("measures mental effort required to understand an interface", "users with cognitive and learning disabilities", "visual complexity"),
    "high contrast": ("increases distinction between text and background colors", "users with low vision or light sensitivity", "color contrast ratio"),
    "keyboard shortcut": ("provides key combinations that trigger interface actions quickly", "keyboard-only and motor-impaired users", "mouse gesture"),
    "text resize": ("allows users to increase text size without loss of content", "users with low vision or reading difficulties", "screen magnification"),
    "keyboard navigation": ("enables full interface control using only the keyboard", "keyboard-only, motor-impaired, and screen reader users", "mouse navigation"),
    "focus management": ("controls where keyboard focus moves after user interactions", "keyboard and screen reader users", "visual design"),
    "skip navigation": ("lets users bypass repeated navigation blocks to reach main content", "keyboard and screen reader users", "skip link"),
    "reflow content": ("allows content to reformat into a single column when zoomed", "users with low vision who zoom to 400 percent", "responsive design"),
    "non-text content": ("any content that is not text, such as images, charts, and video", "blind and deaf-blind users who need text alternatives", "text content"),
    "error identification": ("clearly communicates form errors to all users including assistive technology", "screen reader and cognitive disability users", "inline validation"),
    "input purpose": ("programmatically identifies the purpose of form input fields", "users with cognitive disabilities and autofill tools", "placeholder text"),
    "text spacing": ("allows users to increase letter, word, and line spacing without loss of content", "users with dyslexia and low vision", "font size"),
    "live region": ("marks dynamic content areas that screen readers should announce automatically", "screen reader users relying on dynamic updates", "static region"),
    "alert dialog": ("an ARIA dialog that requires immediate user acknowledgment before proceeding", "screen reader users who need modal announcements", "tooltip"),
    "tree grid": ("an ARIA widget combining a hierarchical tree with a data grid", "screen reader users navigating hierarchical tabular data", "data table"),
}


def make_prompts(term, ctx, users, contrast):
    entries = []

    rec_templates = [
        (f"A {term} is primarily used for: "
         f"A) {ctx.capitalize()} "
         f"B) Increasing page loading speed "
         f"C) Enhancing visual aesthetics "
         f"D) Managing server requests",
         "A", [ctx.capitalize(), "Increasing page loading speed", "Enhancing visual aesthetics", "Managing server requests"]),

        (f"A {term} primarily benefits: "
         f"A) {users.capitalize()} "
         f"B) Network administrators "
         f"C) Database engineers "
         f"D) Graphic designers only",
         "A", [users.capitalize(), "Network administrators", "Database engineers", "Graphic designers only"]),

        (f"True or False: A {term} is a recognized web accessibility technique that benefits disabled users.",
         "True", ["True", "False"]),

        (f"Which practice improves {term} compatibility: "
         f"A) Following WCAG guidelines and semantic markup "
         f"B) Using proprietary frameworks only "
         f"C) Disabling assistive technology hooks "
         f"D) Minimizing HTML structure",
         "A", ["Following WCAG guidelines and semantic markup", "Using proprietary frameworks only", "Disabling assistive technology hooks", "Minimizing HTML structure"]),

        (f"A {term} differs from a {contrast} because: "
         f"A) It addresses a distinct accessibility need through a different mechanism "
         f"B) They are identical tools "
         f"C) It only works on mobile devices "
         f"D) It requires no configuration",
         "A", ["It addresses a distinct accessibility need through a different mechanism", "They are identical tools", "It only works on mobile devices", "It requires no configuration"]),
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
