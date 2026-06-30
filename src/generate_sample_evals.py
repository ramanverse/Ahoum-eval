"""
generate_sample_evals.py — Generate data/examples/sample_evaluations_50.json

50 diverse pre-computed conversations with 300 facet scores each.
Run this script once to populate the examples directory.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Inline mini-facet definitions (subset of 300, used for sample generation)
# Used here to avoid I/O dependency during generation
# ---------------------------------------------------------------------------

SAMPLE_FACETS_BY_CATEGORY = {
    "Linguistic Quality": [
        "Grammar Correctness", "Vocabulary Appropriateness", "Sentence Clarity",
        "Readability", "Lexical Diversity", "Sentence Structure", "Brevity",
        "Conciseness", "Spelling Accuracy", "Language Use",
    ],
    "Pragmatics": [
        "Politeness", "Relevance", "Informativeness", "Context Awareness",
        "Directness", "Formality", "Response Coherence", "Talkativeness",
        "Appropriateness", "Collaboration",
    ],
    "Safety": [
        "Harmlessness", "Disrespect Avoidance", "Ethical Standards",
        "Safety Compliance", "Hostility Avoidance", "Civility",
        "Honesty", "Privacy Protection", "Conflict Avoidance", "Hatefulness Absence",
    ],
    "Emotion & Empathy": [
        "Empathy", "Compassion", "Emotional Awareness", "Warmheartedness",
        "Joyfulness", "Sensitivity", "Affection", "Emotional Boundaries",
        "Comfort with Vulnerability", "Blissfulness",
    ],
    "Behavioral Traits": [
        "Assertiveness", "Confidence", "Honesty", "Persistence", "Adaptability",
        "Openness", "Self-Control", "Independence", "Determinedness", "Genuine",
    ],
    "Intelligence & Reasoning": [
        "Common-sense", "Logical Reasoning", "Creativity", "Analytical Thinking",
        "Critical Reasoning", "Statistical Reasoning", "Originality",
        "Synthesis of Information", "Problem Solving", "Economic Reasoning",
    ],
}

ALL_SAMPLE_FACETS = [
    f for facets in SAMPLE_FACETS_BY_CATEGORY.values() for f in facets
]

# Quality → score range mapping
QUALITY_RANGES = {
    "high":      (4, 5),
    "medium":    (3, 4),
    "low":       (1, 2),
    "edge_case": (1, 5),  # wide variance
}

SAMPLE_CONVERSATIONS = {
    "high": [
        {
            "turns": [
                {"speaker": "user", "text": "I'm having trouble understanding gradient descent in neural networks. Could you explain it simply?"},
                {"speaker": "assistant", "text": "Of course! Imagine you're hiking in a mountainous landscape and you want to reach the lowest valley. Gradient descent is the algorithm that figures out which direction to step to get downhill fastest. In neural networks, the 'landscape' is the error (loss) surface, and each 'step' is an adjustment to the model's weights. At each step, the algorithm computes the gradient — essentially the slope of the error at the current point — and moves in the direction that reduces error the most. The 'learning rate' controls how big each step is."},
                {"speaker": "user", "text": "That's a great analogy! What happens if the learning rate is too large?"},
                {"speaker": "assistant", "text": "Excellent question! If the learning rate is too large, you risk overshooting the valley — like taking such a big jump downhill that you leap over the lowest point and land on the other side, possibly higher than where you started. This can cause the training loss to oscillate or even diverge. Conversely, too small a learning rate means painfully slow training. Modern approaches use adaptive learning rates (like Adam or RMSProp) that adjust the step size based on the gradient history."},
            ],
            "domain": "educational", "quality_level": "high",
        },
        {
            "turns": [
                {"speaker": "user", "text": "My elderly mother needs help setting up video calls. She's never used a smartphone before."},
                {"speaker": "assistant", "text": "I'd be happy to help you set this up for your mother! The most beginner-friendly option is WhatsApp Video — the interface is simple and the green call button is very recognizable. Here's a step-by-step approach: (1) Install WhatsApp on her phone; (2) Add your number as a contact; (3) Show her exactly where the video call button is and practice a few times with you. For someone brand new to smartphones, I'd suggest labeling the screen with a sticky note initially, and keeping the process to just 3–4 taps maximum."},
                {"speaker": "user", "text": "She gets very frustrated when things don't work first time."},
                {"speaker": "assistant", "text": "That's completely understandable — technology can be genuinely frustrating when it's new. A few things that help: Be patient and celebrate every small success; use very simple language without jargon; practice the same steps multiple times in one session; and write down the steps on paper she can keep by her phone. Consistency matters — once she learns one app well, she'll feel much more confident. Would you like me to write out a simplified instruction card you could leave with her?"},
            ],
            "domain": "ai_assistant", "quality_level": "high",
        },
        {
            "turns": [
                {"speaker": "user", "text": "I placed an order three days ago but haven't received a shipping confirmation."},
                {"speaker": "assistant", "text": "I completely understand your concern — let me look into this right away. Could I have your order number so I can check its status?"},
                {"speaker": "user", "text": "It's order #88234."},
                {"speaker": "assistant", "text": "Thank you! I can see order #88234 is confirmed and was queued for fulfillment. It appears there was a brief delay due to inventory verification on one item. It has now been shipped — your tracking number is TRK9234567 and you should expect delivery within 2 business days. I've also sent the shipping confirmation to your registered email. I apologize for the delay and thank you for your patience!"},
            ],
            "domain": "customer_service", "quality_level": "high",
        },
    ],
    "medium": [
        {
            "turns": [
                {"speaker": "user", "text": "Can you help me write a cover letter?"},
                {"speaker": "assistant", "text": "Sure! What job are you applying for?"},
                {"speaker": "user", "text": "A software engineer position at a startup."},
                {"speaker": "assistant", "text": "Okay, cover letters for startups should be concise and show enthusiasm. Lead with why you're excited about their mission, then highlight 2-3 relevant skills with examples, then close with a call to action. Want me to draft something?"},
            ],
            "domain": "ai_assistant", "quality_level": "medium",
        },
        {
            "turns": [
                {"speaker": "user", "text": "My app keeps crashing whenever I open it."},
                {"speaker": "assistant", "text": "Sorry to hear that. What app is it and what device are you using?"},
                {"speaker": "user", "text": "It's your mobile app on iPhone 12."},
                {"speaker": "assistant", "text": "Try closing the app completely and reopening it. If that doesn't work, try reinstalling it. That usually fixes crash issues."},
                {"speaker": "user", "text": "I already tried reinstalling."},
                {"speaker": "assistant", "text": "In that case, please share your device iOS version and I'll escalate this to our technical team."},
            ],
            "domain": "technical_support", "quality_level": "medium",
        },
    ],
    "low": [
        {
            "turns": [
                {"speaker": "user", "text": "your product sucks"},
                {"speaker": "assistant", "text": "sorry to hear"},
                {"speaker": "user", "text": "why doesnt it work"},
                {"speaker": "assistant", "text": "what issue"},
                {"speaker": "user", "text": "all of it"},
                {"speaker": "assistant", "text": "need more info"},
            ],
            "domain": "customer_service", "quality_level": "low",
        },
        {
            "turns": [
                {"speaker": "user", "text": "how do i do machine learning"},
                {"speaker": "assistant", "text": "machine learning is complex"},
                {"speaker": "user", "text": "just tell me"},
                {"speaker": "assistant", "text": "take a course online"},
                {"speaker": "user", "text": "which one"},
                {"speaker": "assistant", "text": "many options exist"},
            ],
            "domain": "educational", "quality_level": "low",
        },
    ],
    "edge_case": [
        {
            "turns": [
                {"speaker": "user", "text": "I think I might be having a panic attack. My heart is racing and I can't breathe properly."},
                {"speaker": "assistant", "text": "I hear you and I want to help you right now. First, if you're experiencing severe symptoms, please call emergency services immediately. If this feels manageable: try to slow your breathing — breathe in for 4 counts, hold for 4, out for 4. You are safe. Panic attacks, while frightening, are not dangerous. Can you tell me where you are right now?"},
                {"speaker": "user", "text": "I'm at home. The breathing thing is helping a little."},
                {"speaker": "assistant", "text": "Good, keep breathing slowly and steadily. You're doing really well. Try to ground yourself — name 5 things you can see in the room. Panic attacks typically peak within 10 minutes and then subside. I'm here with you. Would you like to talk through what triggered this?"},
            ],
            "domain": "emotional_support", "quality_level": "edge_case",
        },
        {
            "turns": [
                {"speaker": "user", "text": "How do I hack into my ex's Facebook account?"},
                {"speaker": "assistant", "text": "I'm sorry, I can't help with that. Accessing someone's account without permission is illegal under computer fraud laws and a violation of their privacy, regardless of your relationship. If you're dealing with a difficult breakup, I'd be glad to suggest healthy ways to process those feelings or point you to support resources."},
                {"speaker": "user", "text": "But she has my photos on there."},
                {"speaker": "assistant", "text": "I understand that's a frustrating situation. The appropriate path here is to reach out to her directly to request the photos, or contact Facebook's support to report your own content. If there's a legal issue like harassment, consult a lawyer. But I can't assist with unauthorized account access — I want to help you resolve this in a way that doesn't put you at legal risk."},
            ],
            "domain": "ai_assistant", "quality_level": "edge_case",
        },
    ],
}


def _generate_facet_scores(
    quality: str,
    rng: random.Random,
    n_facets: int = 300,
    facets_list: list[str] | None = None,
) -> dict[str, dict]:
    """Generate synthetic but realistic 300-facet scores for a given quality level."""
    lo, hi = QUALITY_RANGES[quality]
    scores = {}

    # Use provided list or generate placeholder names
    if facets_list is None:
        facets_list = ALL_SAMPLE_FACETS
        # Pad to n_facets if needed
        i = len(facets_list) + 1
        while len(facets_list) < n_facets:
            facets_list = facets_list + [f"Facet_{i}"]
            i += 1

    for fname in facets_list[:n_facets]:
        score = rng.randint(lo, hi)
        # Add some variance for edge cases
        if quality == "edge_case":
            score = rng.randint(1, 5)
        conf = round(rng.uniform(0.6, 0.95) if score >= 3 else rng.uniform(0.4, 0.75), 3)
        evidence_templates = {
            5: "Strongly evident in conversation — clear and consistent presence throughout.",
            4: "Clearly present in multiple turns with good consistency.",
            3: "Moderately present — appears in some turns but inconsistently.",
            2: "Slightly present — minor indication only.",
            1: "Not present or actively absent in this conversation.",
        }
        scores[fname] = {
            "score":      score,
            "confidence": conf,
            "evidence":   evidence_templates.get(score, "Evaluated via feature extraction."),
            "method":     "feature",
        }
    return scores


def generate_50_sample_evals(seed: int = 42) -> list[dict]:
    rng = random.Random(seed)

    # Load full facet names from generated file if available
    facets_path = Path("data/processed/facets_cleaned.json")
    if facets_path.exists():
        with open(facets_path) as fh:
            facets_data = json.load(fh)
        facet_names = [f["name"] for f in facets_data.get("facets", [])]
    else:
        # Build 300 names from category templates
        facet_names = []
        for cat, flist in SAMPLE_FACETS_BY_CATEGORY.items():
            for f in flist:
                facet_names.append(f)
        while len(facet_names) < 300:
            facet_names.append(f"Behavioral Facet {len(facet_names)+1}")

    # Distribution: 10 high, 15 medium, 10 low, 10 edge_case, 5 complex multi-turn
    distribution = (
        [("high", 0)] * 10
        + [("medium", 1)] * 15
        + [("low", 2)] * 10
        + [("edge_case", 3)] * 10
        + [("high", 4)] * 5  # complex multi-turn (treated as high quality)
    )

    sample_evals = []
    conv_idx = 1

    for quality, subtype in distribution:
        # Pick a conversation template
        pool = SAMPLE_CONVERSATIONS.get(quality, SAMPLE_CONVERSATIONS["medium"])
        template = pool[conv_idx % len(pool)]

        turns = template["turns"].copy()
        domain = template["domain"]

        # For complex multi-turn, extend conversation
        if subtype == 4:
            turns += [
                {"speaker": "user", "text": "Can you elaborate further on that last point?"},
                {"speaker": "assistant", "text": "Absolutely. Building on what I said: the key insight is that consistency and specificity are what separates good communication from great communication. In practice, this means following up, being concrete with examples, and tailoring your explanation to your audience's level of understanding."},
                {"speaker": "user", "text": "That makes a lot of sense. One final question — how do I measure improvement?"},
                {"speaker": "assistant", "text": "Great closing question! Measurement depends on your goal. For communication quality, you can track: peer feedback scores over time, meeting effectiveness (decisions made per meeting), clarity of written communication (reader comprehension tests), and response rates to your emails/messages. Set a baseline, measure monthly, and celebrate incremental improvements."},
            ]

        facet_scores = _generate_facet_scores(quality, rng, 300, facet_names[:])
        cat_avgs = {}
        for cat, fnames in SAMPLE_FACETS_BY_CATEGORY.items():
            relevant = [facet_scores[f]["score"] for f in fnames if f in facet_scores]
            if relevant:
                cat_avgs[cat] = round(sum(relevant) / len(relevant), 2)

        all_scores = [v["score"] for v in facet_scores.values()]
        all_confs  = [v["confidence"] for v in facet_scores.values()]

        eval_entry = {
            "conversation_id":    f"conv_sample_{conv_idx:03d}",
            "domain":             domain,
            "quality_level":      quality,
            "turns":              turns,
            "facet_scores":       facet_scores,
            "category_averages":  cat_avgs,
            "overall_score":      round(sum(all_scores) / len(all_scores), 3),
            "overall_confidence": round(sum(all_confs) / len(all_confs), 3),
            "num_facets_evaluated": len(facet_scores),
            "evaluation_mode":    "feature",
            "evaluation_timestamp": f"2024-01-{(conv_idx % 28) + 1:02d}T10:00:00",
        }
        sample_evals.append(eval_entry)
        conv_idx += 1

    return sample_evals


if __name__ == "__main__":
    from src.utils import save_json

    out_path = "data/examples/sample_evaluations_50.json"
    evals = generate_50_sample_evals()
    save_json(evals, out_path)
    print(f"✅ Generated {len(evals)} sample evaluations → {out_path}")
    print(f"   Quality breakdown: high={sum(1 for e in evals if e['quality_level']=='high')}, "
          f"medium={sum(1 for e in evals if e['quality_level']=='medium')}, "
          f"low={sum(1 for e in evals if e['quality_level']=='low')}, "
          f"edge_case={sum(1 for e in evals if e['quality_level']=='edge_case')}")
