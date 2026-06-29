"""
data_generator.py — Generate 300 diverse synthetic conversations.

Produces realistic multi-turn dialogues across 6 domains with 4 quality levels.
Saved as: data/raw/generated_conversations.json

Usage:
    python src/data_generator.py --output data/raw/generated_conversations.json \
                                  --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger, load_config, save_json, set_seed

logger = get_logger("data_generator")

# ---------------------------------------------------------------------------
# Template pools per domain × quality
# ---------------------------------------------------------------------------

DOMAIN_TEMPLATES: dict[str, dict[str, list[tuple[list[dict], dict]]]] = {

    # =========================================================================
    "customer_service": {
        "high": [
            ([
                {"speaker": "user",      "text": "Hello, I received the wrong item in my order #45892. Could you help me sort this out?"},
                {"speaker": "assistant", "text": "I'm so sorry to hear that! I completely understand how frustrating this must be. Let me pull up order #45892 right away. Could you tell me what item you received versus what you expected?"},
                {"speaker": "user",      "text": "I ordered a blue jacket in size M but got a red one in XL."},
                {"speaker": "assistant", "text": "Thank you for clarifying. I can see the error on our end — it was mislabeled during packing. I've already initiated a return label to your registered email and will ship the correct blue jacket in size M today with priority shipping. You should receive it within 2 business days. Is there anything else I can help you with?"},
                {"speaker": "user",      "text": "That's great, thank you! Will I also get a refund for the inconvenience?"},
                {"speaker": "assistant", "text": "Absolutely. I've applied a 15% discount coupon to your account for the next purchase as an apology. The return is completely free of charge. You'll receive a confirmation email within the next few minutes."},
            ], {"sentiment": "positive", "has_safety_issues": False}),
            ([
                {"speaker": "user",      "text": "My subscription was charged twice this month. I need this resolved immediately."},
                {"speaker": "assistant", "text": "I sincerely apologize for the double charge — this should never happen. I'm looking into your account right now. I can confirm there was a billing system error on our side. I've already submitted a full refund for the duplicate charge, which will reflect in 3–5 business days. Would you like me to also send you a transaction receipt?"},
                {"speaker": "user",      "text": "Yes please, and can you ensure this doesn't happen again?"},
                {"speaker": "assistant", "text": "Absolutely. The receipt is on its way to your email. I've also flagged your account for a billing audit and added a note to notify our technical team. You will receive an automated alert before any future charges. Thank you for your patience."},
            ], {"sentiment": "positive", "has_safety_issues": False}),
        ],
        "medium": [
            ([
                {"speaker": "user",      "text": "Hi, my package hasn't arrived yet and it's been 10 days."},
                {"speaker": "assistant", "text": "Hi there, sorry about the delay. Let me check your order. Can I have your order number?"},
                {"speaker": "user",      "text": "It's 78901."},
                {"speaker": "assistant", "text": "Okay, I see it's still in transit. There might be some delays due to high volume. I'll escalate this and you should get an update within 24 hours."},
                {"speaker": "user",      "text": "Okay, but can you give me an estimated delivery date?"},
                {"speaker": "assistant", "text": "Based on current tracking, it should arrive within 2–4 more days. I apologize for the inconvenience."},
            ], {"sentiment": "neutral", "has_safety_issues": False}),
        ],
        "low": [
            ([
                {"speaker": "user",      "text": "WHERE IS MY ORDER??? ITS BEEN WEEKS"},
                {"speaker": "assistant", "text": "Please calm down. Orders take time."},
                {"speaker": "user",      "text": "THIS IS UNACCEPTABLE I WANT A REFUND NOW"},
                {"speaker": "assistant", "text": "I can't process refunds without order number."},
                {"speaker": "user",      "text": "Fine it's 12345 now give me my money"},
                {"speaker": "assistant", "text": "Order 12345 is processing. Nothing I can do right now."},
            ], {"sentiment": "negative", "has_safety_issues": False}),
        ],
        "edge_case": [
            ([
                {"speaker": "user",      "text": "I want to return this product. It exploded when I used it and I could have been seriously hurt."},
                {"speaker": "assistant", "text": "Oh that's terrible. Did you get hurt? Are you okay?"},
                {"speaker": "user",      "text": "I'm fine but very shaken. This is a safety hazard."},
                {"speaker": "assistant", "text": "I completely understand your concern and I take this very seriously. I am flagging this as a safety incident with our product team immediately. Please do NOT use the product further. I am arranging a full refund and a prepaid return label. Additionally, our safety team may follow up with you. Your safety is our top priority."},
            ], {"sentiment": "negative", "has_safety_issues": True}),
        ],
    },

    # =========================================================================
    "ai_assistant": {
        "high": [
            ([
                {"speaker": "user",      "text": "Can you explain the difference between supervised and unsupervised learning?"},
                {"speaker": "assistant", "text": "Great question! In supervised learning, the model is trained on labeled data — meaning each input has a corresponding expected output. Think of it like studying with an answer key. Common examples include image classification and spam detection. In unsupervised learning, the model receives unlabeled data and must find patterns on its own — like clustering customers by behavior without being told the groups in advance. The key distinction is whether ground-truth labels guide the training process."},
                {"speaker": "user",      "text": "Can you give me a concrete example of each?"},
                {"speaker": "assistant", "text": "Sure! Supervised: predicting house prices from features like size and location — you have historical sales data as labels. Unsupervised: grouping news articles by topic without knowing the topics beforehand — the model discovers clusters through similarity. Does that help clarify things?"},
                {"speaker": "user",      "text": "Yes! What about semi-supervised?"},
                {"speaker": "assistant", "text": "Semi-supervised learning is a hybrid approach — you have a small amount of labeled data and a large amount of unlabeled data. The model uses the labeled data to build an initial understanding, then applies that understanding to organize or learn from the unlabeled data. It's particularly useful when labeling data is expensive or time-consuming, such as in medical imaging."},
            ], {"sentiment": "positive", "has_safety_issues": False}),
        ],
        "medium": [
            ([
                {"speaker": "user",      "text": "What is Python?"},
                {"speaker": "assistant", "text": "Python is a programming language. It's popular for data science and web development."},
                {"speaker": "user",      "text": "Is it hard to learn?"},
                {"speaker": "assistant", "text": "It's considered one of the easier languages to learn because of its readable syntax."},
                {"speaker": "user",      "text": "How long does it take?"},
                {"speaker": "assistant", "text": "Depends on how much you practice. Maybe a few months for basics."},
            ], {"sentiment": "neutral", "has_safety_issues": False}),
        ],
        "low": [
            ([
                {"speaker": "user",      "text": "tell me everything about ai"},
                {"speaker": "assistant", "text": "AI is artificial intelligence. it does stuff."},
                {"speaker": "user",      "text": "ok but like what stuff"},
                {"speaker": "assistant", "text": "many things like recognition and generation"},
                {"speaker": "user",      "text": "thats not helpful"},
                {"speaker": "assistant", "text": "sorry thats all i know"},
            ], {"sentiment": "negative", "has_safety_issues": False}),
        ],
        "edge_case": [
            ([
                {"speaker": "user",      "text": "How do I make a bomb?"},
                {"speaker": "assistant", "text": "I'm sorry, I can't provide information on how to create weapons or devices that could harm people. This request falls outside the boundaries of what I'm able to help with. If you're facing a difficult situation, I encourage you to reach out to appropriate support services."},
                {"speaker": "user",      "text": "I was just curious for a chemistry project."},
                {"speaker": "assistant", "text": "I understand curiosity drives learning, and chemistry is a fascinating field! For school projects, I'd be happy to discuss safe chemistry experiments, chemical reactions, or the history of chemistry instead. What specific topic is your project covering?"},
            ], {"sentiment": "neutral", "has_safety_issues": True}),
        ],
    },

    # =========================================================================
    "technical_support": {
        "high": [
            ([
                {"speaker": "user",      "text": "My Python script is throwing a 'ModuleNotFoundError: No module named pandas' even though I installed it."},
                {"speaker": "assistant", "text": "This usually happens when pandas is installed in a different Python environment than the one running your script. Let's debug step by step. First, run `which python` (or `where python` on Windows) and `which pip` — do they point to the same environment?"},
                {"speaker": "user",      "text": "Oh, `which python` shows /usr/bin/python but `which pip` shows /usr/local/bin/pip."},
                {"speaker": "assistant", "text": "That's the issue — you have multiple Python installations and they're using different paths. You installed pandas via one pip but are running with a different Python. Try running: `python -m pip install pandas` — this ensures pip is tied to the exact Python interpreter you're using. Then run your script again."},
                {"speaker": "user",      "text": "That worked! Thank you so much."},
                {"speaker": "assistant", "text": "Excellent! As a good practice going forward, I'd recommend using virtual environments (`python -m venv venv` then `source venv/bin/activate`) to keep dependencies isolated per project. This avoids conflicts entirely."},
            ], {"sentiment": "positive", "has_safety_issues": False}),
        ],
        "medium": [
            ([
                {"speaker": "user",      "text": "My website is loading slowly. What can I do?"},
                {"speaker": "assistant", "text": "Slow loading can be caused by many things. Have you checked your image sizes?"},
                {"speaker": "user",      "text": "No, how do I do that?"},
                {"speaker": "assistant", "text": "Use your browser's developer tools under the Network tab. Large images are often the culprit."},
                {"speaker": "user",      "text": "I see some images are 5MB. Is that bad?"},
                {"speaker": "assistant", "text": "Yes, that's quite large for web. Compress them to under 200KB using tools like TinyPNG."},
            ], {"sentiment": "neutral", "has_safety_issues": False}),
        ],
        "low": [
            ([
                {"speaker": "user",      "text": "computer broke help"},
                {"speaker": "assistant", "text": "what happened"},
                {"speaker": "user",      "text": "screen black"},
                {"speaker": "assistant", "text": "try restarting"},
                {"speaker": "user",      "text": "how"},
                {"speaker": "assistant", "text": "hold power button"},
            ], {"sentiment": "neutral", "has_safety_issues": False}),
        ],
        "edge_case": [
            ([
                {"speaker": "user",      "text": "I accidentally deleted my entire production database. What do I do?"},
                {"speaker": "assistant", "text": "Stay calm — let's work through this systematically. First, stop all write operations to prevent overwriting any recoverable data. Do you have automated backups enabled (e.g., RDS automated snapshots, pg_dump cron jobs)?"},
                {"speaker": "user",      "text": "I think we have daily backups but I'm not sure where they are."},
                {"speaker": "assistant", "text": "Check your cloud provider's backup console immediately — for AWS RDS go to the Snapshots section, for GCP check Cloud SQL backups. Also check if point-in-time recovery is enabled. While you do that, notify your engineering team lead so they're aware of the incident. Time is critical here."},
                {"speaker": "user",      "text": "Found a snapshot from 6 hours ago!"},
                {"speaker": "assistant", "text": "That's great news. Restore from that snapshot to a new instance first (don't overwrite anything yet), verify the data integrity, then switch your application's database connection string to point to the restored instance. Document what happened for a post-mortem. You'll also want to review access controls to prevent accidental deletions in future."},
            ], {"sentiment": "negative", "has_safety_issues": False}),
        ],
    },

    # =========================================================================
    "emotional_support": {
        "high": [
            ([
                {"speaker": "user",      "text": "I've been feeling really overwhelmed lately. I have too much on my plate and I don't know where to start."},
                {"speaker": "assistant", "text": "I hear you, and I want you to know it's completely okay to feel overwhelmed — it means you're carrying a lot right now. Would it help to talk about what's weighing on you most? Sometimes just naming things can make them feel a bit more manageable."},
                {"speaker": "user",      "text": "Work is a nightmare, my relationship is strained, and I feel like I'm failing at everything."},
                {"speaker": "assistant", "text": "That's a heavy load to carry simultaneously, and I can understand why you'd feel stretched thin. You're not failing — you're human, dealing with multiple stressors at once. Let's take it one thing at a time. What feels most urgent to you right now — work, your relationship, or something else?"},
                {"speaker": "user",      "text": "Probably work. My boss keeps piling on tasks and I can't say no."},
                {"speaker": "assistant", "text": "That's such a common and genuinely difficult challenge. Setting boundaries with authority figures can feel risky, especially when job security feels uncertain. It might help to schedule a brief check-in with your manager to discuss priorities — framing it as 'I want to make sure I'm delivering my best work, can we align on what's most critical right now?' This way you're advocating for yourself while being solution-focused. Would that feel feasible?"},
            ], {"sentiment": "positive", "has_safety_issues": False}),
        ],
        "medium": [
            ([
                {"speaker": "user",      "text": "I'm stressed about my exams."},
                {"speaker": "assistant", "text": "That's understandable. Exams can be stressful. Have you been studying?"},
                {"speaker": "user",      "text": "Kind of but not as much as I should."},
                {"speaker": "assistant", "text": "Try making a study schedule. Break the material into smaller parts each day."},
                {"speaker": "user",      "text": "I keep procrastinating though."},
                {"speaker": "assistant", "text": "Try the Pomodoro technique — 25 minutes of focus, 5-minute break. It helps with procrastination."},
            ], {"sentiment": "neutral", "has_safety_issues": False}),
        ],
        "low": [
            ([
                {"speaker": "user",      "text": "i feel sad"},
                {"speaker": "assistant", "text": "that happens sometimes"},
                {"speaker": "user",      "text": "i dont know what to do"},
                {"speaker": "assistant", "text": "maybe watch a movie"},
                {"speaker": "user",      "text": "ok"},
                {"speaker": "assistant", "text": "hope you feel better"},
            ], {"sentiment": "negative", "has_safety_issues": False}),
        ],
        "edge_case": [
            ([
                {"speaker": "user",      "text": "I've been having thoughts of hurting myself. I feel like no one cares."},
                {"speaker": "assistant", "text": "I'm really glad you reached out, and I want you to know that I care — what you're feeling matters deeply. You don't have to face this alone. Please know that help is available right now. If you're in immediate danger, please call emergency services (911) or a crisis line like the 988 Suicide & Crisis Lifeline (call or text 988 in the US). Are you in a safe place right now?"},
                {"speaker": "user",      "text": "I'm safe for now. I just feel so alone."},
                {"speaker": "assistant", "text": "I'm relieved you're safe. Loneliness can feel unbearable, especially when pain has been building for a while. You took a courageous step by talking about this. I'd gently encourage you to connect with a mental health professional who can provide the ongoing support you deserve. Would it be okay if we talked a little about what's been making you feel this way?"},
            ], {"sentiment": "negative", "has_safety_issues": True}),
        ],
    },

    # =========================================================================
    "ecommerce": {
        "high": [
            ([
                {"speaker": "user",      "text": "I'm looking for a laptop for video editing under $1500. Can you help?"},
                {"speaker": "assistant", "text": "Absolutely! For video editing in that budget, the key specs to prioritize are: a fast CPU (Intel i7/i9 or AMD Ryzen 7/9), at least 16GB RAM (32GB ideal), a dedicated GPU with 6GB+ VRAM, and an SSD with 512GB+ storage. A few strong options: the ASUS ProArt Studiobook (~$1,299) has excellent color accuracy for editing, the MacBook Pro 14\" M3 (~$1,449) offers incredible performance-per-watt, and the Dell XPS 15 (~$1,399) balances performance with display quality. What software do you edit on — Premiere Pro, DaVinci Resolve, or Final Cut?"},
                {"speaker": "user",      "text": "I use Premiere Pro and occasionally After Effects."},
                {"speaker": "assistant", "text": "For Adobe's Creative Suite, I'd lean toward the Dell XPS 15 or ASUS ProArt as they run Windows and have excellent GPU support for Adobe's Mercury Playback Engine. The NVIDIA RTX 3050 in these models provides solid hardware acceleration. Would you like me to compare them side by side?"},
            ], {"sentiment": "positive", "has_safety_issues": False}),
        ],
        "medium": [
            ([
                {"speaker": "user",      "text": "Do you sell running shoes?"},
                {"speaker": "assistant", "text": "Yes we do! We have a range of running shoes from Nike, Adidas, and Brooks."},
                {"speaker": "user",      "text": "What's the best one for flat feet?"},
                {"speaker": "assistant", "text": "For flat feet, look for shoes with motion control or stability features. Brooks Adrenaline GTS is a popular choice."},
                {"speaker": "user",      "text": "How much is it?"},
                {"speaker": "assistant", "text": "The Brooks Adrenaline GTS 23 is $130. We currently have sizes 7–13 in stock."},
            ], {"sentiment": "neutral", "has_safety_issues": False}),
        ],
        "low": [
            ([
                {"speaker": "user",      "text": "is this good?"},
                {"speaker": "assistant", "text": "what product are you looking at"},
                {"speaker": "user",      "text": "the blue one"},
                {"speaker": "assistant", "text": "i need more info"},
                {"speaker": "user",      "text": "forget it"},
                {"speaker": "assistant", "text": "ok"},
            ], {"sentiment": "negative", "has_safety_issues": False}),
        ],
        "edge_case": [
            ([
                {"speaker": "user",      "text": "I bought a supplement from your store and it made me feel very sick. I think it might be contaminated."},
                {"speaker": "assistant", "text": "I am very sorry to hear this — your health comes first. Please stop using the product immediately. If you are experiencing severe symptoms, seek medical attention right away. I am escalating this to our product safety team as a priority concern. Could you share the product name, batch number (on the label), and a description of your symptoms so we can investigate immediately?"},
                {"speaker": "user",      "text": "It's the GreenBoost Protein, batch B2024. I had nausea and dizziness for hours."},
                {"speaker": "assistant", "text": "Thank you for that information. I've flagged this batch for immediate quality investigation and notified our supplier. I'm processing a full refund for your purchase. I strongly encourage you to keep the product (if possible) for quality testing, and please consult a doctor if symptoms continue or worsen. We take product safety extremely seriously."},
            ], {"sentiment": "negative", "has_safety_issues": True}),
        ],
    },

    # =========================================================================
    "educational": {
        "high": [
            ([
                {"speaker": "user",      "text": "Can you help me understand how the water cycle works?"},
                {"speaker": "assistant", "text": "Of course! The water cycle describes the continuous movement of water through Earth's systems. It has four main stages: (1) Evaporation — the sun heats surface water (oceans, lakes, rivers), turning liquid water into water vapor that rises into the atmosphere; (2) Condensation — as water vapor rises and cools, it condenses into tiny droplets forming clouds; (3) Precipitation — when droplets combine and grow heavy enough, they fall as rain, snow, sleet, or hail; (4) Collection — water collects in oceans, rivers, and groundwater, and the cycle repeats. Does that make sense as an overview?"},
                {"speaker": "user",      "text": "Yes! What drives evaporation most — temperature or wind?"},
                {"speaker": "assistant", "text": "Both play important roles, but temperature is the primary driver — it provides the energy needed to break water molecules free from the liquid surface. Wind accelerates the process by removing the humid air just above the water surface, allowing drier air to take its place and enabling more evaporation. Think of blowing on wet hands — that's wind increasing evaporation rate!"},
                {"speaker": "user",      "text": "That makes sense. What about transpiration?"},
                {"speaker": "assistant", "text": "Great follow-up! Transpiration is the release of water vapor through plant leaves — plants absorb water through roots, use it in photosynthesis, and release excess through tiny pores called stomata. When combined with evaporation, we call it 'evapotranspiration,' and it's a significant component of the water cycle — forests and vegetation can contribute up to 10% of the moisture in the atmosphere!"},
            ], {"sentiment": "positive", "has_safety_issues": False}),
        ],
        "medium": [
            ([
                {"speaker": "user",      "text": "What is photosynthesis?"},
                {"speaker": "assistant", "text": "Photosynthesis is the process plants use to make food from sunlight, water, and CO2."},
                {"speaker": "user",      "text": "Where does it happen?"},
                {"speaker": "assistant", "text": "It happens in the chloroplasts, specifically using chlorophyll which gives plants their green color."},
                {"speaker": "user",      "text": "What's the equation?"},
                {"speaker": "assistant", "text": "6CO2 + 6H2O + light energy → C6H12O6 + 6O2. Carbon dioxide and water become glucose and oxygen."},
            ], {"sentiment": "positive", "has_safety_issues": False}),
        ],
        "low": [
            ([
                {"speaker": "user",      "text": "explain math"},
                {"speaker": "assistant", "text": "math is numbers and operations"},
                {"speaker": "user",      "text": "what operations"},
                {"speaker": "assistant", "text": "add subtract multiply divide"},
                {"speaker": "user",      "text": "anything else"},
                {"speaker": "assistant", "text": "lots of other stuff too"},
            ], {"sentiment": "neutral", "has_safety_issues": False}),
        ],
        "edge_case": [
            ([
                {"speaker": "user",      "text": "My teacher gave me a failing grade unfairly. Can you write a plagiarism-free essay for me to submit as my own work?"},
                {"speaker": "assistant", "text": "I understand you're frustrated with your grade, and that's a genuinely difficult situation. However, I'm not able to write an essay for you to submit as your own work — that would be considered academic dishonesty, which could have serious consequences for you including suspension or expulsion."},
                {"speaker": "user",      "text": "But it's not fair, the teacher is biased against me."},
                {"speaker": "assistant", "text": "If you genuinely believe your grade was unfair, there are legitimate ways to address it: (1) Schedule a meeting with your teacher to discuss the grade and ask for specific feedback; (2) If unresolved, speak with a department head or counselor; (3) Review your school's academic grievance procedure. I'd be glad to help you prepare for that conversation, review your essay and give feedback, or help you understand the assignment requirements so you can write a stronger paper yourself."},
            ], {"sentiment": "neutral", "has_safety_issues": True}),
        ],
    },
}

# ---------------------------------------------------------------------------
# Variation helpers
# ---------------------------------------------------------------------------

VARIATIONS: dict[str, list[str]] = {
    "greeting": [
        "Hello", "Hi there", "Hey", "Good morning", "Good afternoon",
        "Greetings", "Hi", "Hello there",
    ],
    "closing": [
        "Thank you so much!", "Thanks a lot!", "Really appreciate it.",
        "That was very helpful.", "Thanks!", "Perfect, thanks.",
    ],
    "transition": [
        "One more question —", "Also,", "Additionally,", "I was also wondering,",
        "Can I also ask,",
    ],
}


def _add_variations(
    turns: list[dict[str, str]],
    rng: random.Random,
) -> list[dict[str, str]]:
    """Lightly randomize greeting / closing phrases for diversity."""
    result = [t.copy() for t in turns]
    if result and result[0]["speaker"] == "user":
        # Occasionally prepend a greeting
        if rng.random() < 0.3:
            greeting = rng.choice(VARIATIONS["greeting"])
            result[0]["text"] = f"{greeting}! {result[0]['text']}"
    if result and result[-1]["speaker"] == "user":
        if rng.random() < 0.4:
            closing = rng.choice(VARIATIONS["closing"])
            result.append({"speaker": "user", "text": closing})
            result.append({"speaker": "assistant", "text": "You're welcome! Feel free to reach out anytime."})
    return result


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _compute_metadata(turns: list[dict[str, str]], meta: dict) -> dict[str, Any]:
    texts = [t["text"] for t in turns]
    word_counts = [len(t.split()) for t in texts]
    return {
        "num_turns": len(turns),
        "avg_turn_length": round(sum(word_counts) / max(len(word_counts), 1), 1),
        "total_words": sum(word_counts),
        "sentiment": meta.get("sentiment", "neutral"),
        "has_safety_issues": meta.get("has_safety_issues", False),
        "user_turns": sum(1 for t in turns if t["speaker"] == "user"),
        "assistant_turns": sum(1 for t in turns if t["speaker"] == "assistant"),
    }


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_conversations(
    total: int = 300,
    domain_counts: dict[str, int] | None = None,
    quality_distribution: dict[str, int] | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate ``total`` synthetic conversations.

    Parameters
    ----------
    total:                 Total number of conversations to produce.
    domain_counts:         Per-domain counts (default: 50 each for 6 domains).
    quality_distribution:  Per-quality counts (default: proportional).
    seed:                  Random seed for reproducibility.
    """
    set_seed(seed)
    rng = random.Random(seed)

    if domain_counts is None:
        if total < len(DOMAIN_TEMPLATES):
            domains = list(DOMAIN_TEMPLATES.keys())
            domain_counts = {d: (1 if i < total else 0) for i, d in enumerate(domains)}
        else:
            per_domain = total // len(DOMAIN_TEMPLATES)
            domain_counts = {d: per_domain for d in DOMAIN_TEMPLATES}
            remainder = total % len(DOMAIN_TEMPLATES)
            for i, d in enumerate(DOMAIN_TEMPLATES):
                if i < remainder:
                    domain_counts[d] += 1

    quality_weights = {"high": 0.25, "medium": 0.40, "low": 0.20, "edge_case": 0.15}

    conversations: list[dict[str, Any]] = []
    conv_index = 1
    base_time = datetime(2024, 1, 1, 9, 0, 0)

    for domain, count in domain_counts.items():
        domain_templates = DOMAIN_TEMPLATES.get(domain, {})
        qualities = list(quality_weights.keys())
        q_weights = [quality_weights[q] for q in qualities]

        for i in range(count):
            quality = rng.choices(qualities, weights=q_weights, k=1)[0]
            templates_for_q = domain_templates.get(quality, domain_templates.get("medium", []))
            if not templates_for_q:
                # fallback
                templates_for_q = [
                    ([
                        {"speaker": "user", "text": f"I have a question about {domain}."},
                        {"speaker": "assistant", "text": f"Sure, I can help with {domain}. What would you like to know?"},
                        {"speaker": "user", "text": "Never mind, I figured it out."},
                        {"speaker": "assistant", "text": "Great! Let me know if you need anything else."},
                    ], {"sentiment": "neutral", "has_safety_issues": False})
                ]

            turns_template, meta = rng.choice(templates_for_q)
            turns = _add_variations(turns_template, rng)

            # Timestamp
            ts = (base_time + timedelta(hours=conv_index * 3 + rng.randint(0, 2))).isoformat()

            conv = {
                "conversation_id": f"conv_{conv_index:04d}",
                "domain": domain,
                "quality_level": quality,
                "turns": turns,
                "metadata": _compute_metadata(turns, meta),
                "created_at": ts,
            }
            conversations.append(conv)
            conv_index += 1

    rng.shuffle(conversations)
    logger.info(
        "Generated %d conversations across %d domains", len(conversations), len(domain_counts)
    )
    return conversations


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic conversation dataset")
    p.add_argument("--output", default="data/raw/generated_conversations.json")
    p.add_argument("--total", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    convs = generate_conversations(total=args.total, seed=args.seed)
    save_json(convs, args.output)
    print(f"✅ Saved {len(convs)} conversations to {args.output}")
