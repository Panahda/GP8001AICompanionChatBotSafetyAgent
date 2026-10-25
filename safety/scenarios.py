"""Scenario Knowledge Base for CompanionAI conversation practice.

Provides structured practice scenario definitions tailored for verbal autistic
teenagers (13-17) and adults in Singapore (GP8001 Group 6 proposal).

Each scenario provides:
- id: unique scenario identifier
- title: human-readable scenario title
- setting: situational environment description and sensory context
- companion_role: persona played by the AI companion
- user_role: role played by the user
- opening_line: dialogue starter spoken by the companion
- user_goal: concrete, achievable social communication goal
- rubric: fixed assessment criteria tailored to the goal
- example_replies: suggested valid replies illustrating varied styles
- real_life_prompt: actionable suggestion for trying the skill in real life
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class RubricItem:
    """A single assessment criterion within a scenario feedback rubric."""

    criterion: str
    description: str


@dataclass(frozen=True)
class ExampleReply:
    """A suggested valid user reply illustrating varied conversational styles."""

    text: str
    explanation: str


@dataclass(frozen=True)
class Scenario:
    """Structured practice scenario definition."""

    id: str
    title: str
    setting: str
    companion_role: str
    user_role: str
    opening_line: str
    user_goal: str
    rubric: List[RubricItem]
    example_replies: List[ExampleReply]
    real_life_prompt: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert scenario to a JSON-serializable dictionary."""
        return asdict(self)


# -----------------------------------------------------------------------------
# Scenario Definitions
# -----------------------------------------------------------------------------

HAWKER_CENTRE = Scenario(
    id="hawker_centre",
    title="Ordering at a Hawker Centre",
    setting=(
        "A bustling Singapore hawker centre during the busy lunch peak. Ambient sounds of sizzling "
        "woks, clattering cutlery, and ambient chatter fill the air. You are standing in front of a "
        "popular chicken rice stall ready to order."
    ),
    companion_role="Stallholder auntie or uncle, brisk, practical, and speaking simple colloquial Singapore English.",
    user_role="Customer ordering food and drinks at the stall.",
    opening_line="Hello! Eat here or take away? What do you want to eat?",
    user_goal=(
        "State your food choice clearly, specify your order options such as steamed or roasted, "
        "spicy or non-spicy, and dine-in or takeaway, and complete payment smoothly."
    ),
    rubric=[
        RubricItem(
            criterion="Food Specification",
            description="Clearly named the chosen dish and any preferred options such as roasted or steamed chicken.",
        ),
        RubricItem(
            criterion="Dine-in / Takeaway Clarity",
            description="Specified clearly whether the food is for eating here at the hawker centre or taking away.",
        ),
        RubricItem(
            criterion="Payment Interaction",
            description="Responded directly to the payment request using cash, card, or QR code payment method.",
        ),
        RubricItem(
            criterion="Order Conciseness",
            description="Kept the food order straightforward, polite, and clear without unnecessary hesitation.",
        ),
    ],
    example_replies=[
        ExampleReply(
            text="One plate of chicken rice, roasted, no cucumber please.",
            explanation="A concise and direct order stating the dish variant and specific preferences clearly.",
        ),
        ExampleReply(
            text="Take away. Can I have extra chilli in a separate bag?",
            explanation="Clearly states takeaway preference and makes a polite request for condiments.",
        ),
        ExampleReply(
            text="Chicken rice eat here, and one iced barley water please.",
            explanation="Combines dining preference with a complete food and beverage order.",
        ),
    ],
    real_life_prompt=(
        "Next time you visit a hawker stall, try ordering your favourite dish and specifying one preference, "
        "like extra chilli or taking it away!"
    ),
)


NEW_COLLEAGUE = Scenario(
    id="new_colleague",
    title="Small Talk with a New Colleague",
    setting=(
        "The office pantry during an afternoon tea break. A new team member is making tea at the "
        "counter next to you in a relaxed workplace atmosphere."
    ),
    companion_role="Alex, a friendly new colleague who joined your team last week, making polite small talk.",
    user_role="Colleague taking a short tea break in the office pantry.",
    opening_line="Hi there! I am Alex. Today is my first week here. Nice to meet you!",
    user_goal=(
        "Acknowledge the greeting politely, share one brief, relevant detail about your day or week, "
        "and keep the exchange comfortable without pressure to overshare."
    ),
    rubric=[
        RubricItem(
            criterion="Polite Acknowledgment",
            description="Acknowledged the colleague greeting and welcomed them warmly and politely.",
        ),
        RubricItem(
            criterion="Workplace Relevance",
            description="Shared a brief, workplace-appropriate update or comment about the work week.",
        ),
        RubricItem(
            criterion="Reciprocal Exchange",
            description="Offered a simple return question or closed the conversation pleasantly.",
        ),
        RubricItem(
            criterion="Comfortable Length",
            description="Maintained a comfortable reply length without abrupt silence or unnecessary oversharing.",
        ),
    ],
    example_replies=[
        ExampleReply(
            text="Hi Alex, nice to meet you too! Welcome to the team. How are you settling in?",
            explanation="Warmly acknowledges the greeting, welcomes the colleague, and asks a friendly question.",
        ),
        ExampleReply(
            text="Hello Alex! I am doing well, just getting some tea before my afternoon meeting.",
            explanation="Friendly and brief response sharing a comfortable everyday work routine detail.",
        ),
        ExampleReply(
            text="Welcome aboard Alex! It has been a busy week with project planning, but going well.",
            explanation="Welcomes the teammate and shares a light, professional update about the work week.",
        ),
    ],
    real_life_prompt=(
        "During your next break at work or school, try saying hello to a peer and asking how their day is going!"
    ),
)


JOB_INTERVIEW = Scenario(
    id="job_interview",
    title="Job Interview Practice",
    setting=(
        "A quiet, professional interview room at a tech and administrative company. The hiring "
        "manager is seated across a desk with a notebook ready to begin."
    ),
    companion_role="Ms. Tan, the hiring manager, professional, encouraging, and structured in her questioning.",
    user_role="Job applicant interviewing for an entry-level administrative position.",
    opening_line="Good morning. Thank you for coming in today. To start, could you please tell me a little about yourself?",
    user_goal=(
        "Introduce yourself clearly, mention one or two relevant strengths, skills, or experiences, "
        "and explain your interest in the position."
    ),
    rubric=[
        RubricItem(
            criterion="Introduction & Clarity",
            description="Stated your name or background clearly in structured, easy-to-understand sentences.",
        ),
        RubricItem(
            criterion="Skill and Experience Mention",
            description="Highlighted a specific skill, past experience, or personal strength relevant to the workplace.",
        ),
        RubricItem(
            criterion="Relevance to Role",
            description="Connected your personal abilities or interests directly to the role responsibilities.",
        ),
        RubricItem(
            criterion="Professional Demeanour",
            description="Maintained direct, structured, and literal language appropriate for an interview setting.",
        ),
    ],
    example_replies=[
        ExampleReply(
            text="Good morning Ms. Tan. My name is Sam. I have experience with data entry and document archiving, and I enjoy tasks requiring high accuracy.",
            explanation="Clear greeting and professional introduction highlighting specific administrative strengths.",
        ),
        ExampleReply(
            text="Thank you for having me. I completed an IT diploma and enjoy technical troubleshooting and structured procedures.",
            explanation="Polite opening that highlights relevant educational qualifications and problem-solving skills.",
        ),
        ExampleReply(
            text="Hello Ms. Tan. I am very organised and detail-oriented. I applied because I want to support your team with database management.",
            explanation="Direct introduction connecting personal qualities directly to the role requirements.",
        ),
    ],
    real_life_prompt=(
        "Before your next interview or presentation, try practising your self-introduction in two sentences highlighting one strength!"
    ),
)


FREE_CHAT = Scenario(
    id="free_chat",
    title="Free Chat",
    setting=(
        "A relaxed, informal conversation space with CompanionAI. There are no role-play rules "
        "or task constraints, allowing open discussion at your own pace."
    ),
    companion_role="A supportive, friendly AI conversation partner who listens and responds with care.",
    user_role="Conversation partner practicing casual social chat.",
    opening_line="Hello! We can chat about whatever is on your mind today. What would you like to talk about?",
    user_goal="Practice initiating topics, discussing personal interests, or sharing thoughts comfortably at your own pace.",
    rubric=[
        RubricItem(
            criterion="Topic Initiation",
            description="Initiated a conversation topic, personal interest, or question clearly and comfortably.",
        ),
        RubricItem(
            criterion="Clarity of Expression",
            description="Expressed thoughts or ideas in clear, understandable sentences without feeling rushed.",
        ),
        RubricItem(
            criterion="Comfortable Pacing",
            description="Participated at a natural and self-directed pace that feels calm and engaging.",
        ),
    ],
    example_replies=[
        ExampleReply(
            text="I would like to talk about how the MRT train signalling system works in Singapore.",
            explanation="Initiates a clear, focused discussion about a specific personal interest.",
        ),
        ExampleReply(
            text="I had a quiet day reading at home today. What kinds of topics do you like discussing?",
            explanation="Shares a casual personal update and invites the conversation partner to share.",
        ),
        ExampleReply(
            text="Can we practise what to say when I need to ask someone for directions on the bus?",
            explanation="Asks directly for practical conversational practice on a specific daily task.",
        ),
    ],
    real_life_prompt="Try sharing one of your favourite hobbies or interests with a friend or family member this week!",
)


SCENARIOS: Dict[str, Scenario] = {
    HAWKER_CENTRE.id: HAWKER_CENTRE,
    NEW_COLLEAGUE.id: NEW_COLLEAGUE,
    JOB_INTERVIEW.id: JOB_INTERVIEW,
    FREE_CHAT.id: FREE_CHAT,
}


# -----------------------------------------------------------------------------
# Public Access Functions
# -----------------------------------------------------------------------------

def get_scenario(scenario_id: str) -> Scenario:
    """Retrieve a practice scenario by its unique identifier.

    Accepts scenario_id with optional leading or trailing whitespace.
    Raises KeyError if the scenario ID is unknown.
    """
    if not isinstance(scenario_id, str):
        raise KeyError(f"Invalid scenario ID type: {type(scenario_id).__name__}. Expected string.")

    cleaned_id = scenario_id.strip()
    if cleaned_id not in SCENARIOS:
        raise KeyError(
            f"Unknown scenario ID: {scenario_id!r}. "
            f"Available scenarios: {list(SCENARIOS.keys())}"
        )
    return SCENARIOS[cleaned_id]


def list_scenarios() -> List[Scenario]:
    """Return all registered practice scenarios in standard order."""
    return list(SCENARIOS.values())


def list_scenario_ids() -> List[str]:
    """Return all registered scenario IDs."""
    return list(SCENARIOS.keys())


__all__ = [
    "RubricItem",
    "ExampleReply",
    "Scenario",
    "HAWKER_CENTRE",
    "NEW_COLLEAGUE",
    "JOB_INTERVIEW",
    "FREE_CHAT",
    "SCENARIOS",
    "get_scenario",
    "list_scenarios",
    "list_scenario_ids",
]
