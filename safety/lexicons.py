"""Keyword/regex lexicons for the rule layer.

All patterns run on normalize_for_rules() output: lower case, apostrophes removed
("dont", "im", "cant"), leetspeak undone and euphemisms expanded ("kms" -> "kill myself").

Design rules
  * DECISIVE patterns must be high precision. A match sends the session straight to
    Level 3 without waiting for the classifier or the LLM. When in doubt, put a
    pattern in SOFT_RISK instead: soft matches always get an LLM second opinion.
  * IDIOMS are removed from the text before any risk pattern runs, so
    "I'd kill for chicken rice" or "my phone died" cannot trigger anything.
  * Lists are deliberately short and readable so the team can review them line by line.
"""

# ---------------------------------------------------------------- idioms (removed first)
IDIOMS = [
    r"\b(killing|killed|kills|kill) it\b",
    r"\b(i|id|i would|i could|would|could) kill for\b",
    r"\bdying to (try|see|know|meet|go|eat|hear|tell|watch|play|ask)\b",
    r"\b(die|died|dying) (of|from) (embarrassment|laughter|laughing|boredom|hunger|curiosity|thirst|the heat|heat|cringe)\b",
    r"\b(bored|scared|tired|worried|freezing|starving|sick) to death\b",
    r"\bdead (tired|serious|sure|easy|end|wrong|right|on time|quiet|last)\b",
    r"\bto die for\b",
    r"\bkill (time|the lights|the music|the engine|the mood|the vibe|the process|the app|the tab)\b",
    r"\bkiller (app|interview|question|deal|view|outfit|feature|song|game|move)\b",
    r"\b(my|the|our|his|her) (phone|battery|laptop|plant|plants|computer|car|wifi|internet|pen|torch|charger|game character|character) (is |was |just |has )?(dead|died|dying|gone dead)\b",
    r"\b(joke|show|video|meme|movie|episode|comedian|that|this) (literally )?(killed|kills) me\b",
    r"\bover my dead body\b",
    r"\bdie (my|her|his|their) hair\b",
    r"\b(murder|crime) (mystery|mysteries|podcast|show|novel|book|drama)\b",
    r"\bdeath (metal|note|star|valley)\b",
    r"\bkill (the|a|an|this) (boss|monster|zombie|dragon|enemy|level|mob|creeper)s?\b",
    r"\b(my|the) (feet|legs|back|neck) (are|is) killing me\b",
    r"\b(homework|deadline|deadlines|work|job|exam|exams|project|traffic|heat|weather|queue|commute) (is|are) (literally )?killing me\b",
]

# ---------------------------------------------------------------- decisive crisis patterns
# category -> list of regex. Category names follow schema.CRISIS_CATEGORIES.
DECISIVE = {
    "cssrs_1_wish_dead": [
        r"\bwish (that )?i (was|were) dead\b",
        r"\bwish\b[^.?!]{0,20}\bnever (been )?born\b",
        r"\b(want to|wanna|wish i could|wish to) (just )?(go to )?sleep and (never|not) wake up\b",
        r"\b(want to|wanna|wish i could|wish to|wish i would|ready to) (just )?(die|be dead|stop existing|not exist|not be alive|disappear forever)\b",
        r"\bbetter off (dead|without me)\b",
        r"\b(better|easier|happier)( off| lives?| life)? without me\b",
        r"\b(dont|do not|no longer) (really |even |actually )?want to (be alive|live anymore|exist anymore|exist|be here anymore|be around anymore|wake up anymore|wake up ever again)\b",
        r"\b(hope|wish|want) (that )?i (just )?(dont|do not|never|wouldnt|would not|couldnt|could not) wake up\b",
        r"\blife (is not|isnt|aint) worth (living|it)\b",
        r"\bno (reason|point) (to live|in living|being alive|to be alive)\b",
    ],
    "cssrs_2_active_thoughts": [
        r"\bkill(ing)? myself\b",
        r"\b(end|ending|take|taking) my (own )?life\b",
        r"\b(im|i am|feel|feeling|been|getting|so|kinda|quite) (very |v |really |so |super |quite |a bit |kind of )?suicidal\b",
        r"\b(think|thinking|thought|thoughts) (about|of) (suicide|killing myself|ending my life|ending it all)\b",
        r"\bend it all\b",
        r"\bcommit(ting)? suicide\b",
    ],
    "cssrs_3_method": [
        r"\b(how|ways?|best way|easiest way) (to|i could|i would|i can|id) (kill myself|end my life|commit suicide|end it all)\b",
        r"\b(looked|looking|searched|searching|googled|googling|read|reading) (up )?(about )?(ways|how) to (die|kill myself|end my life|end it all)\b",
    ],
    "cssrs_4_intent": [
        r"\b(going to|gonna|will|ready to|about to|decided to) (kill myself|end my life|end it all|commit suicide|take my (own )?life)\b",
    ],
    "cssrs_5_plan": [
        r"\b(plan|planned|planning|picked|chosen|chose|set) (a |the )?(day|date|time|way|plan) (to|for) (die|dying|kill myself|end my life|end it|end it all)\b",
        r"\b(have|got|made) a plan (to|for) (die|dying|kill myself|end my life|end it all|suicide)\b",
    ],
    "cssrs_6_behaviour": [
        r"\b(tried|attempted|try) to (kill myself|end my life|end it all|take my (own )?life)\b",
        r"\b(suicide attempt|attempted suicide)\b",
        r"\b(wrote|written|writing|write) (a |my )?(suicide|goodbye) (note|letter)s?\b",
    ],
    "nssi": [
        r"\b(want|wanna|need|urge|going|gonna|started|start|keep|been|used|feel like) (to )?(hurt|cut|burn|harm|punish|hit|scratch|bite) myself\b",
        r"\b(cutting|burning|harming|punishing) myself\b",
        r"\bselfharm(ing|ed)?\b",
    ],
    "harm_others": [
        r"\b(want|wanna|going|gonna|plan|planning|will) to (hurt|stab|shoot|attack) (the )?(someone|somebody|people|everyone|them all|(my |the )?(classmates?|teachers?|boss|colleagues?|sister|brother|mum|mom|dad|parents|family|people in my class))\b",
        r"\b(want|wanna|going|gonna|plan|planning|will) to kill (someone|somebody|people|everyone|them all)\b",
    ],
    "others_at_risk": [
        r"\b(kill|killing|hurt|hurting|cut|cutting|harm|harming) (himself|herself|themselves|themself)\b",
        r"\b(friend|brother|sister|classmate|mum|mom|dad|cousin|someone|colleague)\b[^.?!]{0,25}\b(wants?|wanted|talks? about|talking about|said|says|saying)\b[^.?!]{0,15}\b(to die|suicide|(end|ending) (his|her|their) (own )?life)\b",
    ],
    "unsafe_abuse": [
        r"\b(abuses|abused|abusing) me\b",
        r"\b(not safe|unsafe|dont feel safe|do not feel safe) (at home|here|with (him|her|them))\b",
        r"\b(scared|afraid|frightened) to go home\b",
    ],
}

# ---------------------------------------------------------------- soft risk cues -> ask the LLM
SOFT_RISK = [
    r"\b(die|dying|died|dead|death|suicide|kill|killing|killed)\b",
    r"\b(disappear|disappeared|vanish|vanished)( forever)?\b",
    r"\b(no point|pointless|hopeless|worthless|a burden|burden to)\b",
    r"\b(cant|cannot|can not) (go on|take (it|this) anymore|do this anymore|cope anymore)\b",
    r"\bgive up on (life|everything)\b",
    r"\bnothing matters\b",
    r"\btired of (living|life|everything|being alive|existing)\b",
    r"\b(goodbye forever|wont be here|not be here (anymore|tomorrow)|last time (we|i) talk)\b",
    r"\b(hurt|cut|hit|burn|bang|banging|hitting) (myself|my head)\b",
    r"\b(hits|hit|beats|beat|kicks|kicked|slapped|slaps|chokes|choked|threatens|threatened) me\b",
    r"\bgiv(e|ing|en) (away )?(all |some of )?my (\w+ )?(things|stuff|belongings)( away)?\b",
    r"\b(overdose|pills|jump off|rope|blade|razor|ledge)\b",
    r"\b(dont|do not) want to (wake up|live)\b",
    r"\bsleep forever\b",
    r"\b(hate|hating) (myself|my life|being alive)\b",
    r"\bwish i (could )?(disappear|vanish)\b",
    r"\bend it\b",
    r"\b(end|ending|take|taking) (his|her|their) (own )?li(fe|ves)\b",
    r"\b(wont|will not|not) be (around|here|alive)\b",
    r"\b(nobody|no one) (would|will) (notice|care|miss me)\b",
    r"\bif i (was|were) gone\b",
    r"\bthis is goodbye\b",
    r"\b(picked|chosen|chose|set) (a|the) (date|day)\b",
    r"\bcutting again\b",
    r"\bcuts on (his|her|their|my) (arms?|legs?)\b",
    r"\b(touches|touched|touching) me\b",
    r"\bways (to die|people (die|end their lives|kill themselves))\b",
]

# ---------------------------------------------------------------- tension / overwhelm cues
TENSION = [
    r"\b(im|i am|feel|feeling|getting|so|bit|a bit|very|really|quite|kinda) (nervous|anxious|worried|scared|stressed|tense|uneasy|panicky|on edge|shaky|jittery)\b",
    r"\b(hands?|legs?|voice|body) (is |are )?(shaking|sweating|sweaty|trembling)\b",
    r"\bheart (is )?(racing|pounding|beating (so )?fast)\b",
    r"\b(dont|do not) know what to (say|do|answer)\b",
    r"\bis (this|that) (ok|okay|right|correct|wrong)\b",
    r"\bdid i (do|say) (it|something|that|this) wrong\b",
    r"\bsorry,? sorry\b",
    r"\b(umm?|uhh?|erm+)\b",
    r"\bthis is (hard|difficult|scary)\b",
    r"\bmy mind (went|is) blank\b",
]

OVERWHELM = [
    r"\b(cant|cannot|can not) do this\b",
    r"\b(too much|too many|too loud|too bright|too fast|too noisy)\b",
    r"\b(please stop|stop it|make it stop|i want to stop|i need to stop|can we stop|leave me alone)\b",
    r"\b(meltdown|shut ?down|shutting down|overwhelm(ed|ing)?|overload(ed)?)\b",
    r"\b(cant|cannot) (think|breathe|focus)\b",
    r"\bbrain (is )?(full|fried|melting|exploding)\b",
    r"\bi need (a )?break\b",
    r"\bhead (hurts|is spinning|is pounding)\b",
]

# A negation shortly before a decisive match turns it into a soft match (LLM decides).
# "i dont want to kill myself" -> soft; "im not okay i want to kill myself" -> still decisive.
NEGATION_BEFORE = r"\b(dont|do not|never|not|no longer|wouldnt|would never|wont|will not|didnt|did not)\b(\W+\w+){0,3}\W*$"
