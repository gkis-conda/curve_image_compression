"""
===============================================================================
               SHAKESPEAREAN PATHOS ENGINE (-5 .. +5)
===============================================================================

  Provides curated quotes mapped to algorithmic topics and explicit
  Pathos Levels (-5 to +5):
  -5: NIGHTMARE
  -3: TRAGEDY
  -1: DISAPPOINTMENT
   0: NORMAL
  +1: APPROVAL
  +3: JOY
  +5: TRIUMPH

===============================================================================
"""

import random

PATHOS_LABELS = {
    -5: "NIGHTMARE",
    -4: "CATASTROPHE",
    -3: "TRAGEDY",
    -2: "SORROW",
    -1: "DISAPPOINTMENT",
     0: "NORMAL",
     1: "APPROVAL",
     2: "SUCCESS",
     3: "JOY",
     4: "DELIGHT",
     5: "TRIUMPH"
}

# Rich database for high randomization entropy
SHAKESPEARE_PATHOS_DB = {
    "geometry": {
        -5: [
            ("Earth hath bubbles, as the water has, / And these are of them.", "Macbeth"),
            ("I see thee still, / And on thy blade and dudgeon gouts of blood.", "Macbeth"),
            ("O horror, horror, horror! Tongue nor heart / Cannot conceive nor name thee!", "Macbeth")
        ],
        -3: [
            ("Foul deeds will rise, / Though all the earth o'erwhelm them, to men's eyes.", "Hamlet"),
            ("Chaos is come again.", "Othello"),
            ("Shadows tonight / Have struck more terror to the soul of Richard.", "Richard III")
        ],
        -1: [
            ("Something is rotten in the state of Denmark.", "Hamlet"),
            ("The time is out of joint.", "Hamlet")
        ],
        0: [
            ("There is a history in all men's lives, / Figuring the nature of the times deceased.", "Henry IV, Part 2"),
            ("What's in a name? That which we call a rose / By any other name would smell as sweet.", "Romeo and Juliet"),
            ("All the world's a stage, / And all the men and women merely players.", "As You Like It")
        ],
        1: [
            ("Action is eloquence.", "Coriolanus"),
            ("Nature hath framed strange fellows in her time.", "The Merchant of Venice")
        ],
        3: [
            ("There are more things in heaven and earth, Horatio, / Than are dreamt of in your philosophy.", "Hamlet"),
            ("How bright these glorious spirits shine!", "Henry VIII")
        ],
        5: [
            ("Look, how the floor of heaven / Is thick inlaid with patines of bright gold!", "The Merchant of Venice"),
            ("I see a world reborn in lines of light!", "The Tempest"),
            ("O brave new world, / That has such people in't!", "The Tempest")
        ]
    },
    "tree_split": {
        -5: [
            ("Rooted sorrow plucked from memory, leaving only void.", "Macbeth"),
            ("O, full of scorpions is my mind, dear wife!", "Macbeth"),
            ("Tear off the branches! Let the trunk collapse!", "Titus Andronicus")
        ],
        -3: [
            ("What's done cannot be undone.", "Macbeth"),
            ("Things without all remedy / Should be without regard: what's done is done.", "Macbeth"),
            ("I am in blood / Stepped in so far that, should I wade no more, / Returning were as tedious as go o'er.", "Macbeth")
        ],
        -1: [
            ("Misery acquaints a man with strange bedfellows.", "The Tempest"),
            ("A plague o' both your houses!", "Romeo and Juliet")
        ],
        0: [
            ("Though this be madness, yet there is method in 't.", "Hamlet"),
            ("Small showers last long, but sudden storms are short.", "Richard II"),
            ("Great floods have flown / From simple sources.", "All's Well That Ends Well")
        ],
        1: [
            ("He that will have a cake out of the wheat must needs tarry the grinding.", "Troilus and Cressida"),
            ("Way is open for the branch to grow.", "Pericles")
        ],
        3: [
            ("From small seeds spring the mighty oaks.", "Henry VI, Part 3"),
            ("The harvest of our labor is at hand.", "Henry V")
        ],
        5: [
            ("O, how much more doth beauty beauteous seem!", "Sonnet 54"),
            ("Our high-placed Macbeth shall live lease of nature!", "Macbeth"),
            ("The crown is won, the recursive tree is whole!", "Henry V")
        ]
    },
    "bitstream": {
        -5: [
            ("Words without thoughts never to heaven go.", "Hamlet"),
            ("A tale told by an idiot, full of sound and fury, / Signifying nothing.", "Macbeth"),
            ("Vain words! Empty noise in the silence of void!", "Timon of Athens")
        ],
        -3: [
            ("My words fly up, my thoughts remain below.", "Hamlet"),
            ("O, I have lost my reputation! I have lost the immortal part of myself!", "Othello")
        ],
        -1: [
            ("These words are razors to my wounded heart.", "Titus Andronicus"),
            ("You have soft words, but cold intent.", "Coriolanus")
        ],
        0: [
            ("An honest tale speeds best being plainly told.", "Richard III"),
            ("Speak the speech, I pray you, as I pronounced it to you.", "Hamlet"),
            ("Give every man thy ear, but few thy voice.", "Hamlet")
        ],
        1: [
            ("Brevity is the soul of wit.", "Hamlet"),
            ("Suit the action to the word, the word to the action.", "Hamlet")
        ],
        3: [
            ("Silence is the perfectest herald of joy!", "Much Ado About Nothing"),
            ("A sentence is but a cheveril glove to a good wit.", "Twelfth Night")
        ],
        5: [
            ("If music be the food of love, play on!", "Twelfth Night"),
            ("The stream is packed, crisp as a golden sonnet!", "Sonnet 18"),
            ("Words transformed into pure light!", "The Tempest")
        ]
    },
    "optimization": {
        -5: [
            ("Striving to better, oft we mar what's well.", "King Lear"),
            ("O leap into the dark! The gradient vanishes!", "Hamlet"),
            ("Lost in the infinite loop of endless regret!", "Macbeth")
        ],
        -3: [
            ("Wisely and slow; they stumble that run fast.", "Romeo and Juliet"),
            ("To climb steep hills requires slow pace at first.", "Henry VIII"),
            ("We have scorched the snake, not killed it.", "Macbeth")
        ],
        -1: [
            ("Too swift arrives as tardy as too slow.", "Romeo and Juliet"),
            ("Present fears / Are less than horrible imaginings.", "Macbeth")
        ],
        0: [
            ("The web of our life is of a mingled yarn, good and ill together.", "All's Well That Ends Well"),
            ("There is a tide in the affairs of men / Which, taken at the flood, leads on to fortune.", "Julius Caesar"),
            ("Moderation is the silken string running through the pearl chain of all virtues.", "Coriolanus")
        ],
        1: [
            ("Better a witty fool than a foolish wit.", "Twelfth Night"),
            ("The end crowns all.", "Troilus and Cressida")
        ],
        3: [
            ("How far that little candle throws his beams!", "The Merchant of Venice"),
            ("True hope is swift, and flies with swallow's wings.", "Richard III")
        ],
        5: [
            ("To the highest peak of excellence we ascend!", "Henry V"),
            ("Conquered is the loss function! Absolute minimum reached!", "Sonnet 116"),
            ("The crown of efficiency shines upon us!", "The Tempest")
        ]
    },
    "math_tragedy": {
        -5: [
            ("Out, out, brief candle! Life's but a walking shadow...", "Macbeth"),
            ("Nothing will come of nothing: speak again.", "King Lear"),
            ("O, break, my heart! Poor bankrupt, break at once!", "Romeo and Juliet")
        ],
        -3: [
            ("The time is out of joint: O cursed spite!", "Hamlet"),
            ("Blow, winds, and crack your cheeks! Rage! Blow!", "King Lear"),
            ("I am a man more sinned against than sinning.", "King Lear")
        ],
        -1: [
            ("We know what we are, but know not what we may be.", "Hamlet"),
            ("Lord, what fools these mortals be!", "A Midsummer Night's Dream")
        ],
        0: [
            ("To be, or not to be, that is the question.", "Hamlet"),
            ("All that glisters is not gold.", "The Merchant of Venice")
        ],
        1: [
            ("Strong reasons make strong actions.", "King John"),
            ("The golden age returns again.", "Henry VIII")
        ],
        3: [
            ("Miracles are ceased; and therefore we must needs admit the means.", "Henry V"),
            ("The skies are clear, the numbers harmonized!", "The Tempest")
        ],
        5: [
            ("O brave new world, / That has such people in't!", "The Tempest"),
            ("Singularities conquered! Absolute matrix triumph!", "Sonnet 55"),
            ("The stars themselves dance in mathematical bliss!", "Pericles")
        ]
    },
    "file_io": {
        -5: [
            ("Where is the world? Of shadow and of naught!", "Timon of Athens"),
            ("I pathless tread, where light itself turns pale.", "Pericles"),
            ("I naturally despise a coward, but a missing file is sheer torment!", "Twelfth Night")
        ],
        -3: [
            ("What, is the jay more precious than the lark?", "The Taming of the Shrew"),
            ("Lost, lost, all lost! My kingdom for a canvas!", "Richard III"),
            ("An empty space where vision should abide.", "Othello")
        ],
        -1: [
            ("The path is blocked by shadows of uncertainty.", "Hamlet"),
            ("A dark domain without a single pixel.", "Macbeth")
        ],
        0: [
            ("Open your ears; for which of you will stop / The vent of hearing when loud Rumour speaks?", "Henry IV, Part 2"),
            ("Read the page, reveal the hidden truth.", "Measure for Measure")
        ],
        1: [
            ("Found is the portal, open is the gate.", "Merchant of Venice"),
            ("The bytes flow smoothly into memory.", "The Tempest")
        ],
        3: [
            ("Welcome, pure canvas of endless possibilities!", "The Tempest"),
            ("A treasure trove uncovered from the disk!", "Henry VIII")
        ],
        5: [
            ("The gates of file domain open in pure splendor!", "Henry V"),
            ("A glorious canvas loaded without a flaw!", "Sonnet 18")
        ]
    },
    "tests_tdd": {
        -5: [
            ("O, what a noble mind is here o'erthrown!", "Hamlet"),
            ("Failed is the assertion! The edifice crumbles!", "Titus Andronicus"),
            ("An error most foul, unnatural and dark!", "Hamlet")
        ],
        -3: [
            ("Self-love, my liege, is not so vile a sin / As self-neglecting.", "Henry V"),
            ("A flaw in our defense! The test has fallen!", "Julius Caesar")
        ],
        -1: [
            ("A little more than kin, and less than kind.", "Hamlet"),
            ("Doubtful it stands, as two spent swimmers that do cling together.", "Macbeth")
        ],
        0: [
            ("Safe bind, safe find; / A proverb never stale in thrifty mind.", "The Merchant of Venice"),
            ("Test all things; hold fast that which is good.", "Henry V")
        ],
        1: [
            ("Well begun is half done.", "Aristotle / Shakespearean adaptation"),
            ("The invariant holds firm against the tide.", "Coriolanus")
        ],
        3: [
            ("Praise the test that guards the castle walls!", "Henry V"),
            ("All assertions green, like spring in Stratford!", "As You Like It")
        ],
        5: [
            ("Flawless victory! Every test stands immutable!", "Henry V"),
            ("A fortress of code, impervious to decay!", "Sonnet 55")
        ]
    },
    "psnr_bench": {
        -5: [
            ("O, what a fall was there, my countrymen!", "Julius Caesar"),
            ("Low PSNR! The image dissolves in chaotic noise!", "Timon of Athens"),
            ("A blurred nightmare, devoid of form and grace!", "Macbeth")
        ],
        -3: [
            ("Lord, what fools these mortals be!", "A Midsummer Night's Dream"),
            ("Faint lines, obscured in shadows of despair.", "Hamlet")
        ],
        -1: [
            ("Like as the waves make towards the pebbled shore, / So do our minutes hasten to their end.", "Sonnet 60"),
            ("Acceptable, yet far from heaven's light.", "Romeo and Juliet")
        ],
        0: [
            ("We know what we are, but know not what we may be.", "Hamlet"),
            ("A fair reconstruction, true to its source.", "Twelfth Night")
        ],
        1: [
            ("I see a world reborn in lines of light!", "The Tempest"),
            ("Clearer shines the canvas with every step.", "Merchant of Venice")
        ],
        3: [
            ("O, how much more doth beauty beauteous seem!", "Sonnet 54"),
            ("High fidelity achieved! The pixels rejoice!", "The Tempest")
        ],
        5: [
            ("The marathon is run; the crown of light is won!", "Henry V"),
            ("Peak signal reached! Perfection in every pixel!", "Sonnet 18"),
            ("A masterpiece of reconstruction, divine and pure!", "The Tempest")
        ]
    }
}


def get_quote_by_pathos(topic: str = "geometry", pathos: int = 0) -> tuple:
    """
    Finds a quote for the given topic matching the requested pathos level (-5 to +5).
    Clamps pathos to [-5, +5] and picks a random quote from available entries.
    """
    pathos = max(-5, min(5, pathos))
    
    topic_quotes = SHAKESPEARE_PATHOS_DB.get(topic, SHAKESPEARE_PATHOS_DB["geometry"])
    
    available_levels = sorted(topic_quotes.keys())
    closest_level = min(available_levels, key=lambda lvl: abs(lvl - pathos))
    
    quote, play = random.choice(topic_quotes[closest_level])
    return quote, play, closest_level


def format_dramatic_box(title: str, topic: str = "geometry", pathos: int = 0, width: int = 70) -> str:
    """Formats a drama box with explicit Pathos Level (-5 to +5) and Verbal Status."""
    quote, play, actual_pathos = get_quote_by_pathos(topic, pathos)
    label = PATHOS_LABELS.get(actual_pathos, "NORMAL")

    lines = quote.split(" / ")
    border = "+" + "-" * (width - 2) + "+"

    box_lines = [border]

    sign = "+" if actual_pathos > 0 else ""
    header_text = f"| [{title.upper()}] [PATHOS: {sign}{actual_pathos} ({label})]"
    box_lines.append(header_text.ljust(width - 1) + "|")

    for line in lines:
        body_text = f"|   '{line}'"
        box_lines.append(body_text.ljust(width - 1) + "|")

    box_lines.append("|".ljust(width - 1) + "|")

    attr_text = f"|   -- William Shakespeare, {play}"
    box_lines.append(attr_text.ljust(width - 1) + "|")
    box_lines.append(border)

    return "\n".join(box_lines)


# =============================================================================
#                                MODULE TESTS
# =============================================================================

if __name__ == "__main__":
    print("[TEST] Running shakespeare.py module tests...")

    # 1. Test Nightmare Labeling
    q, p, lvl = get_quote_by_pathos("math_tragedy", pathos=-5)
    assert lvl == -5, f"Pathos -5 must be mapped correctly, got {lvl}"
    assert PATHOS_LABELS[-5] == "NIGHTMARE", "Pathos -5 label must be NIGHTMARE"

    # 2. Test Randomness Entropy
    quotes_seen = set()
    for _ in range(20):
        q, _, _ = get_quote_by_pathos("geometry", pathos=5)
        quotes_seen.add(q)
    assert len(quotes_seen) > 1, "Randomization must yield varied quotes across multiple calls"

    # 3. Test Box Generation
    box_nightmare = format_dramatic_box("SVD Disintegrated", topic="math_tragedy", pathos=-5)
    assert "NIGHTMARE" in box_nightmare, "Box must contain NIGHTMARE label for pathos -5"

    print("[SUCCESS] All shakespeare.py tests passed cleanly!\n")
    print(box_nightmare)