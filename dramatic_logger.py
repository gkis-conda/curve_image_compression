"""
===============================================================================
                       DRAMATIC LOGGER MODULE
===============================================================================

  "Speak the speech, I pray you, as I pronounced it to you, 
   trippingly on the tongue..."
  -- William Shakespeare, Hamlet

  Provides philosophical diagnostics by assembling messages and quotes 
  into clean theatrical ASCII boxes with word-boundary wrapping and 
  metric unit orphan protection.

===============================================================================
"""

import sys
import re
import textwrap
import numpy as np
from shakespeare import get_quote_by_pathos, PATHOS_LABELS


class ExistentialMatrixError(Exception):
    """Raised when a matrix collapses into singular nothingness."""
    pass


class VanishingGradientLament(Exception):
    """Raised when energy/entropy dissolves into silence."""
    pass


def _wrap_safe(text: str, width: int) -> list:
    """
    Wraps text to specified width on word boundaries while preventing 
    metric units (dB, ms, KB, bpp, etc.) from becoming orphans on a new line.
    """
    # Protect numbers/words followed by common units from breaking apart
    unit_pattern = re.compile(r'\b(\d+(?:\.\d+)?|[A-Za-z]+)\s+(dB|ms|KB|B|bpp|px|%)\b', re.IGNORECASE)
    
    # Use a non-whitespace control character as placeholder during textwrap
    placeholder = "\u001F"
    protected_text = unit_pattern.sub(r'\1' + placeholder + r'\2', text)
    
    wrapped_lines = []
    for paragraph in protected_text.splitlines():
        if not paragraph.strip():
            wrapped_lines.append("")
        else:
            wrapped = textwrap.wrap(
                paragraph, 
                width=width, 
                break_long_words=False, 
                break_on_hyphens=False
            )
            for line in wrapped:
                # Restore placeholder back to a normal space
                restored_line = line.replace(placeholder, " ")
                wrapped_lines.append(restored_line)
                
    return wrapped_lines if wrapped_lines else [""]


def format_dramatic_box(title: str, topic: str = "geometry", pathos: int = 0, width: int = 70) -> str:
    """
    Formats a theatrical ASCII box. 
    Topic defines the header banner, while title content is safely wrapped in the body.
    """
    quote, play, actual_pathos = get_quote_by_pathos(topic, pathos)

    border = "+" + "-" * (width - 2) + "+"
    box_lines = [border]

    # Header displays cleanly the topic category
    header_text = f"| [{topic.upper()}]"
    box_lines.append(header_text.ljust(width - 1) + "|")
    box_lines.append("|".ljust(width - 1) + "|")

    body_width = width - 6

    # Wrap message/title paragraphs with unit protection
    for paragraph in title.splitlines():
        if not paragraph.strip():
            box_lines.append("|".ljust(width - 1) + "|")
        else:
            wrapped_para = _wrap_safe(paragraph, width=body_width)
            for w_line in wrapped_para:
                body_text = f"|   {w_line}"
                box_lines.append(body_text.ljust(width - 1) + "|")

    box_lines.append("|".ljust(width - 1) + "|")

    # Wrap quote parts with unit protection
    for quote_part in quote.split(" / "):
        wrapped_quote = _wrap_safe(quote_part, width=body_width)
        for q_line in wrapped_quote:
            quote_text = f"|   '{q_line}'"
            box_lines.append(quote_text.ljust(width - 1) + "|")

    box_lines.append("|".ljust(width - 1) + "|")

    # Play attribution
    attr_text = f"|   -- William Shakespeare, {play}"
    box_lines.append(attr_text.ljust(width - 1) + "|")
    box_lines.append(border)

    return "\n".join(box_lines)


def log_event(title: str, topic: str = "geometry", pathos: int = 0):
    """Logs an explicit formatted drama box for any system event."""
    box = format_dramatic_box(title, topic=topic, pathos=pathos)
    stream = sys.stderr if pathos < 0 else sys.stdout
    print(box, file=stream)


def dramatic_guard(topic: str = "math_tragedy", failure_pathos: int = -5):
    """
    Decorator that intercepts mathematical crashes and prints 
    Shakespearean drama boxes with specified pathos.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except np.linalg.LinAlgError as e:
                log_event("Geometric Collapse / Singular Matrix", topic=topic, pathos=failure_pathos)
                raise ExistentialMatrixError("Singular matrix encountered") from e
            except ZeroDivisionError as e:
                log_event("Abyss of Zero Division", topic=topic, pathos=failure_pathos)
                raise VanishingGradientLament("Division by zero") from e
            except Exception as e:
                log_event("Unknown Shadow of Fate", topic=topic, pathos=failure_pathos)
                raise e
        return wrapper
    return decorator


# =============================================================================
#                            MODULE TESTS
# =============================================================================

def test_dramatic_logger():
    """Test suite for dramatic_logger module with metric unit protection checks."""
    print("[TEST] Running dramatic_logger.py module tests...")

    # 1. Test Regular Event Logging with metric units near boundary
    log_event("Tree Split Root Created", topic="tree_split", pathos=0)
    log_event("Compression Completed successfully with metrics PSNR 38.5 dB and latency 12.0 ms recorded", topic="psnr_bench", pathos=5)

    # 2. Test Decorator with Nightmare Pathos
    @dramatic_guard(topic="math_tragedy", failure_pathos=-5)
    def failing_inv():
        return np.linalg.inv(np.array([[1.0, 1.0], [1.0, 1.0]]))

    try:
        failing_inv()
        assert False, "Should have thrown ExistentialMatrixError"
    except ExistentialMatrixError:
        print("  [OK] Intercepted singular matrix with nightmare pathos guard.")

    print("[SUCCESS] All dramatic_logger.py tests passed cleanly!")


if __name__ == "__main__":
    test_dramatic_logger()