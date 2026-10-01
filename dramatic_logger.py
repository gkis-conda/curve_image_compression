"""
===============================================================================
             DRAMATIC LOGGERS & PHILOSOPHICAL ENGINE
===============================================================================

  Provides philosophical diagnostics with customizable Pathos (-5..+5).

===============================================================================
"""

import sys
import numpy as np
from shakespeare import format_dramatic_box


class ExistentialMatrixError(Exception):
    """Raised when a matrix collapses into singular nothingness."""
    pass


class VanishingGradientLament(Exception):
    """Raised when energy/entropy dissolves into silence."""
    pass


def log_event(title: str, topic: str = "geometry", pathos: int = 0):
    """Logs an explicit drama box for any system event."""
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
#                                MODULE TESTS
# =============================================================================

if __name__ == "__main__":
    print("[TEST] Running dramatic_logger.py module tests...")

    # 1. Test Regular Event Logging
    log_event("Tree Split Root Created", topic="tree_split", pathos=0)
    log_event("PSNR Target Exceeded", topic="psnr_bench", pathos=5)

    # 2. Test Decorator with Nightmare Pathos
    @dramatic_guard(topic="math_tragedy", failure_pathos=-5)
    def failing_inv():
        return np.linalg.inv(np.array([[1.0, 1.0], [1.0, 1.0]]))

    try:
        failing_inv()
        assert False, "Should have thrown ExistentialMatrixError"
    except ExistentialMatrixError:
        print("  [OK] Intercepted singular matrix with nightmare pathos guard.")

    print("\n[SUCCESS] All dramatic_logger.py tests passed cleanly!")