from typing import List


def normalize_equation_strings(equations: List[str]) -> List[str]:
    # trivial normalization hook; extend later
    return [e.strip() for e in equations if e and e.strip()]


def to_sympy(expr: str):
    try:
        # placeholder: import sympy lazily
        import sympy as sp
        # very naive parse; replace with latex parsing if needed
        return sp.sympify(expr)
    except Exception:
        return None


