"""Cell formatting shared by the table generators: a ratio with its per-trace SD as a lower-right subscript, and shading.

Shading encodes baseline / SharedScan on a log scale: white at or below 1.05 (the tie band), saturating at 3x. Values are always printed."""
import math
import statistics as st


def shade(r):
    if r <= 1.05:
        return ""
    return f"\\cellcolor{{ratioblue!{round(32 * min(1.0, math.log(r) / math.log(3)))}!white}}"


def ratio(r, per_trace, bold=False):
    """r is the ratio of means shown as the main value; per_trace holds the ratio of each trace, whose sample SD is the subscript."""
    v = f"{r:.2f}"
    sd = f"{st.stdev(per_trace):.2f}" if len(per_trace) > 1 else ""  # one trace has no SD; none is invented
    return f"{shade(r)}\\rsd{{{chr(92) + 'textbf{' + v + '}' if bold else v}}}{{{sd}}}"
