---
name: explain-function
description: Produce a complete, structured explanation of a single Python function — a plain-language summary of what question it answers, a two-column table mapping each code segment to a technical breakdown, a worked example with an input-to-output table, and a recommended Google-style docstring with inline comments. Use this whenever the user shares a function (or points at one in their code) and asks to "explain", "break down", "walk me through", "unpack", "document", or "help me understand" it — even casually, e.g. "what does this do?" or "can you make sense of this for me?". Prefer this over a one-line inline answer whenever the user wants to genuinely understand a function rather than just get a quick fact. Python-focused.
---
 
# Explain Function
 
Turn a single Python function into a clear, four-part explanation. The goal is understanding: someone should finish reading and know *what the function is for*, *how each piece works*, *what it does on a concrete input*, and *how to document it well*.
 
Produce all four sections below, in order, every time. They build on each other — the plain-language framing primes the reader for the technical table, the table primes them for the worked example, and the example primes them for the documentation. Skipping one breaks the staircase.
 
Respond inline in chat using Markdown (so the table and code blocks render). Don't write to a file unless the user asks.
 
## Output structure
 
Always use these four sections, with these exact headings:
 
```
## What this function answers
## Breakdown
## Worked example
## Recommended docstring & comments
```
 
### 1. What this function answers
 
Two to four sentences, **no jargon**. Frame the function as the answer to a question a non-programmer might ask — "how spread out are these numbers?", "is this user allowed in?", "what's the cheapest route?". Name the question, then say in plain terms how the function settles it. Avoid type names, library names, and control-flow vocabulary here; that's what the next section is for. If you find yourself writing "iterates" or "returns a dict", you've drifted too technical — pull back.
 
### 2. Breakdown
 
A two-column Markdown table. Split the function into **logical segments** — a guard clause, the main loop, an accumulation step, the return — rather than one row per physical line. A segment is "one idea." Group lines that work together; split when the purpose changes.
 
- **Left column ("Code"):** the exact code for that segment, verbatim, indentation preserved. Wrap it in backticks. For multi-line segments, separate lines with `<br>` so the cell stays readable.
- **Right column ("What it does"):** a precise technical explanation — what the code does, *why* it's there, and any subtlety worth flagging (edge cases, mutation, complexity, a non-obvious idiom). This is where the real vocabulary lives.
Keep the left column faithful — never paraphrase or "clean up" the user's code in this table. If their code has a quirk, the breakdown is where you explain it, not where you silently fix it.
 
### 3. Worked example
 
Pick one realistic, easy-to-follow input — concrete values, not `x` and `y`. Then show the operation as a **simple input → output table** so the transformation is visible at a glance, and state the final return value explicitly.
 
Use a small table whose rows trace the meaningful intermediate states (e.g. each iteration, or each transformation stage), with a column for the input/state and a column for the result after that step. Keep it short — three to five rows is usually enough to make the mechanism click. Close with the final output on its own line: `Output: ...`.
 
Choose inputs that exercise the *interesting* path, not a trivial empty case. If the function has an edge case worth seeing (an empty list, a tie, a missing key), that's often the most illuminating example.
 
### 4. Recommended docstring & comments
 
Provide a ready-to-paste version of the function with documentation added. The style is **Google-style with these specific conventions** — follow them exactly:
 
**Summary line:** Imperative mood, ends with a period. Can carry a colon-elaboration in the same line when the function name alone isn't self-explanatory — `"Tournament odds: probabilities per team from the latest Monte Carlo run."` — but keep it to one line.
 
**Description paragraph:** After a blank line, write a substantive paragraph — not a restatement of the summary. Include behavioral notes, edge cases, and relevant context. If the function can raise an error, mention it inline here (`"Raises 503 if no predictions have loaded."`) rather than in a formal `Raises:` block. Skip this paragraph only if the summary is genuinely complete.
 
**`Args:`** Standard Google format: `name (type): description`. Descriptions explain context that isn't obvious from the name alone — how an argument arrives (e.g. "injected by dependency provider"), what range it accepts, or what happens when it's absent. Do not end arg descriptions with a period.
 
**`Returns:`** Inline format on one line: `TypeName: description.` — not the indented multi-line Google variant. Use the real return type name (the class, `list`, `dict`, `float`, etc.), follow it with a colon and a short description, and end with a period. For lists and dicts, make the contents explicit: `list: Array of MatchPrediction objects.` / `dict: Dictionary with keys [generated_at, n_simulations, odds].`
 
**`Raises:`** Do not add a formal `Raises:` block. Mention errors in the description paragraph instead.
 
**Inline comments:** Sparse and purposeful. Comments explain *why*, never *what*. A comment that restates what the code obviously does is noise and should be omitted. Use standalone comment lines above the relevant code (not trailing on the same line, except for very short annotations). Section dividers (`# --- Label ---`) are appropriate when a function has distinct phases.
 
Present the result as a single Python code block so the user can copy it directly.
 
## Full demonstration
 
Here is the complete shape of a good response, for this function:
 
```python
def normalize(values):
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]
```
 
---
 
## What this function answers
 
"Given a list of numbers, how does each one rank between the smallest and the largest?" This function rescales every value so the smallest becomes 0, the largest becomes 1, and everything else lands proportionally in between — a common way to put numbers on a level playing field before comparing or combining them.
 
## Breakdown
 
| Code | What it does |
|---|---|
| `if not values:`<br>`    return []` | Guard clause for the empty-input case. An empty list has no min or max, so the later math would fail — returning `[]` early sidesteps that and gives a sensible result. |
| `lo, hi = min(values), max(values)` | Captures the range endpoints in one tuple-unpacking assignment. These two values define the scale that every element is mapped onto. |
| `if hi == lo:`<br>`    return [0.0] * len(values)` | Handles the all-identical case. When every value is the same, `hi - lo` is `0`, which would cause a division-by-zero below; collapsing everything to `0.0` is the convention here. |
| `return [(v - lo) / (hi - lo) for v in values]` | The core min-max formula, applied to each element via a list comprehension. Subtracting `lo` shifts the smallest value to 0; dividing by the span `hi - lo` scales the largest to 1. |
 
## Worked example
 
Input: `values = [10, 20, 40]` → `lo = 10`, `hi = 40`, span `= 30`.
 
| Value `v` | `(v - lo) / span` | Result |
|---|---|---|
| 10 | (10 − 10) / 30 | 0.0 |
| 20 | (20 − 10) / 30 | 0.333… |
| 40 | (40 − 10) / 30 | 1.0 |
 
Output: `[0.0, 0.3333333333333333, 1.0]`
 
## Recommended docstring & comments
 
```python
def normalize(values):
    """Scale a list of numbers to the [0, 1] range using min-max normalization.
 
    The smallest value maps to 0.0 and the largest to 1.0, with all others
    placed proportionally between them. Returns [] for empty input; returns
    all-zeros if every value is identical (avoids division by zero).
 
    Args:
        values (list[float]): The numbers to normalize.
 
    Returns:
        list[float]: Normalized values in the same order as the input.
    """
    if not values:
        return []
 
    lo, hi = min(values), max(values)
 
    # Span is zero when all values are identical — return flat zeros rather than divide by zero.
    if hi == lo:
        return [0.0] * len(values)
 
    return [(v - lo) / (hi - lo) for v in values]
```
 
---
 
## Reminders
 
- The four sections are a ladder from intuition to detail — keep that order so each section sets up the next.
- Section 1 is for the non-technical reader; sections 2–4 carry the technical load. Don't blur them.
- Keep the user's code verbatim in the breakdown table; explain quirks rather than quietly correcting them. (If the code has a genuine bug, mention it after the four sections so it isn't lost — but the explanation itself should describe the code as written.)
- Match the worked example to the function's actual domain when you can tell what it is (ML, web, data wrangling); a relevant example lands harder than a generic one.