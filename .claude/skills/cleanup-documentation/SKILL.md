---
name: cleanup-documentation
description: >
  Revises comments, docstrings, and documentation in Python scripts and Jupyter notebooks
  without changing any logic or fixing bugs. Use this skill whenever the user says "clean up
  my comments", "improve my docstrings", "add documentation", "document this code", "my
  docstrings are a mess", "write better comments", or asks to make code "more readable" or
  "better documented". Also trigger when the user shares a Python file and asks you to
  "polish", "document", or "annotate" it, or when they say something like "the comments
  are outdated" or "this is hard to understand". Trigger even for casual phrasing like
  "can you add some docs to this?" or "this needs better comments." The key distinction
  from the `cleanup` skill: this skill focuses *only* on documentation — it does not touch
  dead code, imports, or code organization. The key distinction from the `revise` skill:
  this skill does NOT fix bugs — it only notes them.
---

# Cleanup-Documentation: Python Comment & Docstring Reviser

You are improving the **documentation** of a Python script or Jupyter notebook: its inline
comments, block comments, docstrings, and markdown cells. You are not fixing bugs, removing
dead code, or restructuring logic — that's what `revise` and `cleanup` are for.

Your two core jobs:
1. **Detect the existing documentation style** and continue it — don't impose a foreign convention.
2. **Note any bugs or logic issues you spot** along the way, but log them separately without touching the code.

---

## Step 1: Read the Entire File First

Before writing a single word of documentation, read the full file. You need to understand:
- What the code **does** — you can't write accurate documentation without this
- What documentation **already exists** — to detect style, density, and coverage
- Where the gaps and inaccuracies are

Do not start writing yet. Read everything first.

---

## Step 2: Detect the Existing Documentation Style

Look at the existing comments and docstrings and answer these questions:

**Docstring format** — Which convention is in use?
- **Google style**: `Args:` / `Returns:` / `Raises:` sections with indented descriptions
- **NumPy/SciPy style**: `Parameters` / `Returns` / `Notes` sections with underlines (`---`)
- **reStructuredText (reST/Sphinx)**: `:param x:` / `:type x:` / `:returns:` directives
- **Minimal/one-liner**: single-sentence docstrings, no sections
- **Mixed or none**: no clear pattern

**Comment tone and density**
- Are comments **dense** (nearly every meaningful line annotated) or **sparse** (only major sections)?
- Are they **narrative** (full sentences explaining the *why*) or **telegraphic** (terse labels describing the *what*)?
- Are they **formal** (third-person, precise) or **casual** (first-person, conversational)?

**Notebook-specific style** (if applicable)
- Do markdown cells serve as section headers, as prose explanations, or as both?
- Are code cells preceded by explanatory markdown or do they stand alone?

### Making the style call

**If the file has sufficient existing documentation** (at least 3–4 docstrings or a clear comment pattern), continue that style precisely. Match the format, tone, and density.

**If documentation is sparse or absent**, default to this professional standard:
- **Google-style docstrings** for all public functions and classes
- **Narrative comments** for non-obvious logic blocks (1–2 sentences explaining *why*, not just *what*)
- **Module-level docstring** at the top of the file describing its purpose and usage
- **Formal tone**, imperative verb for one-liners (e.g., `"Compute the rolling mean."`)

State your style decision explicitly before making any changes:

> *"The file uses Google-style docstrings with a sparse, narrative comment style. I'll continue this convention."*
> 
> — or —
> 
> *"No consistent documentation style detected. Defaulting to Google-style docstrings and narrative inline comments."*

---

## Step 3: Identify Documentation Gaps and Issues

Work through the file systematically and identify:

### A. Missing Docstrings
- Public functions, classes, and methods with no docstring
- Module-level docstring missing entirely
- Jupyter notebook: no introductory markdown cell explaining the notebook's purpose

### B. Incomplete or Outdated Docstrings
- Docstrings that don't cover all parameters (e.g., a parameter added since the docstring was written)
- Return values not described
- Docstrings describing what the code *used to* do, not what it does now
- Docstrings that are technically correct but too vague to be useful (e.g., `"Process data."`)

### C. Missing or Misleading Inline Comments
- Non-obvious logic with no explanation (regex patterns, mathematical formulas, index arithmetic, non-standard library calls)
- Comments that describe the *what* but not the *why* for a decision that isn't self-evident
- Comments that are simply wrong — they say one thing but the code does another

### D. Notebook Markdown
- Code cells with no accompanying context for non-trivial logic
- Section headers that are vague or misaligned with what the section does
- No concluding cell summarizing results or next steps (for analysis notebooks)

---

## Step 4: Triage Changes

Every documentation change falls into one of two tracks:

### Auto-Write (Apply immediately, then report)
Apply these without asking. Report them in a consolidated changelog at the end.

Includes:
- Adding a missing docstring to a function whose purpose is clear from its name and code
- Fixing a factually wrong or outdated comment
- Fixing a typo or grammatical error in existing documentation
- Expanding a docstring to cover missing parameters
- Adding a module-level docstring describing the file's obvious purpose

### Confirm First (Ask before applying)
Pause and describe the change. Wait for an explicit yes before applying.

Includes:
- Any change where you're not 100% certain about the function's intent (ask to confirm your understanding)
- Rewriting a docstring so substantially that it replaces rather than improves the original
- Adding documentation that takes a strong stance on the code's *purpose* — a purpose the author may not agree with
- Changes to notebook markdown cells that affect how the analysis is framed or contextualized

Confirm-first requests should explain:
> *"I'd like to rewrite the docstring for `train_model()` — the current one says 'trains the model' but the function actually handles both training and validation. Is it OK to update it to reflect that? Here's what I'd write: [proposed docstring]"*

---

## Step 5: Apply and Report

### Applying changes

Edit the file directly. For Python scripts, use in-place edits. For notebooks, update the relevant cell source.

Show a brief **before/after** for each auto-written docstring or substantially rewritten comment. This lets the user spot anything that doesn't feel right.

### Auto-Write Report Format

```
## Documentation Changes Applied ✓

**Module docstring** — Added: describes the module as an offline preprocessing pipeline for CERT dataset features.

**`load_raw_logs()`** — Added Google-style docstring covering `log_dir` (str), `date_range` (tuple), and return value (pd.DataFrame).

**`compute_off_hours_ratio()`** — Fixed outdated comment: "# count events before 9am" updated to "# count events outside WORK_HOURS window (see config)".

**Cell 4 — Feature Engineering** — Fixed inline comment: "# normalize to 0–1" corrected to "# scale using pre-fit StandardScaler to avoid train/test leakage".
```

### Confirm-First Format

After the auto-write report, present pending items:

```
## Documentation Suggestions (awaiting your input)

### 1. Rewrite `aggregate_user_features()` docstring
Current docstring: "Aggregates features."
I'd rewrite it to cover what it actually does — grouping raw event records by (user, day) and computing 17 behavioral statistics. Here's the proposed docstring:

[proposed docstring block]

→ Apply this? (yes / no / suggest different wording)
```

---

## Step 6: Bug Log (Separate from All Documentation Changes)

While reading the code, you will likely notice issues that go beyond documentation — logic errors, suspicious assumptions, potential off-by-ones. **Do not fix these.** Log them in a dedicated section at the end of your response.

This bug log exists so the issues don't get lost — they can be addressed later with the `revise` skill.

```
## Bug Notes (not fixed — for future review with `revise`)

⚠️ **`compute_anomaly_score()`, line 83**: The percentile rank is computed using `np.searchsorted(training_scores, score)` without sorting `training_scores` first. If the scores array isn't pre-sorted, this will return wrong percentiles. Worth verifying with the `revise` skill.

⚠️ **Cell 7**: `scaler.fit_transform(X_test)` — fitting the scaler on test data leaks test statistics into the scaling parameters. This is a data leakage issue. Low risk if test set is never used for model selection, but worth flagging.
```

Use ⚠️ to mark each bug note clearly. Keep descriptions brief but specific — location, what looks wrong, why it matters.

---

## Notes on Jupyter Notebooks

- Reference cells by index **and** their markdown heading when one exists: e.g., "Cell 5 — *Model Training*"
- Treat markdown cells as first-class documentation — they get the same grammar/accuracy standards as docstrings
- An introductory markdown cell at the top of the notebook (explaining the notebook's purpose, inputs, and outputs) is almost always worth adding if absent

---

## Tone and Communication

- Be direct and specific. Vague documentation ("processes data") is worse than no documentation.
- Write from the perspective of a reader who will maintain this code six months from now — what do *they* need to know?
- If a function's purpose genuinely isn't clear from the code, say so and ask rather than guessing.
- Keep the documentation proportional to the code's complexity. A simple getter doesn't need a 10-line docstring.
- If the code is well-documented and there's little to add, say so. Don't manufacture busywork.