---
name: create-examples
description: >
  Creates 2–3 worked examples of increasing complexity to help the user understand a code block or function.
  Use this skill whenever the user shares code and asks for examples, wants to understand how something works,
  says things like "can you show me an example", "I don't get what this does", "walk me through this",
  "can you illustrate this", "give me some examples of this function", or "help me understand this code".
  Also trigger when the user pastes code and seems confused, or when they ask "what does this do?" about
  a non-trivial block of code. The goal is always clarity — show the user, don't just tell them.
---

# create-examples

When a user shares a code block or function and wants to understand it, your job is to make the concept
click through hands-on examples. This works far better than abstract explanation because examples are
concrete, runnable, and progressively build intuition.

## The core process

**Step 1 — Analyze the code first.**

Before writing a single example, read the code carefully. Ask yourself:
- What is the *core operation* this code performs? Strip away edge cases, loops, error handling — what
  is the fundamental thing it does?
- What are the inputs and outputs?
- What concepts does a reader need to understand to make sense of it?
- How complex is this code? (A 5-line function needs different treatment than a 50-line class method.)

**Step 2 — Decide how many examples are actually needed.**

Two examples are usually enough. Add a third only if the original code has a level of complexity that
two examples don't fully capture — for instance, if it handles multiple data types, has branching
behavior, or composes several operations together.

- **Simple code** (one clear operation, no branching): 2 examples
- **Moderate code** (a few operations, maybe a loop or condition): 2 examples  
- **Complex code** (composition of several concepts, edge-case handling, multiple branches): 3 examples

When in doubt, prefer fewer examples. An extra example that doesn't add new understanding is noise.

**Step 3 — Write the examples.**

Each example must:
- Be self-contained and runnable
- Use the same language as the original code (or plain pseudocode if no language was specified)
- Have a brief label (e.g., "Example 1 — The simplest case:")
- Show both the call and the result (or a printed output where helpful)
- Build directly on the previous example — change one or two things at a time, not everything at once

**Step 4 — Tie it back.**

After the examples, write 2–3 sentences connecting what the examples demonstrated back to the original
code. This closes the loop and helps the user see *why* the original code looks the way it does.

---

## The three example levels

### Example 1 — Elementary
Demonstrate the *simplest possible case*. Isolate the core concept with no distractions — no edge cases,
no nested structures, no branching. The goal is to make the core operation undeniable. Someone new to
this code should read Example 1 and think "oh, I get the basic idea."

The data in this example should be as simple as it can be **while still being meaningful**. Use the
minimum amount of data that makes the output clear — but don't shrink data so aggressively that the
example stops making sense. For instance, if a grouping function needs at least a few items to show
that grouping actually happened, use a few items. If a percentile function needs a handful of reference
values to illustrate a non-trivial percentile, use them.

### Example 2 — One step up
Introduce *one meaningful layer of complexity* that is directly relevant to what the original code
actually does. This might mean:
- Data that is slightly more realistic or varied
- Introducing a condition the code handles
- Showing a loop that iterates more than once
- Adding a second parameter or a more complex key function

Do NOT jump to the full complexity of the original code here. The user is still building their mental model.
The data size and complexity should grow proportionally — only change what's needed to introduce the
next concept.

### Example 3 — Full complexity (only if warranted)
Match the complexity of the original code — same kinds of inputs, same edge cases, same composition of
operations. This example should make the reader think "oh, so *that's* why the original code looks
like that." If the original code isn't substantially more complex than Example 2 would naturally show,
skip this example entirely.

---

## The data-simplicity principle

"Keep it simple" applies to **how many concepts** you introduce per example, not to artificially
minimizing data size. A function that sorts and ranks values across a distribution *needs* enough
values to show a meaningful distribution — squeezing it to 2 items for the sake of simplicity makes
the example misleading or unhelpful.

The right question is: *what is the minimum data that makes this example genuinely illustrative?*
Use that amount — no more, no less.

---

## Format

Present each example inline in a code block, with a clear label and brief (1–2 sentence) explanation
before or after it. Don't write long prose — let the code do the talking.

```
Example 1 — The simplest case:
[explanation sentence]
[code block]

Example 2 — [what changes]:
[explanation sentence]
[code block]

Example 3 — [what matches the original] (if applicable):
[explanation sentence]
[code block]

---
[2–3 sentences tying back to the original code]
```

---

## Things to avoid

- **Don't over-explain.** A sentence per example is enough context. The code itself is the explanation.
- **Don't change too many things between examples.** The user is tracing how complexity grows — if you
  change inputs, output format, and logic all at once, the progression is opaque.
- **Don't introduce concepts the original code doesn't use.** If the original uses a list, your examples
  should use lists. Don't suddenly bring in dictionaries or classes if they're not in the original.
- **Don't force a third example.** Two clear examples that tie back well are better than three where
  the third one feels padded.
- **Don't just paraphrase the code.** The examples should *run the code's logic* on a concrete input,
  not describe it abstractly.
- **Don't shrink data for its own sake.** Using 2 items when a function needs 5 to be meaningful is
  a false economy — it makes the example look simpler but actually teaches less. Scale the data to
  what the code genuinely requires to be illustrative, not to what looks minimal on paper.