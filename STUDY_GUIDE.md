# 🌍 Asteroid Danger Detector: Study Guide

> **The big question:** Can a computer learn NASA's danger rule just by looking at examples?

---

## The big picture

Each file solves **one problem**, and each one leads to the next.

| # | Problem | File | Result |
|---|---|---|---|
| 1 | We have no data | `get_asteroids.py` | `asteroids.csv` (42,806 asteroids) |
| 2 | Some data is incomplete | `explore_data.py` | known (42,669) + unknown (137) |
| 3 | Can a tree learn the rule? | `train_model.py` | Tree learned `H <= 22.035` |
| 4 | Why 22.035 and not 22.0? | `experiment_labels.py` | Found 50 messy labels. Fixed them → `H <= 22.005` |
| 5 | Does it catch dangerous asteroids it has never seen? | `evaluate_model.py` | Caught 505 of 510 (recall 99%). 7 of 8 "mistakes" were messy labels |
| 6 | Near-cutoff asteroids get called safe | `close_calls.py` | Human-review zone down to 50 m: 0 missed, 860 reviews |

---

## The space words

| Word | Meaning |
|---|---|
| **PHA** | **P**otentially **H**azardous **A**steroid. NASA's label: `Y` (yes) or `N` (no) |
| **H** | Brightness, which tells us size. ⚠️ **Backwards:** smaller H = BIGGER asteroid |
| **MOID** | The closest the asteroid's path ever gets to Earth's path. Small MOID = close |
| **au** | Astronomical unit, the Earth-to-Sun distance (about 93 million miles) |
| **NEO / NEA** | Near-Earth Object / Near-Earth Asteroid |

### NASA's rule

```
If  H <= 22.0      ("H is 22 or less"     → big enough, 140 m or more)
AND moid <= 0.05   ("moid is 0.05 or less" → close enough)
then → PHA = Y (hazardous)
```

```
         Is H <= 22.0?   (big enough?)
          /          \
        YES           NO → N (safe)
         |
   Is moid <= 0.05?  (close enough?)
      /        \
    YES         NO → N (safe)
     |
   ⚠️ Y
```

---

## Comparison signs

| Sign | Read it as | Example |
|---|---|---|
| `<` | "is less than" | `3 < 5` ✅ |
| `>` | "is more than" | `5 > 3` ✅ |
| `<=` | "is ___ or less" | `22 <= 22` ✅ (the `=` means the number itself counts) |
| `>=` | "is ___ or more" | `7 >= 5` ✅ |
| `!=` | "is NOT equal to" | `"Y" != "N"` ✅ |

 **Trick:** the sign is a mouth that opens toward the **bigger** number.

**Two-step method for asteroids:**
1. Just check the number: "Is it 22 or less? Yes or no?"
2. Then translate: "Yes means it's big."

 **Decimals tip:** add zeros so they're the same length. **0.050** vs **0.149** → 149 is bigger than 50.

---

## AND (`&`): both must be True

Like driving: you need a license **AND** a car.

| | | Result |
|---|---|---|
| True | & True | **True** ✅ (the only way to get True!) |
| True | & False | **False** |
| False | & True | **False** |
| False | & False | **False** |

**Example: Eros** (H = 10.4, moid = 0.149)

```
(10.4 <= 22.0)  &  (0.149 <= 0.05)
     True       &       False        →  False  →  NOT hazardous (N)
```

---

## 📁 File 1: `get_asteroids.py`

**Problem:** To teach a computer, you need examples. We had none.
**Solution:** Ask NASA's database (an **API**) for them.

| Code | Why we needed it |
|---|---|
| `import json` | A tool to **read** NASA's answer (JSON text) |
| `import urllib.request` | A tool to **visit** a website |
| `import pandas as pd` | A tool to make **tables** (`as pd` = nickname) |
| `url = "...?fields=full_name,H,moid,pha..."` | Tell NASA **which columns** we want: H and moid (the rule's clues), pha (the answer), name (to tell asteroids apart) |
| `with urllib.request.urlopen(url) as response:` | **Open a connection** to NASA (and close it automatically) |
| `result = json.load(response)` | **Read** the reply into a dictionary |
| `data = pd.DataFrame(result["data"], columns=result["fields"])` | Turn the reply into a **table** |
| `data.to_csv("asteroids.csv", index=False)` | **Save** it, so we never have to download again |

---

## 📁 File 2: `explore_data.py`

**Problem:** 40,121 N + 2,550 Y ≠ 42,806. Some asteroids had **blanks**.
**Why it matters:** A model can't learn from an example with no answer, and blanks might be dangerous asteroids we don't know enough about yet.

| Code | Why we needed it |
|---|---|
| `data = pd.read_csv("asteroids.csv")` | **Load** the saved file (the opposite of `to_csv`) |
| `data.isna().sum()` | **Count** the blanks in each column |
| `has_blank = data.isna().any(axis=1)` | Mark **every row** that has a blank |
| `unknown = data[has_blank]` | Pile 1: rows **with** blanks |
| `known = data[~has_blank]` | Pile 2: rows **without** blanks (`~` = NOT) |
| `known.to_csv(...)` / `unknown.to_csv(...)` | Save both piles |

**Your decision (Option C):** *Unknown does NOT mean safe.* Set them aside; don't delete them and don't guess.

---

## 📁 File 3: `train_model.py`

**Problem:** The big question itself. Can a tree learn the rule?

| Code | Why we needed it |
|---|---|
| `from sklearn.tree import DecisionTreeClassifier, export_text` | Grab just the tree tools we need |
| `data = pd.read_csv("known_asteroids.csv")` | Use **only complete rows** (the Option C safeguard) |
| `X = data[["H", "moid"]]` | The **clues**. Same as NASA's rule. Name left out, since it could teach a **shortcut** |
| `y = data["pha"]` | The **answers** |
| `model = DecisionTreeClassifier(max_depth=2, random_state=0)` | An empty tree allowed **2 questions** (like NASA's rule). `random_state=0` = same result every time |
| `model.fit(X, y)` | **Training**: the tree studies 42,669 examples |
| `print(export_text(model, feature_names=["H", "moid"]))` | **Look inside**: what rule did it learn? |

**Result:** `H <= 22.035 & moid <= 0.05`. Almost NASA's rule, but not exactly. 🤔

---

## 📁 File 4: `experiment_labels.py`

**Problem:** Why 22.035 instead of 22.0?
**Hypothesis:** Some of NASA's labels are messy and pulled the line.

| Code | Why we needed it |
|---|---|
| `follows_rule = (data["H"] <= 22.0) & (data["moid"] <= 0.05)` | **Apply NASA's rule ourselves** to every asteroid |
| `rule_label = follows_rule.map({True: "Y", False: "N"})` | Turn True/False into Y/N so we can compare |
| `mismatches = data[rule_label != data["pha"]]` | Find where **NASA's label ≠ NASA's rule** |
| `y_clean = rule_label` | Use our **fixed labels** as the answers |
| Same `DecisionTreeClassifier(max_depth=2, random_state=0)` | **Change only one thing** (the labels), so we know what caused any change |

**Results:**
- Part A: found **50** messy labels, all with H between 21.70 and 22.57 (right on the border)
  - 17 labeled **N** that the rule says are **Y** ⚠️ (missed problems: the dangerous kind)
  - 33 labeled **Y** that the rule says are **N** (false alarms)
- Part B: the line moved to **22.005** ✅ **Hypothesis confirmed!**

---

## 📁 File 5: `evaluate_model.py`

**Problem:** Does the tree catch dangerous asteroids it has **never seen**? (NASA finds new ones every night.)

**The idea in 3 sentences:**
1. **Hide some asteroids** from the tree.
2. **The tree studies the rest.**
3. **Quiz it** on the hidden ones and count how many **dangerous** ones it caught.

Like a fair exam: a student who studied the exact exam questions could just memorize them. New questions show whether they really learned.

| Code | Why we needed it |
|---|---|
| `from sklearn.model_selection import train_test_split` | The tool that splits the data (from scikit-learn, not built into Python) |
| `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)` | Deal the asteroids into a 📚 **study pile** (80%, `_train`) and a 🙈 **hidden pile** (20%, `_test`). `stratify=y` keeps the same ~6% dangerous in both |
| `model.fit(X_train, y_train)` | Study **only** the study pile. This is what keeps the hidden pile hidden |
| `predictions = model.predict(X_test)` | Quiz: give it the hidden **clues only**. We keep `y_test` as the answer key |
| `caught = ((y_test == "Y") & (predictions == "Y")).sum()` | Count: really dangerous **AND** the tree said dangerous. (`==` asks "is it equal?"; `=` stores) |
| `recall = caught / (caught + missed)` | The smoke detector score |
| `(predictions == y_test).mean()` | Accuracy. `.mean()` of True/False = the share that are True |

**Results** (8,534 hidden asteroids: 510 dangerous, 8,024 safe):

```
                          TREE SAID
                     Y (danger)    N (safe)
                   ┌────────────┬────────────┐
  REALLY   Y  510  │ ✅ CAUGHT  │ ❌ MISSED  │
  IS               │    505     │     5      │
                   ├────────────┼────────────┤
           N 8,024 │ ⚠️ FALSE   │ ✅ CORRECT │
                   │  ALARM 3   │ SAFE 8,021 │
                   └────────────┴────────────┘
```

| | Accuracy | Recall (dangerous ones caught) |
|---|---|---|
| 💤 Lazy "always safe" | 94% | **0%** (0 of 510) |
| 🌳 Our tree | 99.9% | **99%** (505 of 510) |

**Recall** = caught ÷ (caught + missed) = 505 ÷ 510 = 0.990 → ×100 → **99%**

---

### Investigating the mistakes (Steps 9–10)

`missed_mask = (y_test == "Y") & (predictions == "N")` is the same as counting, but **without `.sum()`**, so you keep the True/False list (a **mask**, like a stencil) and can print those rows.

- **The tree doesn't know it missed.** *We* know, because we kept the answer key (`y_test`), like a teacher grading.
- All **5 missed** have H 22.02–22.57, so NASA's **rule** says N. The tree followed the rule. NASA's **labels** said Y.
- **2 of 3 false alarms** were the same: messy labels. Only **row 7762** (H = 22.01) was a real tree mistake, because the tree's line (22.035) had been pulled by messy labels.
- **A test score is only as good as its answer key.**

---

## 📁 File 6: `close_calls.py`

**Problem:** NASA's line is sharp. H = 22.01 is called safe, but it's almost the same size as H = 22.00. Our caution should be **chosen on purpose**, not an accident of messy labels.

**The idea:** if an asteroid is **close** (moid ≤ 0.05) **and** just under the size line, send it to a **human** (🔎 REVIEW). This is **human-in-the-loop**, another slice of Swiss cheese 🧀.

```
          big enough │  CLOSE CALL ZONE  │   clearly small
   ◄─────────────────┼═══════════════════┼──────────────────►
                   22.0              zone_top
                    → Y      → REVIEW           → N
```

| Code | Why |
|---|---|
| `zone_top = 24.3` | The edge, in **one variable**, so changing one number changes everything |
| `near_cutoff = (X_test["H"] > 22.0) & (X_test["H"] <= zone_top)` | "Past NASA's line, but not by more than zone_top" |
| `in_review = close & near_cutoff` | Close **AND** near the cutoff |
| `results["tree"] = predictions` | **Add a column** to a table |
| `results.loc[in_review, "final"] = "REVIEW"` | "In these rows, in this column, put REVIEW," which **overrides** the tree |

**What happened:**

| zone_top | How chosen | Still missed | REVIEW |
|---|---|---|---|
| 22.5 | First guess | 1 (H = 22.57) | 131 |
| 22.6 | ⚠️ After seeing the miss (tuning on the test set) | 0 | 160 |
| **24.3** | ✅ **For a reason:** 50 m (Tunguska size) could flatten a town | **0** | **860** |

**🎯 Drawing the target after you shoot = cheating**, even by accident. **Tuning on the test set** makes a score look better than it really is. Fixes:
1. **Decide first, for a reason**, write it down, then test **once**.
2. Use a third pile, a **validation set** (practice test), to try settings, and keep the final exam untouched.

**Size ↔ H** (NASA's formula, 14% reflection): 140 m = 22.0 · 100 m = 22.75 · 50 m = 24.3. H up by 5 = 10× smaller.

**The trade-off:** wider zone → fewer missed, more human work. And **what counts as "caught" depends on how you define danger.** NASA's labels only measure danger down to 140 m, so they can't score the 50–140 m asteroids our zone was built to catch.

---

## 🧪 The scientific method

| Step | What we did |
|---|---|
| 1. Notice something odd | "Why 22.035?" |
| 2. Make a **hypothesis** (a guess you can test) | "Messy labels pulled the line" |
| 3. Predict | "Clean labels → line moves to 22.0" |
| 4. Test (**change one thing**) | Line moved to 22.005 ✅ |

---

## 🐍 Python cheat sheet

| Code | Meaning |
|---|---|
| `# ...` | Comment. Python skips it |
| `import x` / `import x as y` | Load a tool / give it a nickname |
| `from x import y` | Grab one specific tool from a toolbox |
| `box = something` | **Variable.** The right side runs first, then it's stored on the left. Read `=` as "gets" |
| `tool(input)` | **Function.** Hand it something in `( )`, and it does a job |
| `thing.tool()` | **Method.** A tool that belongs to a thing |
| `name=value` inside `( )` | **Named argument.** An input with a label |
| `{"a": 1}` | **Dictionary.** Values stored under labels |
| `result["data"]` | Look something up by its label |
| `data[["H", "moid"]]` | Grab two columns (a **list** inside the brackets) |
| `data[True/False list]` | **Filtering.** Keep only the True rows |
| `True` / `False` | Yes / No (capital letters!) |
| `&` / `~` / `!=` | AND / NOT / not equal |
| `with ... :` + indent | Open something, use it, close it automatically |
| `==` | "Is it equal?" (a question). Different from `=`, which stores |
| `.mean()` | The average. On True/False, it gives the share that are True |

**3 kinds of tools:**

| Kind | Install? | Import? | Examples |
|---|---|---|---|
| Built-in | No | No | `print()`, `len()` |
| Standard library | No | Yes | `json`, `urllib` |
| Third-party | Yes (pip) | Yes | `pandas`, `sklearn` |

---

## 🛡️ AI safety lessons

1. **Unknown ≠ safe.** A decision tree *always* gives an answer, so **our code** must set incomplete data aside (a safeguard, one slice of Swiss cheese 🧀).
2. **The smoke detector problem.** Only ~6% of asteroids are hazardous. A lazy model that always says "safe" is 94% accurate but misses **every** dangerous asteroid. *It fails when it's needed most.* Always ask: "Does it catch the dangerous ones?"
3. **Models learn the labels, not what we meant.** Messy answers → messy model (**label noise**).
4. **Look inside the model** (**interpretability**). Reading the tree with `export_text` is how we caught the label problem.

---

## ✏️ Practice questions

Try them without peeking!

1. An asteroid has **H = 18** and **moid = 0.02**. Is it hazardous? Show both steps.
2. What does `True & False` give?
3. Is a bigger H a bigger or a smaller asteroid?
4. Why did we use `known_asteroids.csv` instead of `asteroids.csv` to train?
5. Why did we leave `full_name` out of `X`?
6. In the experiment, why did we keep the tree settings exactly the same?
7. A model is 94% accurate. Why might that still be bad?
8. Why do we hide some asteroids (the test set) before training?
9. What would go wrong with `model.fit(X, y)` instead of `model.fit(X_train, y_train)`?
10. Is `train_test_split()` built into Python?

<details>
<summary>👀 Click for answers</summary>

1. `18 <= 22.0` → True. `0.02 <= 0.05` → True. `True & True` → **True → hazardous (Y)**.
2. **False.** AND needs both to be True.
3. **Smaller.** H is backwards: bigger H = smaller asteroid.
4. So the model only learns from complete rows. The unknowns are set aside, never called safe.
5. A name can't make an asteroid dangerous, so it could teach the model a **shortcut**.
6. **Change only one thing.** Then if the line moves, we know the **labels** caused it.
7. If only 6% are dangerous, a model that always says "safe" gets 94% while catching **zero** dangerous asteroids.
8. To check whether the model works on asteroids it has **never seen**, like a fair exam with new questions.
9. The tree would study the hidden asteroids too, so the quiz would be like grading a student on questions they already saw. The score would look better than it really is.
10. **No.** It's from the **scikit-learn** library, so it needs pip install and `from sklearn.model_selection import train_test_split`.

</details>

---

## ⏭️ Next up

**Make the problem harder and more realistic.**
