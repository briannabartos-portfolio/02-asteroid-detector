# =====================================================================
# PROJECT 02: ASTEROID DANGER DETECTOR
# FILE: close_calls.py
# =====================================================================
#
# THE BIG QUESTION (for the whole project):
#   Can a machine learning model figure out which asteroids are
#   dangerous to Earth just by studying examples, without anyone
#   telling it the rule?
#
# THE QUESTION FOR THIS FILE:
#   NASA's size line (H <= 22.0) is sharp: H = 21.99 counts as big
#   enough, H = 22.01 doesn't, even though they're almost the same
#   size. Can we add a DELIBERATE safety margin, so asteroids that are
#   close to Earth's path AND just under the size cutoff get sent to a
#   human for review instead of being called safe?
#
# WHY IT MATTERS:
#   In evaluate_model.py, the tree "missed" 5 asteroids that were very
#   close to Earth's path and only a hair too small. They're still
#   dangerous (roughly 110-140 meters). And the tree's extra caution
#   on row 7762 happened by ACCIDENT (messy labels moved its line).
#   We want caution that's CHOSEN on purpose and written down.
#
# AI SAFETY CONNECTION:
#   HUMAN-IN-THE-LOOP: when an AI isn't sure, a person checks.
#   This is another slice of Swiss cheese 🧀: the tree is one layer,
#   and our close-call rule is another layer that catches what slips
#   through. Just like Option C in explore_data.py, unknown or
#   borderline does NOT mean safe.
#
# ML WORDS IN THIS FILE:
#   safety margin = extra room you add on purpose to be careful
#   close call    = an asteroid near the edge of NASA's rule
#   human review  = sending a case to a person instead of letting
#                   the model decide alone
#   trade-off     = getting more of one good thing costs you some of
#                   another. Here: a wider zone catches more dangerous
#                   asteroids, but gives humans more work.
#
# THE PLAN:
#   1. get_asteroids.py     <- done! downloaded the data
#   2. explore_data.py      <- done! set aside the 137 unknowns
#   3. train_model.py       <- done! tree learned H <= 22.035
#   4. experiment_labels.py <- done! found 50 messy labels
#   5. evaluate_model.py    <- done! caught 505 of 510 on the exam
#   6. close_calls.py       <- THIS FILE: add a human-review safety zone
#
# HOW TO READ THIS FILE:
#   Lines that start with "#" are COMMENTS. Python skips them.
#   Wherever you see "👉 YOUR TURN", type the code line right below it.
# =====================================================================


# ---------------------------------------------------------------
# STEP 1: Bring in the tools (same as evaluate_model.py)
# ---------------------------------------------------------------
# 👉 YOUR TURN: import pandas as pd
import pandas as pd

# 👉 YOUR TURN: from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# 👉 YOUR TURN: from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------
# STEP 2: Load the data, clues (X) and answers (y) (same as before)
# ---------------------------------------------------------------
# 👉 YOUR TURN: data = pd.read_csv("known_asteroids.csv")
data = pd.read_csv("known_asteroids.csv")

# 👉 YOUR TURN: X = data[["H", "moid"]]
X = data[["H", "moid"]]

# 👉 YOUR TURN: y = data["pha"]
y = data["pha"]


# ---------------------------------------------------------------
# STEP 3: Split into study pile and hidden pile (same as before)
# ---------------------------------------------------------------
# Because random_state=0 is the same, we get the EXACT SAME piles as
# evaluate_model.py. That's reproducibility paying off: we can
# compare results fairly.
# 👉 YOUR TURN: type these 2 lines (the second one is indented 4 spaces)
#   X_train, X_test, y_train, y_test = train_test_split(
#       X, y, test_size=0.2, random_state=0, stratify=y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y)

# ---------------------------------------------------------------
# STEP 4: Train the tree and take the exam (same as before)
# ---------------------------------------------------------------
# 👉 YOUR TURN: model = DecisionTreeClassifier(max_depth=2, random_state=0)
model = DecisionTreeClassifier(max_depth=2, random_state=0)

# 👉 YOUR TURN: model.fit(X_train, y_train)
model.fit(X_train, y_train)

# 👉 YOUR TURN: predictions = model.predict(X_test)
predictions = model.predict(X_test)


# =====================================================================
# THE NEW PART: THE CLOSE CALL ZONE
# =====================================================================
#
#   H:   ... 21.5 ── 22.0 ══════════ zone_top ── 23.0 ...
#                     │   CLOSE CALL    │
#        big enough   │      ZONE       │   clearly too small
#                     │  "just under    │
#                     │  the size line" │
#
#   An asteroid goes to HUMAN REVIEW if:
#     its path comes close to Earth's   (moid <= 0.05)
#     AND it's just under the size line (H is more than 22.0
#                                        AND H is zone_top or less)


# ---------------------------------------------------------------
# STEP 5: Choose where the zone ends. This is YOUR decision!
# ---------------------------------------------------------------
# We store the edge in a variable, so it's easy to change later:
# change ONE number here, and the whole file uses the new edge.
#
# HISTORY (being honest about how we got here):
#   Try 1: 22.5, a first guess. Result: 1 still missed (H = 22.57).
#   Try 2: 22.6. Result: 0 missed, BUT we chose 22.6 after seeing the
#          miss, so that was tuning on the test set (drawing the
#          target after shooting).
#
# DECISION (made for a reason, before looking at the result):
#   zone_top = 24.3
#   REASON: An asteroid about 50 meters wide (about 2 blue whales)
#   could flatten a forest or a town, like the 1908 Tunguska event.
#   If one passes close to Earth's path, a human should check it.
#   Using NASA's size formula (assuming the surface reflects 14% of
#   sunlight), 50 meters is about H = 24.3.
#
#   Known limit: we had already seen this test pile, so this result
#   isn't perfectly fresh. Ideally it should be checked on brand-new
#   asteroids.
# 👉 YOUR TURN: replace the line below with  zone_top = 24.3
zone_top = 24.3


# ---------------------------------------------------------------
# STEP 6: Find the asteroids in the close call zone
# ---------------------------------------------------------------
# Each line gives one True/False per test asteroid (a MASK, like
# in evaluate_model.py).
#
# Read them out in words:
#   X_test["moid"] <= 0.05   -> "Is moid 0.05 or less?"   (close?)
#   X_test["H"] > 22.0       -> "Is H MORE than 22.0?"    (past the line?)
#   X_test["H"] <= zone_top  -> "Is H zone_top or less?"  (but not by much?)
#
# Why ">" and not ">=" for 22.0? An asteroid with H = 22.0 exactly
# already counts as big enough under NASA's rule. The zone starts
# just AFTER it.
# 👉 YOUR TURN: close = X_test["moid"] <= 0.05
close = X_test["moid"] <= 0.05

# Both size questions must be True to be "just under the line,"
# so we join them with & (AND). Each one needs its own ( ).
# 👉 YOUR TURN: near_cutoff = (X_test["H"] > 22.0) & (X_test["H"] <= zone_top)
near_cutoff = (X_test["H"] > 22.0) & (X_test["H"] <= zone_top)


# Close AND near the cutoff -> send to human review.
# 👉 YOUR TURN: in_review = close & near_cutoff
in_review = close & near_cutoff


# ---------------------------------------------------------------
# STEP 7: Build a results table with the final decisions
# ---------------------------------------------------------------
# .copy() makes a separate copy of the test clues, so changes we make
# to "results" don't mess up X_test.
# 👉 YOUR TURN: results = X_test.copy()
results = X_test.copy()

# NEW IDEA: adding a column. results["new name"] = values
# creates a new column with that name (or replaces it if it exists).
#   "real" = NASA's label (the answer key)
#   "tree" = what the tree guessed
# 👉 YOUR TURN: results["real"] = y_test
results["real"] = y_test

# 👉 YOUR TURN: results["tree"] = predictions
results["tree"] = predictions

# "final" starts as a copy of the tree's answer...
# 👉 YOUR TURN: results["final"] = predictions
results["final"] = predictions

# ...then our safeguard OVERRIDES it for the close calls.
#   .loc[ rows , column ]  = "in these rows, in this column"
#   in_review              = the rows (our mask: only the True rows)
#   "final"                = the column
#   = "REVIEW"             = put the word REVIEW there
# So: "for every close call, change the final decision to REVIEW."
# 👉 YOUR TURN: results.loc[in_review, "final"] = "REVIEW"
results.loc[in_review, "final"] = "REVIEW"


# ---------------------------------------------------------------
# STEP 8: How did our safeguard do?
# ---------------------------------------------------------------
# Count the final decisions: how many Y, N, and REVIEW?
# (value_counts, like in get_asteroids.py)
# 👉 YOUR TURN: print(results["final"].value_counts())
print(results["final"].value_counts())

# The scary number: dangerous asteroids (by NASA's label) that STILL
# end up called safe (N). Before the safeguard, this was 5.
# 👉 YOUR TURN: still_missed = ((results["real"] == "Y") & (results["final"] == "N")).sum()
still_missed = ((results["real"] == "Y") & (results["final"] == "N")).sum()

# 👉 YOUR TURN: print("Still missed:", still_missed)
print("Still missed:", still_missed)

# Let's see the dangerous-labeled asteroids that our zone sent to review.
# Two masks joined with &: in the zone AND labeled Y.
# 👉 YOUR TURN: print(results[in_review & (results["real"] == "Y")])
print(results[in_review & (results["real"] == "Y")])