# =====================================================================
# PROJECT 02: ASTEROID DANGER DETECTOR, LEVEL 2 (HARDER!)
# FILE: train_orbits.py
# =====================================================================
#
# THE BIG QUESTION (for the whole project):
#   Can a machine learning model figure out which asteroids are
#   dangerous to Earth just by studying examples, without anyone
#   telling it the rule?
#
# THE QUESTION FOR THIS FILE:
#   If we take MOID away and give the tree only H (size) plus the
#   orbit's SHAPE (q, ad, a, e, i), how many dangerous asteroids can
#   it still catch?
#
#   Level 1 result to compare against (WITH moid):
#       caught 505 of 510   ->   recall 0.99
#
# WHY IT MATTERS:
#   MOID needs the orbit's shape AND which way it points. We're only
#   giving the shape. This tests whether the tree can find danger
#   from incomplete, raw clues, which is what real problems look like.
#
# AI SAFETY CONNECTION:
#   Watch the SIZE of the tree. A bigger tree may catch more, but it
#   gets harder for humans to read and check. That's the trade-off
#   between performance and INTERPRETABILITY.
#
# ML WORDS IN THIS FILE:
#   depth  = how many yes/no questions the tree can ask in a row
#            (max_depth=2 means at most 2 questions)
#   leaf   = an end point of the tree, where it gives an answer
#            ("class: Y" or "class: N"). More leaves = a bigger tree.
#   None   = Python's word for "nothing" or "no limit." max_depth=None
#            means "ask as many questions as you want."
#
# THE PLAN FOR LEVEL 2:
#   1. get_orbits.py    <- done! downloaded size + orbit shape
#   2. train_orbits.py  <- THIS FILE: train a tree WITHOUT moid
#
# HOW TO READ THIS FILE:
#   Lines that start with "#" are COMMENTS. Python skips them.
#   Wherever you see "👉 YOUR TURN", type the code line right below it.
# =====================================================================


# ---------------------------------------------------------------
# STEP 1: Bring in the tools
# ---------------------------------------------------------------
# 👉 YOUR TURN: import pandas as pd
import pandas as pd

# 👉 YOUR TURN: from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.tree import DecisionTreeClassifier, export_text

# 👉 YOUR TURN: from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------
# STEP 2: Load the data and set aside rows with blanks
# ---------------------------------------------------------------
# 👉 YOUR TURN: data = pd.read_csv("orbits.csv")
data = pd.read_csv("orbits.csv")

# Same Option C idea as explore_data.py: only learn from complete rows.
#   .dropna() = "drop rows that have any NA (blank)." It's a shortcut
#               for what we did by hand before:  data[~has_blank]
# (We already saved the unknowns in Level 1, so we don't need a
# separate unknown file this time.)
# 👉 YOUR TURN: known = data.dropna()
known = data.dropna()


# ---------------------------------------------------------------
# STEP 3: Pick the clues (X) and the answer (y)
# ---------------------------------------------------------------
# A LIST of column names, stored in a variable so we can reuse it.
# Notice what's MISSING: moid! (and full_name, which could teach
# a shortcut, like in Level 1)
# 👉 YOUR TURN: features = ["H", "q", "ad", "a", "e", "i"]
features = ["H", "q", "ad", "a", "e", "i"]

# We pass the list inside the brackets to grab all 6 columns.
# (This is the same as  known[["H", "q", "ad", "a", "e", "i"]],
# just tidier.)
# 👉 YOUR TURN: X = known[features]
X = known[features]

# 👉 YOUR TURN: y = known["pha"]
y = known["pha"]


# ---------------------------------------------------------------
# STEP 4: Study pile and hidden pile (same as evaluate_model.py)
# ---------------------------------------------------------------
# 👉 YOUR TURN: type these 2 lines (the second one indented 4 spaces)
#   X_train, X_test, y_train, y_test = train_test_split(
#       X, y, test_size=0.2, random_state=0, stratify=y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y)

# ---------------------------------------------------------------
# STEP 5: Choose how deep the tree can go, then train and test it
# ---------------------------------------------------------------
# Like zone_top in close_calls.py, we keep this setting in ONE
# variable, so you can change one number and rerun.
#
# Start with 2, the same as Level 1. You'll try other depths after.
#
# Honest note: trying several depths and comparing them on the hidden
# pile is EXPLORING, not a final score. (Remember "drawing the target
# after you shoot"?) To report a final, fair score, we'd choose the
# depth with a practice pile (a validation set) first.
# 👉 YOUR TURN: tree_depth = 2
tree_depth = None

# 👉 YOUR TURN: model = DecisionTreeClassifier(max_depth=tree_depth, random_state=0)
model = DecisionTreeClassifier(max_depth=tree_depth, random_state=0)

# 👉 YOUR TURN: model.fit(X_train, y_train)
model.fit(X_train, y_train)

# 👉 YOUR TURN: predictions = model.predict(X_test)
predictions = model.predict(X_test)


# ---------------------------------------------------------------
# STEP 6: Grade the exam (same as evaluate_model.py)
# ---------------------------------------------------------------
# 👉 YOUR TURN: caught = ((y_test == "Y") & (predictions == "Y")).sum()
caught = ((y_test == "Y") & (predictions == "Y")).sum()

# 👉 YOUR TURN: missed = ((y_test == "Y") & (predictions == "N")).sum()
missed = ((y_test == "Y") & (predictions == "N")).sum()

# 👉 YOUR TURN: false_alarms = ((y_test == "N") & (predictions == "Y")).sum()
false_alarms = ((y_test == "N") & (predictions == "Y")).sum()

# 👉 YOUR TURN: recall = caught / (caught + missed)
recall = caught / (caught + missed)

# 👉 YOUR TURN: print("Depth:", tree_depth)
print("Depth:", tree_depth)

# 👉 YOUR TURN: print("Caught:", caught, " Missed:", missed, " False alarms:", false_alarms)
print("Caught:", caught, " Missed:", missed, " False alarms:", false_alarms)

# 👉 YOUR TURN: print("Recall:", recall)
print("Recall:", recall)


# ---------------------------------------------------------------
# STEP 7: How BIG is the tree?
# ---------------------------------------------------------------
#   .get_n_leaves() = count the tree's end points (leaves), where it
#                     gives a final Y or N. More leaves = more rules
#                     for a human to read and check.
#   (n is short for "number of")
# 👉 YOUR TURN: print("Leaves:", model.get_n_leaves())
print("Leaves:", model.get_n_leaves())


# ---------------------------------------------------------------
# STEP 8: Look inside, but only if the tree is small enough to read
# ---------------------------------------------------------------
# An IF statement, like the safeguards from the fruit lessons:
#   if <question>:      -> "only do the indented part IF this is True"
#
#   model.get_depth()   -> how deep the trained tree actually got
#   <= 4                -> "is it 4 or less?"
#
# A deep tree prints thousands of lines, which no human can read.
# So we only print it when it's small. That itself shows the
# interpretability problem!
# 👉 YOUR TURN: type these 2 lines (the second one indented 4 spaces)
#   if model.get_depth() <= 4:
#       print(export_text(model, feature_names=features))
if model.get_depth() <= 4:
    print(export_text(model, feature_names=features))