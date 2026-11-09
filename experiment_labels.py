# =====================================================================
# PROJECT 02: ASTEROID DANGER DETECTOR
# FILE: experiment_labels.py
# =====================================================================
#
# THE BIG QUESTION (for the whole project):
#   Can a machine learning model figure out which asteroids are
#   dangerous to Earth just by studying examples, without anyone
#   telling it the rule?
#
# THE QUESTION FOR THIS FILE (our HYPOTHESIS):
#   In train_model.py the tree drew its size line at 22.035 instead of
#   NASA's 22.0. We think that's because about 50 of NASA's labels
#   don't follow NASA's own rule, and they "pulled" the line.
#
#   If that's true, then:
#     PART A: we should be able to FIND those mismatched asteroids.
#     PART B: if we fix the labels and retrain, the line should move
#             back to (almost) 22.0.  <- your prediction!
#
# WHY IT MATTERS:
#   A model learns what the data SAYS, not what we MEANT. Checking
#   whether your labels are trustworthy is a core ML skill.
#
# AI SAFETY CONNECTION:
#   LABEL NOISE: when some of the "right answers" in training data are
#   wrong or inconsistent. A model trained on noisy labels quietly
#   learns the mistakes too. Testing a hypothesis like this is how
#   researchers catch it.
#
# ML WORDS IN THIS FILE:
#   hypothesis    = a guess you can test
#   label noise   = wrong or inconsistent answers in the training data
#   clean labels  = labels we made ourselves by applying NASA's rule
#                   exactly, so they're perfectly consistent
#   experiment    = change ONE thing, keep everything else the same,
#                   and see what happens. Here we change only the
#                   labels; the features (H, moid) and the model
#                   settings stay exactly like train_model.py.
#
# THE PLAN:
#   1. get_asteroids.py     <- done! downloaded the data
#   2. explore_data.py      <- done! set aside the 137 unknowns
#   3. train_model.py       <- done! tree learned H <= 22.035
#   4. experiment_labels.py <- THIS FILE: test why it's 22.035
#   5. check how well it catches dangerous asteroids  (later)
#
# HOW TO READ THIS FILE:
#   Lines that start with "#" are COMMENTS. Python skips them.
#   Wherever you see "👉 YOUR TURN", type the code line right below it.
# =====================================================================


# ---------------------------------------------------------------
# STEP 1: Bring in the tools (same as train_model.py)
# ---------------------------------------------------------------
# 👉 YOUR TURN: import pandas as pd
import pandas as pd

# 👉 YOUR TURN: from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.tree import DecisionTreeClassifier, export_text


# ---------------------------------------------------------------
# STEP 2: Load the known asteroids (same as train_model.py)
# ---------------------------------------------------------------
# 👉 YOUR TURN: data = pd.read_csv("known_asteroids.csv")
data = pd.read_csv("known_asteroids.csv")


# =====================================================================
# PART A: FIND THE ASTEROIDS WHOSE LABEL BREAKS NASA'S RULE
# =====================================================================

# ---------------------------------------------------------------
# STEP 3: Apply NASA's rule ourselves, to every asteroid at once
# ---------------------------------------------------------------
# Read it out in words:
#   data["H"] <= 22.0      -> "Is H 22.0 or less?"      (big enough?)
#   data["moid"] <= 0.05   -> "Is moid 0.05 or less?"   (close enough?)
#
# When you compare a whole COLUMN to a number, pandas asks the
# question for EVERY row and gives back one True/False per asteroid:
#
#     H       H <= 22.0
#     10.4 -> True
#     25.8 -> False
#
#   &   = AND. "Both must be True." (NASA's rule needs both!)
#           True  & True  -> True
#           True  & False -> False
#           False & False -> False
#
#   ( ) = each question MUST be wrapped in parentheses when you use &.
#         Without them, Python gets the order wrong and gives an error.
#
#   follows_rule = one True/False per asteroid:
#                  "Does NASA's rule say this one is hazardous?"
# 👉 YOUR TURN: follows_rule = (data["H"] <= 22.0) & (data["moid"] <= 0.05)
follows_rule = (data["H"] <= 22.0) & (data["moid"] <= 0.05)


# ---------------------------------------------------------------
# STEP 4: Turn True/False into "Y"/"N", like NASA's labels
# ---------------------------------------------------------------
# NASA's pha column uses "Y" and "N", but follows_rule uses True and
# False. To compare them, they need to speak the same language.
#
#   .map( )            = "swap each value for a new one, using this
#                         lookup table"
#   {True: "Y", False: "N"}
#                      = a DICTIONARY (you met these in get_asteroids.py!)
#                         used as the lookup table:
#                           True  -> "Y"
#                           False -> "N"
#
#   rule_label = what NASA's label WOULD be if it followed the rule perfectly.
# 👉 YOUR TURN: rule_label = follows_rule.map({True: "Y", False: "N"})
rule_label = follows_rule.map({True: "Y", False: "N"})


# ---------------------------------------------------------------
# STEP 5: Find the asteroids where NASA's label and the rule disagree
# ---------------------------------------------------------------
#   !=   = "is NOT equal to"   (the ! means NOT)
#            "Y" != "N"  -> True   (they're different)
#            "Y" != "Y"  -> False  (they're the same)
#
#   rule_label != data["pha"]
#        -> True for every asteroid where the rule and NASA's label
#           DISAGREE
#
#   data[ ... ]  -> FILTERING again (like explore_data.py): keep only
#                   the rows that are True
# 👉 YOUR TURN: mismatches = data[rule_label != data["pha"]]
mismatches = data[rule_label != data["pha"]]

# How many did we find? Our hypothesis says about 50.
# 👉 YOUR TURN: print("Mismatched labels:", len(mismatches))
print("Mismatched labels:", len(mismatches))

# Let's SEE them, sorted from smallest H to biggest H, so we can
# check if they're all near the 22.0 border.
#   .sort_values("H") = put the rows in order by the H column
# 👉 YOUR TURN: print(mismatches.sort_values("H"))
print(mismatches.sort_values("H"))


# =====================================================================
# PART B: RETRAIN ON CLEAN LABELS. DOES THE LINE MOVE TO 22.0?
# =====================================================================

# ---------------------------------------------------------------
# STEP 6: Same clues, NEW answers
# ---------------------------------------------------------------
# X (the clues) is exactly the same as in train_model.py.
# 👉 YOUR TURN: X = data[["H", "moid"]]
X = data[["H", "moid"]]

# But for the answers, we use OUR clean labels instead of NASA's pha.
# This is the ONE thing we're changing in our experiment.
# 👉 YOUR TURN: y_clean = rule_label
y_clean = rule_label


# ---------------------------------------------------------------
# STEP 7: Train a tree with the SAME settings as train_model.py
# ---------------------------------------------------------------
# Same settings on purpose! If we changed the settings too, we
# wouldn't know whether the labels or the settings moved the line.
# 👉 YOUR TURN: model = DecisionTreeClassifier(max_depth=2, random_state=0)
model = DecisionTreeClassifier(max_depth=2, random_state=0)

# 👉 YOUR TURN: model.fit(X, y_clean)
model.fit(X, y_clean)


# ---------------------------------------------------------------
# STEP 8: Read the new rules. Where's the line now?
# ---------------------------------------------------------------
#   decimals=3 = show 3 numbers after the decimal point.
#                export_text normally shows only 2, which rounded our
#                last answer from 22.035 to "22.04". We want to see
#                the exact line this time.
#
# Compare to train_model.py:   H <= 22.035   (NASA's noisy labels)
# NASA's real rule:            H <= 22.0
# 👉 YOUR TURN: print(export_text(model, feature_names=["H", "moid"], decimals=3))
print(export_text(model, feature_names=["H", "moid"], decimals=3))