# =====================================================================
# PROJECT 02: ASTEROID DANGER DETECTOR
# FILE: evaluate_model.py
# =====================================================================
#
# THE BIG QUESTION (for the whole project):
#   Can a machine learning model figure out which asteroids are
#   dangerous to Earth just by studying examples, without anyone
#   telling it the rule?
#
# THE QUESTION FOR THIS FILE:
#   The smoke detector question: out of all the DANGEROUS asteroids,
#   how many does our tree actually CATCH? And does it work on
#   asteroids it has NEVER SEEN before?
#
# WHY IT MATTERS:
#   A model is only useful if it works on NEW examples. In real life,
#   NASA discovers new asteroids every night. A model that only works
#   on asteroids it already studied would be useless.
#
# AI SAFETY CONNECTION:
#   Remember the smoke detector: a model can look great ("94% accurate!")
#   and still miss every dangerous case. So we'll measure the thing that
#   actually matters, and compare our tree to a lazy model that always
#   says "safe."
#
# ML WORDS IN THIS FILE:
#   training set = the examples the model STUDIES (like homework)
#   test set     = examples we HIDE from the model during training, then
#                  use to check it (like a final exam with new questions)
#   prediction   = the model's guess for an asteroid (Y or N)
#   caught       = dangerous asteroid, model said Y         ✅ good
#   missed       = dangerous asteroid, model said N         ❌ the scary mistake
#   false alarm  = safe asteroid, model said Y              ⚠️ annoying mistake
#   accuracy     = out of ALL asteroids, how many it got right
#   recall       = out of the DANGEROUS asteroids, how many it caught
#                  ("Does the smoke detector beep when there's a fire?")
#
# THE PLAN:
#   1. get_asteroids.py     <- done! downloaded the data
#   2. explore_data.py      <- done! set aside the 137 unknowns
#   3. train_model.py       <- done! tree learned H <= 22.035
#   4. experiment_labels.py <- done! found 50 messy labels
#   5. evaluate_model.py    <- THIS FILE: how many dangerous ones does it catch?
#   6. make it harder and more realistic  (later)
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

# 👉 YOUR TURN: from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# NEW TOOL: train_test_split splits our data into a training set
# (for studying) and a test set (for the exam).
#   sklearn.model_selection = the drawer in sklearn with tools for
#                             testing and choosing models
# 👉 YOUR TURN: from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------
# STEP 2: Load the data and pick clues (X) and answers (y)
# ---------------------------------------------------------------
# Same as train_model.py. We use NASA's real labels (pha), because
# that's the data a real project would have.
# 👉 YOUR TURN: data = pd.read_csv("known_asteroids.csv")
data = pd.read_csv("known_asteroids.csv")

# 👉 YOUR TURN: X = data[["H", "moid"]]
X = data[["H", "moid"]]

# 👉 YOUR TURN: y = data["pha"]
y = data["pha"]


# ---------------------------------------------------------------
# STEP 3: Split into a training set and a test set
# ---------------------------------------------------------------
# This is the BIG new idea. Picture it:
#
#              all 42,669 known asteroids
#    ┌───────────────────────────────────────────┐
#    │        80% TRAIN          │   20% TEST     │
#    │   the model studies these │ hidden until   │
#    │                           │ the exam       │
#    └───────────────────────────────────────────┘
#
# train_test_split hands back FOUR things at once, so we catch them
# in four variables, separated by commas:
#
#   X_train  = clues for studying       y_train = answers for studying
#   X_test   = clues for the exam       y_test  = answers for the exam
#                                       (we keep these to GRADE the exam)
#
# The order matters! It always comes back as:
#   X_train, X_test, y_train, y_test
#
# The settings inside the ( ):
#   X, y            = the data to split
#   test_size=0.2   = put 20% (0.2) in the test set, 80% in training
#   random_state=0  = the split is random, but this makes it the SAME
#                     random split every time (reproducible)
#   stratify=y      = keep the same mix of Y and N in both sets.
#                     Only ~6% are Y! Without this, the test set
#                     could end up with too few dangerous asteroids
#                     to fairly test the model.
#
# This line is long, so it's split across two lines inside the ( ),
# just like the url in get_asteroids.py.
# 👉 YOUR TURN: type these 2 lines (the second one is indented 4 spaces)
#   X_train, X_test, y_train, y_test = train_test_split(
#       X, y, test_size=0.2, random_state=0, stratify=y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y)


# ---------------------------------------------------------------
# STEP 4: Train the tree on the TRAINING set only
# ---------------------------------------------------------------
# Same settings as train_model.py. But notice: we fit on X_train and
# y_train, NOT on all the data. The test asteroids stay hidden.
# 👉 YOUR TURN: model = DecisionTreeClassifier(max_depth=2, random_state=0)
model = DecisionTreeClassifier(max_depth=2, random_state=0)

# 👉 YOUR TURN: model.fit(X_train, y_train)
model.fit(X_train, y_train)


# ---------------------------------------------------------------
# STEP 5: Give the model the exam
# ---------------------------------------------------------------
#   .predict( )  = "here are some clues; tell me your answer for each."
#                  We only give it X_test (the clues). It NEVER sees
#                  y_test (the answers). That would be cheating!
#
#   predictions  = the model's Y/N guess for every test asteroid
# 👉 YOUR TURN: predictions = model.predict(X_test)
predictions = model.predict(X_test)


# ---------------------------------------------------------------
# STEP 6: Grade the exam. Count each kind of result.
# ---------------------------------------------------------------
# We compare the real answers (y_test) to the model's guesses
# (predictions), using & (AND), which you've been practicing!
#
# Read the first one in words:
#   (y_test == "Y")        -> "Is it REALLY dangerous?"
#   (predictions == "Y")   -> "Did the model SAY dangerous?"
#   &                      -> both must be True
#   .sum()                 -> count the Trues (True counts as 1)
#
#   ==  means "is equal to" (a QUESTION, True or False).
#       Don't mix it up with = , which STORES something in a variable!
#           x = 5     -> "put 5 in the box x"
#           x == 5    -> "is x equal to 5?" -> True or False
#
#   Real answer   Model said   Name
#   ───────────   ──────────   ────
#       Y             Y        caught        ✅
#       Y             N        missed        ❌ (the scary one)
#       N             Y        false alarm   ⚠️
# 👉 YOUR TURN: caught = ((y_test == "Y") & (predictions == "Y")).sum()
caught = ((y_test == "Y") & (predictions == "Y")).sum()

# 👉 YOUR TURN: missed = ((y_test == "Y") & (predictions == "N")).sum()
missed = ((y_test == "Y") & (predictions == "N")).sum()

# 👉 YOUR TURN: false_alarms = ((y_test == "N") & (predictions == "Y")).sum()
false_alarms = ((y_test == "N") & (predictions == "Y")).sum()

# 👉 YOUR TURN: print("Caught:", caught)
print("Caught:", caught)

# 👉 YOUR TURN: print("Missed:", missed)
print("Missed:", missed)

# 👉 YOUR TURN: print("False alarms:", false_alarms)
print("False alarms:", false_alarms)


# ---------------------------------------------------------------
# STEP 7: Recall, the smoke detector score
# ---------------------------------------------------------------
# Recall answers: "Out of the dangerous asteroids, what share did
# we catch?"
#
#                     caught
#   recall  =  ─────────────────
#              caught  +  missed      <- that's ALL the dangerous ones
#
#   /   = divide
#   ( ) = do the adding FIRST, then divide (like in math class)
#
# The answer is between 0 and 1:
#   1.0  = caught every dangerous asteroid (100%)
#   0.5  = caught half (50%)
#   0.0  = caught none (0%), like the broken smoke detector
# 👉 YOUR TURN: recall = caught / (caught + missed)
recall = caught / (caught + missed)

# 👉 YOUR TURN: print("Recall:", recall)
print("Recall:", recall)


# ---------------------------------------------------------------
# STEP 8: Compare accuracy: our tree vs. the lazy "always safe" model
# ---------------------------------------------------------------
#   predictions == y_test  -> True where the model was right
#   .mean()                -> the AVERAGE of the Trues and Falses.
#                             Since True = 1 and False = 0, the average
#                             is the SHARE that are True.
#                             Example: True, True, False, True
#                                      = (1+1+0+1) / 4 = 0.75 = 75%
# 👉 YOUR TURN: accuracy = (predictions == y_test).mean()
accuracy = (predictions == y_test).mean()

# 👉 YOUR TURN: print("Tree accuracy:", accuracy)
print("Tree accuracy:", accuracy)


# The lazy model says "N" for every asteroid. So it's right exactly
# when the real answer is N:
#   (y_test == "N").mean()  -> the share of test asteroids that are N
# 👉 YOUR TURN: lazy_accuracy = (y_test == "N").mean()
lazy_accuracy = (y_test == "N").mean()

# 👉 YOUR TURN: print("Lazy model accuracy:", lazy_accuracy)
print("Lazy model accuracy:", lazy_accuracy)

# The lazy model's recall is 0: it never says Y, so it catches
# nothing. Compare that to our tree's recall above!
#The tree caught 505 out of 510 dangerous asteroids it had never seen, which is 99% recall, while a lazy “always safe” model catches 0.


# =====================================================================
# PART 2: INVESTIGATE THE MISTAKES
# =====================================================================
# Our tree missed 5 dangerous asteroids and gave 3 false alarms.
# Which ones? And WHY? Looking closely at a model's mistakes is one
# of the best ways to understand it.
#
# Remember NASA's rule:   H <= 22.0   &   moid <= 0.05   ->   Y


# ---------------------------------------------------------------
# STEP 9: Show the 5 MISSED asteroids
# ---------------------------------------------------------------
# This is the same question as "missed" in Step 6, but WITHOUT .sum().
# Without .sum(), we keep the True/False list instead of counting it.
#
#   missed_mask = one True/False per test asteroid:
#                 True = "really Y, but the tree said N"
#   (a "mask" is ML slang for a True/False list used to pick rows,
#    like a stencil that only lets some rows show through)
# 👉 YOUR TURN: missed_mask = (y_test == "Y") & (predictions == "N")
missed_mask = (y_test == "Y") & (predictions == "N")

# FILTERING (you've done this before!): keep only the True rows of
# the test clues, so we can see their H and moid.
# The numbers on the left are each asteroid's row number.
# 👉 YOUR TURN: print("MISSED (really Y, tree said N):")
print("MISSED (really Y, tree said N):")

# 👉 YOUR TURN: print(X_test[missed_mask])
print(X_test[missed_mask])


# ---------------------------------------------------------------
# STEP 10: Show the 3 FALSE ALARMS
# ---------------------------------------------------------------
# Same idea, flipped: really N, but the tree said Y.
# 👉 YOUR TURN: false_alarm_mask = (y_test == "N") & (predictions == "Y")
false_alarm_mask = (y_test == "N") & (predictions == "Y")

# 👉 YOUR TURN: print("FALSE ALARMS (really N, tree said Y):")
print("FALSE ALARMS (really N, tree said Y):")

# 👉 YOUR TURN: print(X_test[false_alarm_mask])
print(X_test[false_alarm_mask])

# When you run this, check each asteroid BY HAND with NASA's rule:
#   1. Is H <= 22.0?
#   2. Is moid <= 0.05?
#   3. Both True -> the rule says Y. Otherwise N.
# Then compare: who was "wrong," the tree or the label? 🤔
