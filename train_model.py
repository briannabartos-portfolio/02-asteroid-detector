# =====================================================================
# PROJECT 02: ASTEROID DANGER DETECTOR
# FILE: train_model.py
# =====================================================================
#
# THE BIG QUESTION (for the whole project):
#   Can a machine learning model figure out which asteroids are
#   dangerous to Earth just by studying examples, without anyone
#   telling it the rule?
#
# THE QUESTION FOR THIS FILE:
#   If we show a decision tree 42,669 real asteroids (their size, their
#   closeness, and NASA's answer), will it come up with NASA's rule
#   by itself?
#
#   NASA's rule (the tree never sees this!):
#       H <= 22.0   AND   MOID <= 0.05   ->   hazardous (Y)
#
# WHY IT MATTERS:
#   If the tree finds the same rule, we know it learned the REAL
#   pattern. If it finds something else, it may have learned a
#   SHORTCUT, which could fail on new asteroids.
#
# AI SAFETY CONNECTION:
#   INTERPRETABILITY: being able to look inside a model and understand
#   WHY it makes its decisions. Decision trees are one of the few kinds
#   of models we can read like a flowchart. Most modern AI (like the
#   models behind chatbots) can't be read this easily, which is a big
#   open problem in AI safety.
#
# ML WORDS IN THIS FILE:
#   scikit-learn  = a free Python library full of ML models.
#                   You used it in the fruit lessons. Its code name is
#                   "sklearn".
#   X             = the FEATURES (the clues): H and moid.
#                   ML people use a capital X by tradition.
#   y             = the LABEL (the answer): pha.
#                   Lowercase y, also by tradition.
#   training      = showing the model examples so it learns a pattern.
#                   In code, this is called "fitting" the model.
#   export_text   = prints the tree's learned rules so humans can read them.
#
# THE PLAN:
#   1. get_asteroids.py   <- done! downloaded the data
#   2. explore_data.py    <- done! set aside the 137 unknowns
#   3. train_model.py     <- THIS FILE: train the tree and read its rules
#   4. check how well it catches dangerous asteroids  (later)
#   5. make it harder and more realistic  (later)
#
# HOW TO READ THIS FILE:
#   Lines that start with "#" are COMMENTS. Python skips them.
#   Wherever you see "👉 YOUR TURN", type the code line right below it.
# =====================================================================


# ---------------------------------------------------------------
# STEP 1: Bring in the tools
# ---------------------------------------------------------------
# pandas, for reading our table (same as before).
# 👉 YOUR TURN: import pandas as pd
import pandas as pd

# This is a new way to import. Instead of loading a whole toolbox,
# "from ... import ..." grabs just the specific tools we want:
#
#   from sklearn.tree        = "go into the 'tree' drawer of sklearn..."
#   import                   = "...and take out..."
#   DecisionTreeClassifier   = the decision tree model
#                              (a CLASSIFIER sorts things into groups,
#                              here Y or N)
#   ,                        = "and also..."
#   export_text              = the tool that prints the tree's rules
#
# Why bother? Now we can type DecisionTreeClassifier instead of
# sklearn.tree.DecisionTreeClassifier every time.
# 👉 YOUR TURN: from sklearn.tree import DecisionTreeClassifier, export_text

from sklearn.tree import DecisionTreeClassifier, export_text

# ---------------------------------------------------------------
# STEP 2: Load the KNOWN asteroids
# ---------------------------------------------------------------
# We use known_asteroids.csv, NOT asteroids.csv. That's our Option C
# safeguard: the model only learns from complete rows. The 137
# unknowns stay set aside in their own file.
# 👉 YOUR TURN: data = pd.read_csv("known_asteroids.csv")
data = pd.read_csv("known_asteroids.csv")


# ---------------------------------------------------------------
# STEP 3: Split the table into clues (X) and answers (y)
# ---------------------------------------------------------------
# The model needs these handed over separately:
# "here are the clues" and "here are the right answers."
#
#   data[["H", "moid"]]  -> grabs TWO columns at once.
#       Notice the DOUBLE square brackets:
#       the outer [ ]           = "look something up in the table"
#       the inner [ "H", "moid" ] = a LIST of column names
#       (a list = several things inside [ ], separated by commas)
#
#   data["pha"]          -> grabs ONE column, so single brackets.
#
# We leave out full_name on purpose. An asteroid's NAME doesn't make
# it dangerous, so letting the model use it could teach it a
# SHORTCUT (remember shortcut learning from the fruit lessons?).
#
#    data                    X (clues)       y (answer)
#    full_name H  moid pha   H     moid      pha
#    Eros     10.4 0.149 N   10.4  0.149     N
#    ...                     ...             ...
# 👉 YOUR TURN: X = data[["H", "moid"]]
X = data[["H", "moid"]]

# 👉 YOUR TURN: y = data["pha"]

y = data["pha"]

# ---------------------------------------------------------------
# STEP 4: Create an empty, untrained decision tree
# ---------------------------------------------------------------
# This makes the model, but it hasn't learned anything yet.
# Think of it as a brand-new student with an empty notebook.
#
#   model           = a variable to hold our model
#   max_depth=2     = "you may ask at most 2 yes/no questions in a row."
#                     NASA's rule only needs 2 questions (size, then
#                     closeness), so let's see if the tree can find
#                     them. A small tree is also easier for us to READ.
#   random_state=0  = makes the results exactly the same every time we
#                     run it. Some parts of training involve chance;
#                     fixing this number makes it REPRODUCIBLE.
# 👉 YOUR TURN: model = DecisionTreeClassifier(max_depth=2, random_state=0)
model = DecisionTreeClassifier(max_depth=2, random_state=0)


# ---------------------------------------------------------------
# STEP 5: Train the model!
# ---------------------------------------------------------------
# .fit() is where the learning happens. The model studies all
# 42,669 examples: the clues (X) and the right answers (y).
# It searches for the yes/no questions that best separate the
# Y asteroids from the N asteroids.
#
#   fit = "fit the model to the data," meaning adjust it until it
#         matches the examples as well as it can.
# 👉 YOUR TURN: model.fit(X, y)
model.fit(X, y)


# ---------------------------------------------------------------
# STEP 6: Look inside! What rules did it learn?
# ---------------------------------------------------------------
# export_text turns the trained tree into readable text, like
# a flowchart. You used this in the fruit lessons.
#
#   feature_names=["H", "moid"]  -> use these names in the printout
#   (without it, the printout would say "feature_0" and "feature_1",
#   which is hard to read)
#
# Then compare what it prints to NASA's rule at the top of this file.
# 👉 YOUR TURN: print(export_text(model, feature_names=["H", "moid"]))
print(export_text(model, feature_names=["H", "moid"]))

#A model learns what the data says, not what we meant.
#The tree saw some asteroids just above 22.0 labeled Y, so it moved its line to fit them, and put the line halfway between 22.03 and 22.04.
#The big takeaway: the tree learned from the labels, not the rule. If the labels are a little messy, the model is a little messy too.
#hypothesis = a guess you can test.