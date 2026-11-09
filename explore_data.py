# =====================================================================
# PROJECT 02: ASTEROID DANGER DETECTOR
# FILE: explore_data.py
# =====================================================================
#
# THE BIG QUESTION (for the whole project):
#   Can a machine learning model figure out which asteroids are
#   dangerous to Earth just by studying examples, without anyone
#   telling it the rule?
#
# THE QUESTION FOR THIS FILE:
#   Our data has 42,806 asteroids, but N (40,121) + Y (2,550) = 42,671.
#   That's 135 short. Where did they go?
#
# WHY IT MATTERS:
#   Before you train a model, you have to understand your data.
#   If some rows are broken or incomplete and we don't notice, the
#   model could learn from bad examples, or crash.
#
# AI SAFETY CONNECTION:
#   This is DATA VALIDATION, which you learned in the fruit lessons.
#   Many real AI failures started with data nobody checked.
#
# ML WORDS IN THIS FILE:
#   missing value = a blank spot in the table where data should be
#   NaN           = "Not a Number," how pandas shows a missing value
#   outlier       = a value that IS there but is unusually far from
#                   the others (like an asteroid 1,000x bigger than
#                   the rest). Different from missing!
#
# THE PLAN:
#   1. get_asteroids.py   <- done! downloaded the data
#   2. explore_data.py    <- THIS FILE: check the data for problems
#   3. train the model    (later)
#   4. check if it found NASA's rule  (later)
#
# HOW TO READ THIS FILE:
#   Lines that start with "#" are COMMENTS. Python skips them.
#   Wherever you see "👉 YOUR TURN", type the code line right below it.
# =====================================================================


# ---------------------------------------------------------------
# STEP 1: Bring in pandas
# ---------------------------------------------------------------
# We only need pandas this time. We're not talking to NASA anymore;
# we saved the data to a file, so we just read the file.
# (That's the payoff of saving it: no need to download it again!)
# 👉 YOUR TURN: import pandas as pd
import pandas as pd


# ---------------------------------------------------------------
# STEP 2: Load the saved data
# ---------------------------------------------------------------
# pd.read_csv opens a CSV file and turns it into a table (DataFrame).
# It's the opposite of to_csv:
#     to_csv   = table -> file   (saving)
#     read_csv = file  -> table  (loading)
# You used read_csv in the fruit lessons with fruits.csv.
#
#   data           = a variable (box) to hold the table
#   "asteroids.csv" = the file name, in quotes because it's text
# 👉 YOUR TURN: data = pd.read_csv("asteroids.csv")
data = pd.read_csv("asteroids.csv")



# ---------------------------------------------------------------
# STEP 3: Count the missing values in each column
# ---------------------------------------------------------------
# This line chains two methods together, one after the other:
#
#   data.isna()  -> goes through EVERY cell in the table and asks
#                   "is this missing?" It answers True or False
#                   for each one.
#                   (isna = "is NA," and NA = "not available")
#
#   .sum()       -> adds them up, column by column. In Python, True
#                   counts as 1 and False counts as 0, so adding up
#                   the Trues COUNTS the missing cells.
#
# Picture it:
#     data:              data.isna():         .sum():
#     H     moid         H      moid          H     0
#     10.4  0.149        False  False         moid  1
#     25.8  (blank)  ->  False  True     ->
# 👉 YOUR TURN: print(data.isna().sum())
print(data.isna().sum())



# ---------------------------------------------------------------
# STEP 4: Look at some of the rows with a missing label
# ---------------------------------------------------------------
# Counting is good, but let's SEE a few of them.
#
#   data["pha"].isna()  -> True/False for each row: is pha missing?
#   data[ ... ]         -> putting True/False inside the table's square
#                          brackets keeps ONLY the rows that are True.
#                          This is called FILTERING.
#   .head()             -> show just the first 5 of those rows
# 👉 YOUR TURN: print(data[data["pha"].isna()].head())
print(data[data["pha"].isna()].head())


# =====================================================================
# OUR DECISION: OPTION C, SET THE UNKNOWNS ASIDE
# =====================================================================
# Unknown does NOT mean safe. Instead of deleting rows with blanks or
# guessing what's in them, we split the table into two groups:
#
#     known   -> every column filled in. The model trains on these.
#     unknown -> at least one blank. Saved in their own file so we
#                never lose track of them, and never label them
#                "safe" by accident.
#
#              ┌──────────────────────┐
#              │  data (all 42,806)   │
#              └──────────┬───────────┘
#          any blank in this row?
#              ┌──────────┴───────────┐
#             NO                     YES
#      ┌──────────────┐       ┌──────────────┐
#      │    known     │       │   unknown    │
#      │ train on it  │       │ set aside 🚩 │
#      └──────────────┘       └──────────────┘


# ---------------------------------------------------------------
# STEP 5: Mark every row that has at least one blank
# ---------------------------------------------------------------
#   data.isna()   -> True/False for every cell: "is it missing?"
#                    (you used this in Step 3)
#   .any(axis=1)  -> for each ROW, asks "is ANY cell in this row True?"
#                    axis=1 means "look across each row"
#                    (axis=0 would mean "look down each column")
#
#   has_blank     -> a new variable: one True/False per asteroid.
#                    True = this asteroid has a blank somewhere.
#
# Picture it:
#     H      moid   pha           has_blank
#     False  False  False   ->    False   (complete row)
#     False  True   True    ->    True    (has a blank)
# 👉 YOUR TURN: has_blank = data.isna().any(axis=1)
has_blank = data.isna().any(axis=1)


# ---------------------------------------------------------------
# STEP 6: Split the table into two groups
# ---------------------------------------------------------------
# This is FILTERING again (like Step 4): True/False inside the
# table's square brackets keeps only the rows that are True.
#
#   data[has_blank]   -> keeps rows WITH a blank      -> unknown
#   data[~has_blank]  -> keeps rows WITHOUT a blank   -> known
#
#   ~  = "NOT." It flips every True to False and every False to True.
#        So ~has_blank means "has NO blank."
# 👉 YOUR TURN: unknown = data[has_blank]
unknown = data[has_blank]


# 👉 YOUR TURN: known = data[~has_blank]
known = data[~has_blank]

# ---------------------------------------------------------------
# STEP 7: Save each group to its own file, and check the counts
# ---------------------------------------------------------------
# to_csv again (same as in get_asteroids.py). Two files this time:
#   known_asteroids.csv   -> what the model will learn from
#   unknown_asteroids.csv -> the "we're not sure yet" list
# 👉 YOUR TURN: known.to_csv("known_asteroids.csv", index=False)
known.to_csv("known_asteroids.csv", index=False)


# 👉 YOUR TURN: unknown.to_csv("unknown_asteroids.csv", index=False)
unknown.to_csv("unknown_asteroids.csv", index=False)


# Always check your work! known + unknown should equal the total.
# If they don't, we lost rows somewhere.
# 👉 YOUR TURN: print("Known:", len(known))
print("Known:", len(known))


# 👉 YOUR TURN: print("Unknown (set aside):", len(unknown))
print("Unknown (set aside):", len(unknown))


# 👉 YOUR TURN: print("Total:", len(data))
print("Total:", len(data))
