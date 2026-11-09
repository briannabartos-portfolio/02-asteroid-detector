# =====================================================================
# PROJECT 02: ASTEROID DANGER DETECTOR
# =====================================================================
#
# THE BIG QUESTION:
#   Can a machine learning model figure out which asteroids are
#   dangerous to Earth just by studying examples, without anyone
#   telling it the rule?
#
# WHY IT MATTERS:
#   NASA labels some asteroids as "Potentially Hazardous" (PHA) using
#   a written rule: the asteroid must be BIG enough (H <= 22.0) AND
#   come CLOSE enough to Earth's orbit (MOID <= 0.05 au).
#   If our model rediscovers that rule from examples alone, we know it
#   learned the REAL pattern, not a shortcut.
#
# AI SAFETY CONNECTION:
#   Missing a dangerous asteroid is much worse than a false alarm.
#   Real AI systems face this trade-off all the time: which mistake
#   is worse, and how do we make the model avoid it?
# ML connection: Nearly every ML project follows this same pattern: get raw data (JSON from a website), turn it into a table (DataFrame), and save it (CSV). Saving matters because you only need to download from NASA once. After that, you can train models over and over from the saved file, and everyone who uses your project gets the same data. Being able to repeat an experiment exactly is called reproducibility, and it’s a big deal in AI safety research.
#
# ML WORDS IN THIS PROJECT:
#   feature  = a clue the model looks at (size, distance)
#   label    = the answer we want the model to predict (hazardous: Y/N)
#   training = showing the model many examples so it learns a pattern
#   decision tree = a model that learns a chain of yes/no questions,
#                   like "Is H <= 22? Is MOID <= 0.05?"
#
# THE PLAN:
#   1. get_asteroids.py   <- THIS FILE: download real NASA data
#   2. train the model    (later)
#   3. check if it found NASA's rule  (later)
#   4. make it harder and more realistic  (later)
#
# WHAT THIS FILE DOES:
#   Asks NASA's database for near-Earth asteroids and saves them in a
#   table file called asteroids.csv, so we can train a model on it.
# =====================================================================
#
# HOW TO READ THIS FILE:
#   Lines that start with "#" are COMMENTS. Python skips them completely.
#   They're notes for humans, explaining what the code is doing.
#   Wherever you see "👉 YOUR TURN", type the code line right below it.

# ---------------------------------------------------------------
# STEP 1: Bring in the tools we need
# ---------------------------------------------------------------
# "import" means "load a tool so this program can use it."
# Think of it like taking tools out of a toolbox before a job.
# Python has hundreds of built-in tools, but it doesn't load any of
# them until you ask, which keeps programs fast.

# json: NASA sends its answer as text in a format called JSON
# (JavaScript Object Notation). It looks like {"name": "Eros", "H": 10.4}.
# This tool turns that text into Python lists and dictionaries.
# 👉 YOUR TURN: import json
import json

# urllib.request: Python's built-in tool for visiting a web address,
# like a web browser does, but without showing a page on screen.
#   urllib  = the toolbox for working with URLs (web addresses)
#   .       = "go inside" (urllib.request means "the request tool
#             inside the urllib toolbox")
#   request = the tool that sends a request to a website
# 👉 YOUR TURN: import urllib.request
import urllib.request


# pandas: the table tool you used in the fruit lessons.
# "as pd" gives it a short nickname, so we can type pd instead of pandas.
# pandas does NOT come with Python. It's installed separately, which is
# why you needed pip for it in the fruit lessons.
# 👉 YOUR TURN: import pandas as pd
import pandas as pd



# ---------------------------------------------------------------
# STEP 2: Write the question we'll ask NASA NASA’s own documentation
# ---------------------------------------------------------------
# API (Application Programming Interface) = a website built for programs instead of people.
# Documentation = the instruction manual a company writes for its API. It explains what address to use, what you can ask for, and what every column means.
# A web address can carry a question. Everything after the "?" is
# our question, and each "&" separates one part of it from the next.
#
#   fields=full_name,H,moid,pha   -> "only send me these 4 columns"
#   sb-group=neo                  -> "only Near-Earth Objects"
#   sb-kind=a                     -> "only asteroids (no comets)"
#
# What the 4 columns mean:
#   full_name -> the asteroid's name
#   H         -> how bright it is. Brightness tells us size:
#                a SMALLER H means a BIGGER asteroid.
#   moid      -> the closest the asteroid's orbit ever gets to Earth's
#                orbit, measured in "au" (1 au = the Earth-Sun distance)
#   pha       -> NASA's label: is it a Potentially Hazardous Asteroid?
#                "Y" for yes, "N" for no. This is our LABEL; H and moid
#                are our FEATURES.
#
#   url = a VARIABLE: a name we pick that stores something for later.
#         Here it stores the web address.
#   ( ) = the parentheses let us split one long piece of text across
#         two lines. Python glues the two pieces into one address.
#   " " = quote marks mean "this is text," not code.
# 1. Ask the question      →  "Is there NASA asteroid data I can download?"
# 2. Search the web        →  "NASA asteroid API"
# 3. Find the docs         →  ssd-api.jpl.nasa.gov
# 4. Read the docs         →  learn the address and the column names
# 5. Test a small request  →  ask for only 3 asteroids first
# 6. Write the code
# ML connection: Before you train any model, you have to understand your data: where it came from, who made it, and what each column means. If you skip this, you might train a model on a column that doesn’t mean what you think it does. That’s how ML systems end up making confident but wrong predictions, which is an AI safety problem.
# 👉 YOUR TURN: type these 4 lines
#   url = (
#       "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
#       "?fields=full_name,H,moid,pha&sb-group=neo&sb-kind=a"
#   )

url = (
    "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
    "?fields=full_name,H,moid,pha&sb-group=neo&sb-kind=a"
)


# ---------------------------------------------------------------
# STEP 3: Ask NASA and read the answer
# ---------------------------------------------------------------
# urllib.request.urlopen(url) visits the web address and gets NASA's reply.
#
#   with ... as response:  means "open the connection, call the reply
#                          'response', and close the connection
#                          automatically when we're done."
#   :                      the colon means "the indented lines below
#                          belong to this."
#   indent (4 spaces)      lines pushed in by 4 spaces run while the
#                          connection is open.
#
# json.load(response) reads NASA's JSON text and turns it into a
# Python DICTIONARY (a collection of labeled pieces). We store it
# in a variable called "result".
# 👉 YOUR TURN: type these 2 lines (the second one is indented 4 spaces)
#   with urllib.request.urlopen(url) as response:
#       result = json.load(response)

with urllib.request.urlopen(url) as response:
    result = json.load(response)

# At this point, "result" looks roughly like this:
#   {
#     "fields": ["full_name", "H", "moid", "pha"],        <- column names
#     "data":   [["433 Eros", "10.4", "0.149", "N"], ...] <- one list per asteroid
#   }



# ---------------------------------------------------------------
# STEP 4: Turn NASA's answer into a table and save it
# ---------------------------------------------------------------
# pd.DataFrame builds a table (pandas calls a table a "DataFrame").
#   result["data"]   -> the rows (one row per asteroid)
#   result["fields"] -> the column names for the top of the table
#   [" "]            -> square brackets with a name inside mean
#                       "get the piece with this label"
# 👉 YOUR TURN: data = pd.DataFrame(result["data"], columns=result["fields"])
data = pd.DataFrame(result["data"], columns=result["fields"])

# Save the table as a CSV file (Comma-Separated Values), just like
# fruits.csv. index=False means "don't add an extra column of row
# numbers (0, 1, 2...) to the file."
# 👉 YOUR TURN: data.to_csv("asteroids.csv", index=False)
data.to_csv("asteroids.csv", index=False)



# ---------------------------------------------------------------
# STEP 5: Print a few things so we can check what we got
# ---------------------------------------------------------------
# Always look at your data before training a model on it! This is
# how you catch problems early (remember data validation?).

# head() shows the first 5 rows, a quick peek to check it looks right.
# 👉 YOUR TURN: print(data.head())
print(data.head())


# len(data) counts the rows, which tells us how many asteroids we got.
# 👉 YOUR TURN: print("Total asteroids:", len(data))
print("Total asteroids:", len(data))

# value_counts() counts each different value in the "pha" column,
# so it tells us how many asteroids are hazardous (Y) and how many aren't (N).
# 👉 YOUR TURN: print(data["pha"].value_counts())
print(data["pha"].value_counts())

#The 135 missing asteroids are ones whose orbit isn’t known well enough to calculate MOID, so NASA couldn’t label them.
#So when someone says “my model is 94% accurate,” we should ask: “Okay, but does it catch the dangerous ones?”