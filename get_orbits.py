# =====================================================================
# PROJECT 02: ASTEROID DANGER DETECTOR, LEVEL 2 (HARDER!)
# FILE: get_orbits.py
# =====================================================================
#
# THE BIG QUESTION (for the whole project):
#   Can a machine learning model figure out which asteroids are
#   dangerous to Earth just by studying examples, without anyone
#   telling it the rule?
#
# WHAT'S DIFFERENT IN LEVEL 2:
#   In Level 1 we gave the tree MOID, the closest the asteroid's path
#   gets to Earth's path. That's basically HALF of NASA's rule handed
#   over for free!
#
#   But MOID isn't something a telescope sees. Astronomers have to
#   CALCULATE it, from the orbit's shape AND which way the orbit is
#   pointing in space, compared to Earth's orbit.
#
#   So in Level 2, we take MOID away. The tree gets only H (size) and
#   numbers that describe the orbit's SHAPE. Can it still figure out
#   which asteroids are dangerous?
#
# WHY IT MATTERS:
#   Real problems rarely come with the perfect clue already
#   calculated. Usually you have raw measurements, and the pattern
#   is hidden inside them.
#
# AI SAFETY CONNECTION:
#   When a problem gets harder, models often get BIGGER and harder to
#   read. We'll watch what happens to interpretability (can we still
#   read the tree like a flowchart?) as the problem gets harder.
#
# NEW SPACE WORDS IN THIS FILE (you saw q, Q, and a in the orbit
# families picture!):
#   q   = perihelion: the CLOSEST the asteroid gets to the SUN (in au)
#   ad  = aphelion: the FARTHEST it gets from the Sun (in au).
#         The families picture called this "Q". NASA's API calls it
#         "ad" (short for "aphelion distance").
#   a   = semi-major axis: its AVERAGE distance from the Sun (in au)
#   e   = eccentricity: how STRETCHED the oval is.
#           0   = a perfect circle
#           0.5 = a stretched oval
#           close to 1 = a very long, skinny oval
#   i   = inclination: how TILTED the orbit is compared to Earth's,
#         in degrees. 0 = flat, same as Earth. 90 = straight up and down.
#
#   Picture (looking down from above the Sun):
#
#            ad (farthest)
#              ●
#          .       .
#        .           .
#       .      ☀️      .
#        .           .
#          .       .
#              ●
#            q (closest)
#
# THE PLAN FOR LEVEL 2:
#   1. get_orbits.py    <- THIS FILE: download size + orbit shape
#   2. train_orbits.py  (next) train a tree WITHOUT moid
#
# HOW TO READ THIS FILE:
#   Lines that start with "#" are COMMENTS. Python skips them.
#   Wherever you see "👉 YOUR TURN", type the code line right below it.
#   This file is a lot like get_asteroids.py, so it's good review!
# =====================================================================


# ---------------------------------------------------------------
# STEP 1: Bring in the tools (same as get_asteroids.py)
# ---------------------------------------------------------------
# json             = reads NASA's answer (JSON text)
# urllib.request   = visits a web address
# pandas as pd     = makes tables
# 👉 YOUR TURN: import json
import json

# 👉 YOUR TURN: import urllib.request
import urllib.request

# 👉 YOUR TURN: import pandas as pd
import pandas as pd


# ---------------------------------------------------------------
# STEP 2: Write the question for NASA, now with MORE columns
# ---------------------------------------------------------------
# The same address as get_asteroids.py, but "fields=" asks for 5 more
# columns: q, ad, a, e, i (the orbit-shape numbers from the header).
#
# We still download moid and pha:
#   pha  = we still need NASA's answer (the LABEL) to train and grade
#   moid = we WON'T give it to the tree, but we keep it so we can
#          compare later
#
# Each "&" separates one part of the question from the next:
#   fields=...      -> "send me these columns"
#   sb-group=neo    -> "only near-Earth objects"
#   sb-kind=a       -> "only asteroids"
# 👉 YOUR TURN: type these 4 lines (lines 2 and 3 indented 4 spaces)
#   url = (
#       "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
#       "?fields=full_name,H,moid,pha,q,ad,a,e,i&sb-group=neo&sb-kind=a"
#   )
url = (
    "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
    "?fields=full_name,H,moid,pha,q,ad,a,e,i&sb-group=neo&sb-kind=a"
)   


# ---------------------------------------------------------------
# STEP 3: Ask NASA and read the answer (same as get_asteroids.py)
# ---------------------------------------------------------------
# Open the connection, call the reply "response", read the JSON
# into a dictionary called "result", then close the connection.
# 👉 YOUR TURN: type these 2 lines (the second one indented 4 spaces)
#   with urllib.request.urlopen(url) as response:
#       result = json.load(response)
with urllib.request.urlopen(url) as response:
    result = json.load(response)


# ---------------------------------------------------------------
# STEP 4: Make a table and save it, with a NEW file name
# ---------------------------------------------------------------
# We save to orbits.csv, NOT asteroids.csv, so we don't overwrite
# our Level 1 data. That way all our earlier results stay
# reproducible.
# 👉 YOUR TURN: data = pd.DataFrame(result["data"], columns=result["fields"])
data = pd.DataFrame(result["data"], columns=result["fields"])

# 👉 YOUR TURN: data.to_csv("orbits.csv", index=False)
data.to_csv("orbits.csv", index=False)


# ---------------------------------------------------------------
# STEP 5: Check what we got
# ---------------------------------------------------------------
# 👉 YOUR TURN: print(data.head())
print(data.head())

# 👉 YOUR TURN: print("Total asteroids:", len(data))
print("Total asteroids:", len(data))