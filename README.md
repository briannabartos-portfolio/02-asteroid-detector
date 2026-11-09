# 🌍 Asteroid Danger Detector

**Can a machine learning model learn which asteroids are dangerous to Earth just from examples, without being told the rule?**

NASA labels some near-Earth asteroids as **Potentially Hazardous (PHA)** using a written rule: the asteroid must be big enough (absolute magnitude **H ≤ 22.0**, about 140 m or larger) **and** its orbit must come close enough to Earth's (**MOID ≤ 0.05 au**). This project trains a decision tree on real NASA data to see whether it can rediscover that rule, then makes the problem harder and adds safety checks around the model.

Built as part of my journey learning machine learning and AI safety from scratch.

---

## 📊 Key findings

### Level 1: With MOID, the tree rediscovers NASA's rule
- Trained on 42,669 real asteroids from [NASA JPL's Small-Body Database](https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html), a depth-2 decision tree learned **H ≤ 22.035 AND MOID ≤ 0.05**, almost exactly NASA's rule.
- On a hidden test set (8,534 asteroids), it caught **505 of 510** hazardous asteroids (**recall 0.99**). A lazy "always safe" model scores 94% accuracy but catches **0**.

### 🔎 NASA's own labels have inconsistencies
- **50 asteroids** have PHA labels that disagree with NASA's own rule, all near the H = 22.0 border (H between 21.70 and 22.57).
- An experiment confirmed these noisy labels **pulled the tree's line** from 22.0 to 22.035. Retraining on labels that follow the rule moved it back to 22.005.
- **7 of the tree's 8 test "mistakes"** were actually these noisy labels. The tree had followed the rule.

### 🛡️ Safeguards around the model
- **Unknown ≠ safe:** 137 asteroids missing data were set aside, never labeled safe.
- **Human-review zone:** asteroids close to Earth's path and just under the size cutoff are sent to a human. The zone edge (H = 24.3, about 50 m, Tunguska-event size) was chosen **for a reason, before looking at results**, to avoid tuning on the test set. Result: **0 hazardous asteroids called safe**, with about 10% of test asteroids sent to review.

### Level 2: Without MOID, it gets much harder
Using only size and orbit shape (q, aphelion, a, e, inclination):

| Max depth | Recall | Leaves |
|---|---|---|
| 2 | 0.00 | 4 |
| 4 | 0.33 | 15 |
| 8 | 0.45 | 100 |
| None | 0.51 | 1,307 |

- The tree discovered on its own that **flat (low-inclination) orbits** are more often dangerous.
- The unlimited tree scored 100% on its training data but 51% on new data: **overfitting**.
- **The right clue matters more than a bigger model:** a 4-leaf tree *with* MOID beat a 1,307-leaf tree *without* it.

---

## 📁 Files

| File | What it does |
|---|---|
| `get_asteroids.py` | Downloads H, MOID, and PHA labels from NASA's API |
| `explore_data.py` | Finds missing data and sets incomplete rows aside |
| `train_model.py` | Trains a decision tree and prints its rules |
| `experiment_labels.py` | Finds the 50 mislabeled asteroids and retrains on clean labels |
| `evaluate_model.py` | Train/test split, recall, and an investigation of each mistake |
| `close_calls.py` | Adds a human-review zone for near-cutoff asteroids |
| `get_orbits.py` | Level 2: downloads orbit-shape data |
| `train_orbits.py` | Level 2: trains without MOID at different depths |
| `STUDY_GUIDE.md` | My study notes for every concept in the project |

The `.csv` files are the exact data snapshots used, since NASA's database changes daily.

---

## ▶️ How to run

Requires Python 3 with pandas and scikit-learn:

```bash
python3 -m pip install pandas scikit-learn
```

Then run any file, for example:

```bash
python3 evaluate_model.py
```

---

## 🧠 What I learned

APIs and JSON · pandas · decision trees · train/test splits · recall vs. accuracy · imbalanced data · label noise · overfitting · interpretability · human-in-the-loop safeguards · avoiding tuning on the test set · the scientific method applied to ML
