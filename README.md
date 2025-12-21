# NFL Red Zone Play Sheet

---

## Overview


This project aims to identify optimal red zone play
strategies by analyzing broad offensive schemes and player kinematic tracking data.
In the end, we built a dashboard that allows coaches and players to make split-second decisions by identifying optimal play patterns 5–20 yards from the end zone, including recommended alignments, routes, and acceleration based on predicted defensive coverage. Coaches can also interactively adjust defender angle and distance relative to a targeted receiver to see how alignment affects catch probability.

![Finished Dashboard](NFL%20screenshots/Full%20Dashboard.png)
---

## Objectives

- Discover and define metrics that lead to both a higher proportion of catches and touchdowns
- Create an interface that is intuitive and usable in real-world applications. 

---

## Data Sets

### Train
- 18 weeks of player tracking data captured at 0.1-second frame intervals for every player, play, and game.
- 18 input files containing all passing plays prior to the ball being thrown, including player positioning, player profiles, and ball landing position.
- 18 output files containing all passing plays after the ball was released by the quarterback, with player positioning recorded at each frame interval.


### Supplementary
- Additional play-level information, including catch outcome, offensive and defensive formations, time elapsed, and play commentary.
  
---

## Methodolgy

- Joined input, output, and supplementary datasets and used computational analysis to define new features, including ball catch position, closest defender distance, and defender position relative to the targeted receiver.
- Calculated receiver acceleration using frame-by-frame displacement divided twice by the 0.1-second frame intervals, identified touchdown plays through string matching the keyword “TOUCHDOWN,” and binned the analysis into 5-yard recommendation intervals spanning 5–20 yards from the end zone.
- Explored the data to identify initial trends and develop testable hypotheses.
- Tested the significance of closer defender proximity in the red zone and found the effect to be statistically significant at the 0.01 level.
- Tested defender positioning and found that defenders aligned in front of the receiver were 10% more likely—statistically significant—to allow a completed catch than defenders positioned behind the receiver within the 0.5–2.5 yard proximity range.
- We used a Bayesian approach by defining a prior based on the global average touchdown rate, updating it with observed outcomes to form a posterior distribution, and estimating conservative touchdown probabilities using Monte Carlo simulation.
 

---

## Key Findings

- The distribution of catch frequency by defender proximity is highly right-skewed, with most catches occurring close to a defender, and this skew is even more pronounced in red zone plays.
- At low separation distances (0.5-2.5) yards, receivers classified as in "front of the defender" are expected to catch the ball **at least** 10% more often than under normal conditions.
- Precise receiver alignment is critical, and receiver acceleration has a greater impact later in the route rather than early on.
- The dashboard provides success estimates that are nearly **three times** higher than the baseline red zone touchdown rate at comparable distances to the goal line.


---

## Going Forward

- Integrate quarterback performance metrics to account for passer accuracy.
- Develop play-sequencing recommendations that exploit defensive adjustments across drives.
- Expand the framework beyond touchdown optimization to model success probabilities across the entire field.
- Build on the established framework to support continued innovation in NFL analytics as data-driven decision-making grows in importance.


---

## Project Links
- [pdf to Writeup](Red%20Zone%20Play%20Sheet%20Writeup.pdf)
- [Kaggle Notebook](https://www.kaggle.com/code/karstenlowe0/nfl-analytics-dashboard-final-version/edit)

---

## Tech Stack
- Python
- Pandas, NumPy
- Matplotlib
- Jupyter Notebook
- Seaborn
- Scipy






