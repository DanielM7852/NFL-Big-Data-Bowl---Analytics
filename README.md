#NFL Red Zone Play Sheet
![Catch Rate vs Distance](figures/catch_rate_distance.png) for images
---

## Overview

This project was our submission for the Kaggle Big Data Bowl analytics project 2026
This project aims to identify optimal red zone play
strategies by analyzing broad offensive schemes and player kinematic tracking data.
In the end we built a dashboard that allows coaches and players to make split-second decisions by identifying optimal play patterns 5–20 yards from the end zone, including recommended alignments, routes, and acceleration based on predicted defensive coverage. Coaches can also interactively adjust defender angle and distance relative to a targeted receiver to see how alignment affects catch probability.

---

## Objectives

- Create success metrics for both touchdown probability and catch probability
- ....add more

---

## Data Sets

### Train
- 18 weeks of 0.1s frame intervals tracking every player for each play in every game.
- 18 input files which included all passing plays (before the ball was thrown) with stats on player postioning as well as player profiles and ball landing positon.
- 18 output files which included all passing plays (after the ball was released by QB) with stats on player positoning at frame intervals.

### Supplementary
- Additonal information on plays including catch result, formation (defense and offense), time elapsed, and play commentary.
  
---

## Methodolgy

---

## Key Findings

- The distribution of catch frequency by defender proximity is highly right-skewed, meaning most catches occur close to a defender.
- This effect is even more pronounced in red zone plays, resulting in an even more extreme skew.


- At low seperation distances (0.5-2.5) yards, recivers classified as in "front of the defender" are expected to catch the ball 10% more often than under normal conditions.
- 
This analysis successfully transformed raw NFL tracking data into actionable coaching intelligence for red zone play calling. By combining spatial catch probability modeling with Bayesian statistical techniques and interactive visualization, the system provided coaches with reliable, data-driven play recommendations tailored to specific game situations.

---

## Going forward

Future enhancements could integrate quarterback performance metrics to account for passer accuracy, develop play sequencing recommendations to exploit defensive adjustments across drives, and expand beyond touchdown optimization to include success probabilities at any part of the field. The framework established here provides a robust foundation for ongoing NFL analytics innovation in a time where the power of data continues to increase.

---

## Project Links
- [GitHub Repository](#)
- [Kaggle Notebook](#)

---

## Tech Stack
- Python
- Pandas, NumPy
- Matplotlib
- Jupyter Notebook






