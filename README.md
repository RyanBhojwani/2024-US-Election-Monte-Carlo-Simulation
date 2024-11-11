# 2024-US-Election-Monte-Carlo-Simulation

## Overview
This project implements a Monte Carlo simulation for forecasting U.S. presidential election outcomes using aggregation of polling data. The model incorporates poll quality, sample size, and recency to generate probabilistic predictions for both the Electoral College and state-by-state results.

## Features
- Poll aggregation with intelligent weighting based on:
  - Poll grade (pollster quality)
  - Sample size
  - Recency of poll
- Monte Carlo simulation incorporating:
  - National Electoral College outcomes
  - State-by-state Electoral College outcomes
  - Margin of victory analysis
- Comprehensive output including:
  - Win probabilities for each candidate
  - Expected electoral vote counts
  - State margin analysis
  - Probability distribution of victory margins

## Requirements
- Python 3
- pandas
- numpy

## Installation
```bash
git clone https://github.com/RyanBhojwani/2024-US-Election-Monte-Carlo-Simulation.git
cd 2024-US-Election-Monte-Carlo-Simulation
```

## Usage
1. Prepare your input data:
   - `president_polls.csv`: Polling data with columns for poll grade, state, start date, sample size, party, candidate, and estimated percent of vote
   - `electoral_college.csv`: State electoral vote counts and default predictions

2. Run the simulation:
```python
python election_simulation.py
```

## Input Data Format
Examples of these files are provided
### president_polls.csv
```csv
numeric_grade,state,start_date,end_date,sample_size,party,answer,pct
2.7,Arizona,10/30/24,10/31/24,1005,DEM,Harris,45.9
```

### electoral_college.csv
```csv
State,Votes,in_data,Default_Vote
AL,9,F,Trump
```
Default vote is N/A if in_data is True

## Methodology

### Poll Weighting
Polls are weighted based on three standardized factors:
- Time from reference date (34% weight)
- Sample size (33% weight)
- Pollster grade (33% weight)

### Uncertainty Modeling
Each simulation incorporates:
- Base poll uncertainty (5%)
- Weight-adjusted standard deviations
- Normal distribution sampling

### Electoral College
The simulation follows U.S. Electoral College rules:
- Winner-take-all for all states except ME and NE
- 270 electoral votes needed for victory

## Areas for Enhancement
- Incorporate demographic weighting
- Add economic indicator influences
- Add historical election result calibration
- Enhance uncertainty modeling for different time horizons

