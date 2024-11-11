import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import Dict, Any

def load_and_prepare_polling_data(polling_file: str, reference_date: datetime = datetime(2024, 11, 5)) -> pd.DataFrame:
    """
    Load polling data from CSV and prepare it for analysis by calculating weights
    and adjusting for various factors.
    
    Args:
        polling_file (str): Path to the polling data CSV file
        reference_date (datetime): Reference date for calculating poll age
    
    Returns:
        pd.DataFrame: Processed polling data with calculated weights
    """
    df = pd.read_csv(polling_file)
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['days_from_ref'] = (reference_date - df['start_date']).dt.days

    # Apply poll bias adjustments if needed
    df.loc[df['party'] == 'DEM', 'pct'] += 0  # Adjust Democratic numbers
    df.loc[df['party'] == 'REP', 'pct'] -= 0  # Adjust Republican numbers

    # Calculate mean and standard deviation for each factor
    mean_grade = df['numeric_grade'].mean()
    std_grade = df['numeric_grade'].std()
    mean_sample_size = df['sample_size'].mean()
    std_sample_size = df['sample_size'].std()
    mean_days = df['days_from_ref'].mean()
    std_days = df['days_from_ref'].std()

    # Calculate Z-scores for each factor
    df['z_grade'] = (df['numeric_grade'] - mean_grade) / std_grade
    df['z_sample_size'] = (df['sample_size'] - mean_sample_size) / std_sample_size
    df['z_date'] = -1 * (df['days_from_ref'] - mean_days) / std_days  # Negative so recent polls get higher weight

    # Calculate overall weight combining all factors
    df['overall_z_weight'] = (0.34 * df['z_date'] + 0.33 * df['z_sample_size'] + 0.33 * df['z_grade'])
    
    # Normalize weights to be positive and sum to 1 within groups
    min_weight = df['overall_z_weight'].min()
    df['adjusted_weight'] = df['overall_z_weight'] + abs(min_weight) + 0.01
    df['normalized_weight'] = df.groupby(['state', 'answer'])['adjusted_weight'].transform(lambda x: x / x.sum())

    # Calculate poll standard deviations
    base_stdev = 0.05  # 5% base standard deviation
    df['stdev'] = calculate_poll_standard_deviations(df, base_stdev)
    
    return df


def calculate_poll_standard_deviations(df: pd.DataFrame, base_stdev: float) -> pd.Series:
    """
    Calculate standard deviations for each poll based on their weights.
    
    Args:
        df (pd.DataFrame): Polling data with calculated weights
        base_stdev (float): Base standard deviation to use for calculations
    
    Returns:
        pd.Series: Calculated standard deviations for each poll
    """
    weight_range = df['overall_z_weight'].max() - df['overall_z_weight'].min()
    stdev = base_stdev * (1 - (df['overall_z_weight'] - df['overall_z_weight'].min()) / weight_range)
    
    # Normalize to ensure mean standard deviation matches base_stdev
    mean_stdev = stdev.mean()
    stdev = stdev * (base_stdev / mean_stdev)
    
    return stdev.clip(lower=0.01)  # Ensure minimum std dev of 1%


def aggregate_polling_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate polling data by state and candidate using weighted averages.
    
    Args:
        df (pd.DataFrame): Processed polling data
    
    Returns:
        pd.DataFrame: Aggregated polling data by state and candidate
    """
    weighted_agg = df.groupby(['state', 'answer']).apply(
        lambda x: pd.Series({
            'Weighted_Average_pct': (x['pct'] * x['normalized_weight']).sum(),
            'Weighted_Std_Dev': (x['stdev'] * x['normalized_weight']).sum()
        })
    ).reset_index()
    
    weighted_agg.columns = ['State', 'Candidate', 'Weighted_Average_pct', 'Weighted_Std_Dev']
    return weighted_agg


def add_missing_states(weighted_df: pd.DataFrame, electoral_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add missing states to the weighted data using default votes.
    
    Args:
        weighted_df (pd.DataFrame): Aggregated polling data
        electoral_df (pd.DataFrame): Electoral college data with default votes
    
    Returns:
        pd.DataFrame: Complete dataset including missing states
    """
    existing_states = set(weighted_df['State'].unique())
    new_rows = []

    for _, row in electoral_df.iterrows():
        state = row['State']
        if state not in existing_states:
            if row['Default_Vote'] == 'Trump':
                new_rows.extend([
                    {'State': state, 'Candidate': 'Trump', 'Weighted_Average_pct': 100, 'Weighted_Std_Dev': 0},
                    {'State': state, 'Candidate': 'Harris', 'Weighted_Average_pct': 0, 'Weighted_Std_Dev': 0}
                ])
            elif row['Default_Vote'] == 'Harris':
                new_rows.extend([
                    {'State': state, 'Candidate': 'Harris', 'Weighted_Average_pct': 100, 'Weighted_Std_Dev': 0},
                    {'State': state, 'Candidate': 'Trump', 'Weighted_Average_pct': 0, 'Weighted_Std_Dev': 0}
                ])

    return pd.concat([weighted_df, pd.DataFrame(new_rows)], ignore_index=True)


def simulate_single_election(weighted_df: pd.DataFrame, electoral_df: pd.DataFrame, rng: np.random.RandomState = None) -> Dict[str, Any]:
    """
    Simulate a single election using the polling data and electoral votes.
    
    Args:
        weighted_df (pd.DataFrame): Complete polling data
        electoral_df (pd.DataFrame): Electoral college data
        rng (np.random.RandomState, optional): Random number generator
    
    Returns:
        dict: Election results including electoral votes and state-by-state results
    """
    if rng is None:
        rng = np.random.RandomState()
        
    electoral_dict = electoral_df.set_index('State')['Votes'].to_dict()
    harris_votes = trump_votes = 0
    state_results = []
    
    for state in electoral_dict:
        state_data = weighted_df[weighted_df['State'] == state]
        if len(state_data) == 0:
            continue
            
        harris_data = state_data[state_data['Candidate'] == 'Harris'].iloc[0]
        trump_data = state_data[state_data['Candidate'] == 'Trump'].iloc[0]
        
        # Generate random adjustments based on poll uncertainty
        harris_adj = rng.normal(0, 1) * harris_data['Weighted_Std_Dev'] * 100
        trump_adj = rng.normal(0, 1) * trump_data['Weighted_Std_Dev'] * 100
        
        harris_result = harris_data['Weighted_Average_pct'] + harris_adj
        trump_result = trump_data['Weighted_Average_pct'] + trump_adj
        
        state_results.append({
            'state': state,
            'harris_pct': harris_result,
            'trump_pct': trump_result,
            'margin': harris_result - trump_result,
            'electoral_votes': electoral_dict[state]
        })
        
        if harris_result > trump_result:
            harris_votes += electoral_dict[state]
        else:
            trump_votes += electoral_dict[state]
    
    return {'harris_votes': harris_votes, 'trump_votes': trump_votes, 'state_results': state_results}


def classify_margin_of_victory(margin: float, winner: str) -> str:
    """
    Classify the margin of victory into categories.
    
    Args:
        margin (float): Electoral vote margin
        winner (str): Winner of the election ('harris' or 'trump')
    
    Returns:
        str: Category of victory margin
    """
    categories = [(4, '1_4'), (14, '5_14'), (34, '15_34'), 
                 (64, '35_64'), (104, '65_104'), (float('inf'), '104_plus')]
    
    for threshold, category in categories:
        if margin <= threshold:
            return f"{winner}_win_{category}"
    return f"{winner}_win_104_plus"


def run_election_simulation(weighted_df: pd.DataFrame, electoral_df: pd.DataFrame, n_simulations: int = 1000) -> Dict[str, Any]:
    """
    Run multiple election simulations and analyze the results.
    
    Args:
        weighted_df (pd.DataFrame): Complete polling data
        electoral_df (pd.DataFrame): Electoral college data
        n_simulations (int): Number of simulations to run
    
    Returns:
        dict: Comprehensive simulation results and analysis
    """
    harris_wins = 0
    harris_ev_total = trump_ev_total = 0
    state_margins = defaultdict(list)
    margin_categories = defaultdict(int)
    
    for _ in range(n_simulations):
        result = simulate_single_election(weighted_df, electoral_df)
        
        # Track wins and electoral votes
        if result['harris_votes'] > result['trump_votes']:
            harris_wins += 1
            margin_category = classify_margin_of_victory(
                abs(result['harris_votes'] - result['trump_votes']), 'Harris'
            )
        else:
            margin_category = classify_margin_of_victory(
                abs(result['harris_votes'] - result['trump_votes']), 'Trump'
            )
            
        margin_categories[margin_category] += 1
        harris_ev_total += result['harris_votes']
        trump_ev_total += result['trump_votes']
        
        # Track state margins
        for state_result in result['state_results']:
            state_margins[state_result['state']].append(state_result['margin'])
    
    # Calculate final statistics
    harris_win_prob = (harris_wins / n_simulations) * 100
    state_avg_margins = {
        state: np.mean(margins) for state, margins in state_margins.items()
    }
    
    return {
        'win_probabilities': {
            'harris': harris_win_prob,
            'trump': 100 - harris_win_prob
        },
        'average_electoral_votes': {
            'harris': harris_ev_total / n_simulations,
            'trump': trump_ev_total / n_simulations
        },
        'closest_states': sorted(
            [{'state': state, 'avg_margin': margin} 
             for state, margin in state_avg_margins.items()],
            key=lambda x: abs(x['avg_margin'])
        )[:15],
        'margin_of_victory_probabilities': {
            category: (count / n_simulations) * 100
            for category, count in margin_categories.items()
        }
    }


def print_simulation_results(results: Dict[str, Any], n_simulations: int) -> None:
    """
    Print formatted simulation results.
    
    Args:
        results (dict): Simulation results from run_election_simulation
        n_simulations (int): Number of simulations run
    """
    print("\nElection Simulation Results")
    print("==========================")
    print(f"Based on {n_simulations:,} simulations\n")
    
    print("Win Probabilities:")
    print(f"Harris: {results['win_probabilities']['harris']:.1f}%")
    print(f"Trump:  {results['win_probabilities']['trump']:.1f}%\n")
    
    print("Average Electoral Votes:")
    print(f"Harris: {results['average_electoral_votes']['harris']:.1f}")
    print(f"Trump:  {results['average_electoral_votes']['trump']:.1f}\n")
    
    print("Top 15 Closest States:")
    print("---------------------")
    for state_info in results['closest_states']:
        leader = "Harris" if state_info['avg_margin'] > 0 else "Trump"
        margin = abs(state_info['avg_margin'])
        print(f"{state_info['state']}: {leader} +{margin:.2f}%")
    
    print("\nMargin of Victory Probabilities:")
    print("--------------------------------")
    for category, probability in results['margin_of_victory_probabilities'].items():
        print(f"{category.replace('_', ' ').capitalize()}: {probability:.2f}%")


def main():
    """
    Main function to run the election simulation.
    """
    # Load and process polling data
    polling_df = load_and_prepare_polling_data("president_polls.csv")
    electoral_df = pd.read_csv("electoral_college.csv")
    
    # Aggregate polling data
    weighted_df = aggregate_polling_data(polling_df)
    
    # Add missing states
    complete_df = add_missing_states(weighted_df, electoral_df)
    
    # Run simulation
    results = run_election_simulation(complete_df, electoral_df, n_simulations=10000)
    
    # Print results
    print_simulation_results(results, n_simulations=10000)

if __name__ == "__main__":
    main()