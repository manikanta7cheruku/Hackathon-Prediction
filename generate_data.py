import pandas as pd
import numpy as np

# Set random seed so we get the same results every time
np.random.seed(42)

def generate_hackathon_data(num_rows=2000):
    print(f"Generating {num_rows} rows of synthetic data...")

    # --- 1. Generate Random Features ---
    data = {
        # Random experience between 1 and 10 years
        'Team_Experience_Avg': np.random.randint(1, 11, num_rows),
        
        # Diversity: Low, Medium, High
        'Tech_Stack_Diversity': np.random.choice(['Low', 'Medium', 'High'], num_rows, p=[0.3, 0.5, 0.2]),
        
        # Commit Velocity: Normal distribution (Coding Speed)
        'Commit_Velocity': np.clip(np.random.normal(25, 10, num_rows), 5, 50),
        
        # Communication: Messages per hour
        'Communication_Freq': np.clip(np.random.normal(50, 20, num_rows), 10, 100),
        
        # Sleep: Critical factor (Hours per night)
        'Sleep_Hours': np.clip(np.random.normal(5, 1.5, num_rows), 2, 8),
        
        # Previous wins
        'Prior_Wins': np.random.randint(0, 6, num_rows)
    }

    df = pd.DataFrame(data)

    # --- 2. Calculate "Winner" Logic ---
    # We create a formula so the ML model has a pattern to learn.
    
    # Convert 'High/Low' to numbers for math
    diversity_map = {'Low': 1, 'Medium': 2, 'High': 3}
    div_score = df['Tech_Stack_Diversity'].map(diversity_map)

    # The Winning Formula (Weights)
    score = (
        (df['Team_Experience_Avg'] * 1.5) +
        (df['Commit_Velocity'] * 2.0) +
        (df['Communication_Freq'] * 0.5) +
        (df['Prior_Wins'] * 3.0) +
        (div_score * 5.0)
    )

    # PENALTY: If sleep < 4 hours, remove 30 points (Burnout)
    score = np.where(df['Sleep_Hours'] < 4, score - 30, score)

    # Add random noise (luck)
    score += np.random.normal(0, 15, num_rows)

    # Top 30% of scores get marked as "1" (Winner)
    threshold = np.percentile(score, 70)
    df['Target'] = (score > threshold).astype(int)

    # --- 3. Save Data ---
    df.to_csv('hackathon_data.csv', index=False)
    print("SUCCESS: 'hackathon_data.csv' created.")

if __name__ == "__main__":
    generate_hackathon_data()