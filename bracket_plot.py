import matplotlib.pyplot as plt
import numpy as np
# Define the matchups for each round (each pair is a match)
# For simplicity, the "teams" are their y-positions in this example
rounds = {
    "Round 1": [(1, 2), (3, 4), (5, 6), (7, 8)],
    "Round 2": [(1.5, 3.5), (5.5, 7.5)],
    "Semifinals": [(2.5, 6.5)],
    "Final": [(4.5,)]
}

fig, ax = plt.subplots()

# Initial x position
x = 1

# Plot each round
for round, matchups in rounds.items():
    next_x = x + 1
    for match in matchups:
        
        
        # Plot teams/positions
        plt.plot([x, next_x], [match[0], match[0]], marker='o', markersize=5, color='blue')
        if len(match) > 1:  # If there's a matchup
            plt.plot([x, next_x], [match[1], match[1]], marker='o', markersize=5, color='blue')
            # Connect the match
            plt.plot([next_x, next_x], [match[0], match[1]], marker='', color='grey')
            print(f"match: {match} first val: {match[0]}, second val: {match[1]}")
        print(f"match: {match} first val: {match[0]}")

        print(np.mean(match))
        plt.plot([next_x, next_x+1], [np.mean(match), np.mean(match)], marker='', color='grey')

    x = next_x + 1  # Increment x for the next round

# Customizations
plt.axis('off')  # Hide axes
plt.title("8-Team Tournament Bracket")
plt.tight_layout()
plt.show()
