import matplotlib.pyplot as plt
import numpy as np
# Define the matchups for each round (each pair is a match)
# For simplicity, the "teams" are their y-positions in this example
rounds = {
    "Round of 64": [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)],
    "Round of 32": [(1.5, 3.5), (5.5, 7.5), (9.5, 11.5), (13.5, 15.5)],
    "Sweet 16": [(2.5, 6.5), (10.5, 14.5)],
    "Elite 8": [(4.5,), (12.5,)],
    "Final 4": [(8.5,)],
}


fig, ax = plt.subplots()

# regions = ['East','West','South','Midwest']
# for region in regions:
# Initial x position
x = 1

# Plot each round
for round, matchups in rounds.items():
    next_x = x + 1
    for match in matchups:
        
        
        # Plot teams/positions
        plt.plot([x, next_x], [match[0], match[0]], marker='o', markersize=5, color='blue')
        # plt.text(x, match[0], f"Team {int(match[0])}", ha='right', va='center')
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
