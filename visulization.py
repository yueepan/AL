import matplotlib.pyplot as plt

# Updated percentages for the subsets
percentages = [10, 20, 30, 40]

# Values provided by the user
strategies = {
    "Random": [0.32, 0.39, 0.41, 0.43],
    "Diversity": [0.36, 0.40, 0.43, 0.44],
    "Uncertainty": [0.32, 0.34, 0.39, 0.41],
    "Hybrid1": [0.36, 0.36, 0.40, 0.43],
    "Hybrid2": [0.36, 0.36, 0.39, 0.42]
}

# Set up the plot with distinctive colors
plt.figure(figsize=(10, 6))

# Define a color palette
color_palette = plt.cm.tab10.colors  # Using a color palette for better distinction

# Plot each strategy with a different color from the palette
for i, (strategy, values) in enumerate(strategies.items()):
    plt.plot(percentages, values, '-o', label=strategy, color=color_palette[i % len(color_palette)])

# Formatting the plot according to APA style
plt.title('Figure 2: mAP50 Scores at Various Percentages of the Dataset', fontsize=14)
plt.xlabel('Percentage of the Dataset (%)', fontsize=12)
plt.ylabel('mAP50 Score', fontsize=12)
plt.xticks(percentages)  # Set x-ticks to match the percentages of the dataset used
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()  # Adjust layout to fit all elements
plt.show()
