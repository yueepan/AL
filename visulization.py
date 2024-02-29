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



import matplotlib.pyplot as plt

# Classes and corresponding mAP50 scores for each strategy
classes = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
random_scores = [0.487, 0.32, 0.167, 0.81, 0.497, 0.452, 0.264, 0.212, 0.615, 0.444]
diversity_scores = [0.498, 0.336, 0.168, 0.816, 0.503, 0.473, 0.245, 0.209, 0.63, 0.49]
uncertainty_scores = [0.494, 0.326, 0.159, 0.795, 0.47, 0.484, 0.257, 0.165, 0.546, 0.446]
hybrid1_scores = [0.512, 0.323, 0.166, 0.809, 0.49, 0.464, 0.291, 0.133, 0.584, 0.481]
hybrid2_scores = [0.478, 0.318, 0.142, 0.809, 0.477, 0.439, 0.283, 0.178, 0.57, 0.464]

# Plotting
plt.figure(figsize=[12, 6])
plt.plot(classes, random_scores, label='Random', marker='o')
plt.plot(classes, diversity_scores, label='Diversity', marker='o')
plt.plot(classes, uncertainty_scores, label='Uncertainty', marker='o')
plt.plot(classes, hybrid1_scores, label='Hybrid1', marker='o')
plt.plot(classes, hybrid2_scores, label='Hybrid2', marker='o')

# Customizing the plot
# plt.title('Comparison of Active Learning Strategies by Class')
plt.xlabel('Class')
plt.ylabel('mAP50 Score')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

# Save the plot as a file
plt.tight_layout()

plt.tight_layout()  # Adjust layout to fit all elements
plt.savefig('mAP50byclass.png')  # Save the plot as an image file

