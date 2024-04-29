import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Define the styles for each method
style_dict = {
    "Random selection": {"color": "blue", "linestyle": "--", "linewidth": 1.5},
    "Active selection": {"color": "orange", "linestyle": "-.", "linewidth": 1.5},
    "Power of Choice": {"color": "cyan", "linestyle": "--", "linewidth": 1.5},
    "Greedy": {"color": "magenta", "linestyle": (0, (3, 5, 1, 5)), "linewidth": 1.5},
    "Resource aware": {"color": "red", "linestyle": ":", "linewidth": 1.5},
    "Price First": {"color": "green", "linestyle": "-.", "linewidth": 1.5},
    "FedPROM": {"color": "purple", "linestyle": "-", "linewidth": 1.5}
}

# Specify the figure size here, width and height in inches
figsize = (10, 2)  # You can adjust the width and height as needed

# Create a figure with the specified size
fig, ax = plt.subplots(figsize=figsize)

# Create a list of Line2D objects using the styles specified in the dictionary
legend_handles = [
    Line2D([0], [0], color=style['color'], linestyle=style['linestyle'], linewidth=style['linewidth']) 
    for style in style_dict.values()
]

# Add the Line2D objects as a legend to the axis
legend = ax.legend(handles=legend_handles, labels=list(style_dict.keys()), loc='center', frameon=False)

# Hide the axis
ax.axis('off')

# Set the layout of the figure to tightly fit the legend
fig.tight_layout()

# Save the legend to a file
file_path = 'legend.png'
plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
plt.close()