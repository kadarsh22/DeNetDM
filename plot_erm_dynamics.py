import matplotlib.pyplot as plt
import numpy as np

# Set font to Times New Roman and font size to 12
plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 12

# Data for the first figure
idx_values = list(range(0, 200, 10))

shape_acc_values_three = [
    0.51, 0.77, 0.79, 0.81, 0.81, 0.82, 0.82, 0.82, 0.81, 0.82,
    0.81, 0.82, 0.83, 0.82, 0.83, 0.82, 0.82, 0.82, 0.83, 0.83
]

color_acc_values_three = [
    0.52, 0.89, 0.95, 0.96, 0.98, 0.99, 0.99, 0.99, 0.99, 0.99,
    0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 1.00
]

shape_acc_values_five = [
    0.24, 0.55, 0.68, 0.71, 0.71, 0.71, 0.70, 0.70, 0.69, 0.70,
    0.70, 0.70, 0.71, 0.71, 0.69, 0.70, 0.69, 0.70, 0.70, 0.70
]

color_acc_values_five = [
    0.24, 0.65, 0.84, 0.94, 0.96, 0.98, 0.98, 0.98, 0.98, 0.98,
    0.98, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99
]

# Convert values to percentages
shape_acc_values_three = [value * 100 for value in shape_acc_values_three]
color_acc_values_three = [value * 100 for value in color_acc_values_three]
shape_acc_values_five = [value * 100 for value in shape_acc_values_five]
color_acc_values_five = [value * 100 for value in color_acc_values_five]

# Plotting for the first figure
plt.figure(figsize=(8, 5))
plt.plot(idx_values, shape_acc_values_three, label='Digit (3 layer MLP)', marker='o', linestyle='-', linewidth=2)
plt.plot(idx_values, color_acc_values_three, label='Color (3 layer MLP)', marker='o', linestyle='-', linewidth=2)
plt.plot(idx_values, shape_acc_values_five, label='Digit (5 layer MLP)', marker='*', linestyle='--', linewidth=2)
plt.plot(idx_values, color_acc_values_five, label='Color (5 layer MLP)', marker='*', linestyle='--', linewidth=2)

plt.title('Early Training Dynamics of MLPs with Varying Depth', fontsize=20)
plt.xlabel('Training Iterations', fontsize=20)
plt.ylabel('Linear Decodability (%)', fontsize=20)
# Enhance legend readability
plt.legend(fontsize=14, loc='lower right')

# Add grid for better readability and adjust alpha for visibility of plot lines
plt.grid(True, linestyle='--', alpha=0.5)

# Use tight_layout to adjust spacing to contain the text properly
plt.tight_layout()
plt.savefig('figures/training_dynamics_erm.pdf', format='pdf', bbox_inches='tight',dpi=720)

# Data for the second figure
network_depth = np.array([3, 4, 5])
linear_decodability_digit = np.array([50.22, 26.61, 10.01])
linear_decodability_color = np.array([54.86, 38.63, 12.01])

# Plotting for the second figure
plt.figure(figsize=(8, 5))
plt.plot(network_depth, linear_decodability_digit, label='Digit', marker='o', linestyle='-', color='b', linewidth=2)
plt.plot(network_depth, linear_decodability_color, label='Color', marker='o', linestyle='-', color='r', linewidth=2)

plt.xlabel('Network Depth', fontsize=20)
plt.ylabel('Linear Decodability (%)', fontsize=20)
plt.title('Linear Decodability of Attributes in an Untrained MLP', fontsize=20)
plt.legend(fontsize=14)
plt.xticks(network_depth.astype(int))  # Casting the ticks to integer type
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('figures/linear_decodability_plot.pdf', format='pdf', bbox_inches='tight', dpi=720)

# Show the plots
# plt.show()
