import matplotlib.pyplot as plt

# Data
debias_shape_acc = [48.38, 81.91, 83.70, 83.97, 83.57, 84.41, 84.66, 84.42, 84.28, 84.16, 83.71, 83.18, 84.51, 84.67, 84.57, 84.14, 85.04, 85.23, 84.23, 84.80, 83.91]
debias_color_acc = [50.77, 98.42, 97.10, 95.66, 94.16, 93.79, 94.17, 92.42, 89.29, 90.29, 87.56, 83.43, 79.10, 76.36, 75.70, 72.35, 70.97, 68.17, 70.22, 68.01, 64.41]
bias_shape_acc = [10.37, 65.84, 65.47, 63.21, 59.20, 58.90, 55.56, 54.75, 50.78, 48.21, 49.64, 44.61, 45.04, 45.95, 41.63, 41.96, 37.92, 38.87, 40.06, 37.12, 38.06]
bias_color_acc = [22.57, 98.21, 99.28, 98.49, 99.25, 99.01, 99.20, 99.60, 98.54, 99.39, 99.44, 99.14, 98.74, 99.29, 99.59, 99.45, 99.88, 99.18, 99.38, 99.64, 99.13]

# Use Times New Roman font
plt.rcParams['font.family'] = 'Times New Roman'

# Plot settings for beautification
plt.figure(figsize=(10, 6))
plt.plot(debias_shape_acc, label='debias shape_acc', marker='o', linestyle='-')
plt.plot(debias_color_acc, label='debias color_acc', marker='s', linestyle='--')
plt.plot(bias_shape_acc, label='bias shape_acc', marker='D', linestyle='-')
plt.plot(bias_color_acc, label='bias color_acc', marker='^', linestyle='--')
plt.xlabel('Training iteration', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.title('Training Dynamics of DeCaM', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
