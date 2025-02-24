import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create the figure and axis
fig, ax = plt.subplots(figsize=(6, 6))

# δ (delta) - positioned externally to α (Externally Connected)
delta = patches.Rectangle((-2, 2), 2, 2, edgecolor='blue', facecolor='none', lw=2, label=r'$\delta$')

# α (alpha) - inside β but touching its boundary (TPPi relation)
alpha = patches.Rectangle((1, 1.5), 2.5, 3, edgecolor='red', facecolor='none', lw=2, label=r'$\alpha$')

# β (beta) - containing α (TPPi relation)
beta = patches.Rectangle((0, 1), 5, 4, edgecolor='green', facecolor='none', lw=2, label=r'$\beta$')

# Add the rectangles to the plot
ax.add_patch(delta)
ax.add_patch(beta)
ax.add_patch(alpha)

# Labels
ax.text(-1, 3, r'$\delta$', fontsize=14, color='blue', verticalalignment='center', horizontalalignment='center')
ax.text(2.25, 3, r'$\alpha$', fontsize=14, color='red', verticalalignment='center', horizontalalignment='center')
ax.text(3.5, 3, r'$\beta$', fontsize=14, color='green', verticalalignment='center', horizontalalignment='center')

# Set limits and remove grid lines
ax.set_xlim(-3, 6)
ax.set_ylim(0, 6)
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Title
ax.set_title(r'$\delta \text{ EC } \alpha, \alpha \text{ TPPi } \beta \Rightarrow \delta \in \{DC, EC\} \beta$', fontsize=12)

# Show the plot
plt.show()
