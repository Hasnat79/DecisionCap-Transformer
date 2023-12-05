import json
import matplotlib.pyplot as plt
import pandas as pd

with open("./log_lite.json",'r') as f:
    log_lite_data = json.load(f)

df = pd.DataFrame(log_lite_data)

# Plot the training/train_loss_mean
plt.figure(figsize=(10, 6))
plt.plot(df["training/train_loss_mean"], label="Train Loss Mean")
plt.title("Training Loss Mean Over Time")
plt.xlabel("Episodes")
plt.ylabel("Train Loss Mean")
plt.legend()
plt.grid(True)

# Save the figure with an appropriate name
plt.savefig("./train_loss_mean_plot.png")

# Show the plot if needed
plt.show()
