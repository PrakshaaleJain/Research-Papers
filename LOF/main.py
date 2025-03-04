import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor


df = pd.read_csv("lof_dataset.csv") # Synthetic Dataset created using Chagpt
X = df[["X", "Y"]].values

# LOF 
lof = LocalOutlierFactor(n_neighbors=20)  # MinPts = 20
outlier_labels = lof.fit_predict(X)  # -1 for outliers
lof_scores = -lof.negative_outlier_factor_  # Higher = more likely outlier

# Adding columns of outliers and LOF score in the df
df["LOF_Score"] = lof_scores
df["Outlier"] = outlier_labels

# Printing the top 7 outlier with the most LOF score
top_outliers = df.nlargest(7, "LOF_Score")
print("Top 7 Outliers:\n", top_outliers)

# SCatter Plotting the data-points
plt.figure(figsize=(8, 6))
plt.scatter(df["X"], df["Y"], c=df["Outlier"], cmap="coolwarm", edgecolors="k")
plt.colorbar(label="Outlier Score (LOF)")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("LOF Outlier Detection")
plt.savefig("lof_plot.png") 
print("Plot Saved")


