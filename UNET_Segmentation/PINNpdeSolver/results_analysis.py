import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Paths
csv_path = Path(r'C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\DNP-3DDMR-IR\UNET_Segmentation\PINNpdeSolver\results\pinn_fea_results.csv')
RESULTS_DIR = csv_path.parent
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

# Load CSV
if not csv_path.exists():
    print(f"Error: {csv_path} not found!")
    exit(1)

df = pd.read_csv(csv_path)

# Clean failed patients
df_clean = df.dropna(subset=["r_t_mm", "Q_max"]).copy()

# Group data
benign = df_clean[df_clean["label"].str.lower() == "benign"]
malignant = df_clean[df_clean["label"].str.lower() == "malignant"]

print(f"Total clean patients: {len(df_clean)} (Benign: {len(benign)}, Malignant: {len(malignant)})")

# Statistical tests
# Radius comparison
u_stat_r, p_val_r = stats.mannwhitneyu(benign["r_t_mm"], malignant["r_t_mm"], alternative="two-sided")
t_stat_r, t_p_val_r = stats.ttest_ind(benign["r_t_mm"], malignant["r_t_mm"], equal_var=False)

# Volume comparison
u_stat_v, p_val_v = stats.mannwhitneyu(benign["volume_mm3"], malignant["volume_mm3"], alternative="two-sided")
t_stat_v, t_p_val_v = stats.ttest_ind(benign["volume_mm3"], malignant["volume_mm3"], equal_var=False)

# Write summary text
summary_text = f"""==================================================
STATISTICAL ANALYSIS SUMMARY REPORT
==================================================
Total Patients Analyzed: {len(df_clean)}
- Benign: {len(benign)}
- Malignant: {len(malignant)}

--------------------------------------------------
1. TUMOR RADIUS (r_t_mm) COMPARISON:
- Benign Mean: {benign['r_t_mm'].mean():.3f} ± {benign['r_t_mm'].std():.3f} mm
- Malignant Mean: {malignant['r_t_mm'].mean():.3f} ± {malignant['r_t_mm'].std():.3f} mm

Statistical Significance:
- Mann-Whitney U test p-value: {p_val_r:.5f} ({"Significant" if p_val_r < 0.05 else "Not Significant"})
- Welch's T-test p-value: {t_p_val_r:.5f} ({"Significant" if t_p_val_r < 0.05 else "Not Significant"})

--------------------------------------------------
2. TUMOR VOLUME (volume_mm3) COMPARISON:
- Benign Mean: {benign['volume_mm3'].mean():.3f} ± {benign['volume_mm3'].std():.3f} mm³
- Malignant Mean: {malignant['volume_mm3'].mean():.3f} ± {malignant['volume_mm3'].std():.3f} mm³

Statistical Significance:
- Mann-Whitney U test p-value: {p_val_v:.5f} ({"Significant" if p_val_v < 0.05 else "Not Significant"})
- Welch's T-test p-value: {t_p_val_v:.5f} ({"Significant" if t_p_val_v < 0.05 else "Not Significant"})

==================================================
"""
print(summary_text)

with open(PLOTS_DIR / "statistical_summary.txt", "w") as f:
    f.write(summary_text)

# Plotting settings
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14})

# Plot 1: Boxplots of Tumor Radius and Volume side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Radius Boxplot
sns.boxplot(data=df_clean, x="label", y="r_t_mm", palette=["#2ecc71", "#e74c3c"], ax=axes[0])
sns.stripplot(data=df_clean, x="label", y="r_t_mm", color="black", alpha=0.3, jitter=0.2, ax=axes[0])
axes[0].set_title("Reconstructed Tumor Radius ($r_t$)")
axes[0].set_xlabel("Clinical Diagnosis")
axes[0].set_ylabel("Radius (mm)")
axes[0].text(0.5, 0.92, f"Welch's T-test p = {t_p_val_r:.3f}", 
             horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# Volume Boxplot
sns.boxplot(data=df_clean, x="label", y="volume_mm3", palette=["#2ecc71", "#e74c3c"], ax=axes[1])
sns.stripplot(data=df_clean, x="label", y="volume_mm3", color="black", alpha=0.3, jitter=0.2, ax=axes[1])
axes[1].set_title("Reconstructed Tumor Volume ($V_t$)")
axes[1].set_xlabel("Clinical Diagnosis")
axes[1].set_ylabel("Volume ($mm^3$)")
axes[1].text(0.5, 0.92, f"Welch's T-test p = {t_p_val_v:.3f}", 
             horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.tight_layout()
plt.savefig(PLOTS_DIR / "benign_vs_malignant_boxplots.png", dpi=200)
plt.close()

# Plot 2: Scatter plot of Radius vs Volume colored by class
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_clean, x="r_t_mm", y="volume_mm3", hue="label", palette=["#2ecc71", "#e74c3c"], alpha=0.7, s=80)
plt.title("Estimated Tumor Geometric Characteristics")
plt.xlabel("Tumor Radius $r_t$ (mm)")
plt.ylabel("Tumor Volume $V_t$ ($mm^3$)")
plt.legend(title="Diagnosis")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "radius_vs_volume_scatter.png", dpi=200)
plt.close()

# Plot 3: Quadrant Distribution
plt.figure(figsize=(10, 5))
sns.countplot(data=df_clean, x="quadrant", hue="label", palette=["#2ecc71", "#e74c3c"])
plt.title("Tumor Anatomical Quadrant Distribution")
plt.xlabel("Breast Quadrant")
plt.ylabel("Patient Count")
plt.xticks(rotation=15)
plt.legend(title="Diagnosis")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "quadrant_distribution.png", dpi=200)
plt.close()

print(f"All plots and report successfully saved in {PLOTS_DIR}!")
