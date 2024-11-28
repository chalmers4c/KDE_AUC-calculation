import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import integrate

#%% Function to extract peak positions and plot histograms/KDE
def extract_peak_positions_and_plot(data_dict, bins, kde_bw, hist_color, kde_color, grid_shape=(8, 5)):
    hist_data = {}
    kde_data = {}

    fig, axes = plt.subplots(*grid_shape, figsize=(18, 8), sharex=True)
    axes = axes.flatten()

    for i, (key, df) in enumerate(data_dict.items()):
        ax = axes[i]
        
        # Plot histogram
        sns.histplot(
            data=df, x="Peak Amplitude (nA)", bins=bins, color=hist_color, alpha=0.7, ax=ax
        )
        ax.set(xlabel="Peak Amplitude (nA)", ylabel="Frequency", title=key, xlim=(0, 1.0))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Add KDE on secondary axis
        kde_ax = ax.twinx()
        sns.kdeplot(
            data=df,
            x="Peak Amplitude (nA)",
            bw_method=kde_bw,
            color=kde_color,
            ax=kde_ax,
        )
        kde_ax.set_ylabel("")
        kde_ax.spines["right"].set_visible(False)
        kde_ax.spines["top"].set_visible(False)
        kde_ax.set_yticks([])

        # Save histogram and KDE data
        hist_data[key] = pd.DataFrame({
            "Peak Amplitude (nA)": [patch.get_x() for patch in ax.patches],
            "Frequency": [patch.get_height() for patch in ax.patches],
        })
        kde_line = kde_ax.get_lines()[0]
        kde_data[key] = pd.DataFrame({
            "Peak Amplitude (nA)": kde_line.get_xdata(),
            "KDE": kde_line.get_ydata(),
        })

    # Hide unused subplots
    for i in range(len(data_dict), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    return hist_data, kde_data

#%% Extract highest peak amplitude from histograms
def extract_highest_peak(hist_data):
    return pd.DataFrame([
        {"dataframe": name, "highest value": df["Peak Amplitude (nA)"].iloc[df["Frequency"].idxmax()]}
        for name, df in hist_data.items()
    ])

#%% Calculate margins around peak position
def calculate_margins(peak_value, margin_percent):
    margin_ratio = margin_percent / 100
    return peak_value * (1 + margin_ratio), peak_value * (1 - margin_ratio)

#%% alternative way to calculate the margins around peak position
def calculate_margins_std(peak_value, std_dev, num_std=1):
    margin_above = peak_value + (num_std * std_dev)
    margin_below = peak_value - (num_std * std_dev)
    return margin_above, margin_below



#%% Plot KDE lines with margin
def plot_kde_with_margin(kde_data, margin_above, margin_below, kde_bw, kde_color, grid_shape=(8, 5)):
    fig, axes = plt.subplots(*grid_shape, figsize=(18, 8), sharex=True)
    axes = axes.flatten()

    auc_data = []

    for i, (key, df) in enumerate(kde_data.items()):
        ax = axes[i]

        # Plot KDE
        sns.kdeplot(
            data=df,
            x="Peak Amplitude (nA)",
            bw_method=kde_bw,
            color=kde_color,
            ax=ax,
        )
        ax.axvline(margin_above, color="red", linestyle="--")
        ax.axvline(margin_below, color="green", linestyle="--")
        ax.set(xlabel="Peak Amplitude (nA)", ylabel="KDE", title=key, xlim=(0, 1.0))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Calculate area under curve (AUC) within margins
        x, y = df["Peak Amplitude (nA)"], df["KDE"]
        auc = integrate.trapz(y[(x >= margin_below) & (x <= margin_above)], x[(x >= margin_below) & (x <= margin_above)])
        auc_data.append({"dataframe": key, "Probability": auc * 100})

    # Hide unused subplots
    for i in range(len(kde_data), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
    return pd.DataFrame(auc_data)

#%% Scatter plot for AUC over time
def plot_time_probability(data, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(7, 7))

    sns.scatterplot(
        x=data["Time (min)"],
        y=data["Probability"],
        hue=data["Time (min)"],
        palette=sns.color_palette(cmap, len(data)),
        s=200,
        edgecolor="black",
        ax=ax,
    )

    ax.legend().remove()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set(xlabel="Time (min)", ylabel="Probability (%)", ylim=(0, 100))
    plt.tight_layout()
    plt.show()

