import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Example dataset name
dataset = "logos"

# Load the data dictionary
with open("results/{dataset}_plotting_data.json", "r") as f:
    data_dict = json.load(f)

if dataset == "logos":
    num_shapes = 10
    num_textures = 11
elif dataset == "sin":
    num_shapes = 15
    num_textures = 11
elif dataset == "icons":
    num_shapes = 6
    num_textures = 11
elif dataset == "messages":
    num_shapes = 11
    num_textures = 5
else:
    raise ValueError(f"Dataset '{dataset}' not recognised.")


# Define the contexts, tasks, models, and colors
contexts = ["neither", "shape", "background", "both"]
tasks = ["shape", "texture", "both"]
models = [
    "idefics-9b-instruct",
    "llava16-7b",
    "mmicl-t5-xxl",
    "otter-mpt",
    "qwen-vl-chat",
]
context_map = {
    "neither": "ICL1 (Icl. Neither)",
    "shape": "ICL2 (Icl. Shape)",
    "background": "ICL3 (Icl. Texture)",
    "both": "ICL4 (Icl. Both)",
}
colors = sns.color_palette("husl", len(models))

# Assume the number of shapes and backgrounds are provided

# Create a 3x4 grid for the subplots with shared y-axis
fig, axs = plt.subplots(3, 4, figsize=(20, 15), sharey=False)
fig.suptitle(
    f"ICL shape and texture accuracy for {dataset} dataset.",
    fontsize=16,
    x=0.54,
    y=0.95,
    weight="bold",
)

# Set up row labels for prediction tasks
for i, task in enumerate(tasks):
    fig.text(
        1.005,
        0.77 - i * 0.275,
        "Predict " + task.capitalize(),
        va="center",
        ha="center",
        rotation=270,
        fontsize=14,
        weight="bold",
    )

    # Add text annotations for y-axis labels
    fig.text(
        0.05,
        0.82 - i * 0.275,
        "Shape Acc.",
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=12,
    )
    fig.text(
        0.05,
        0.71 - i * 0.275,
        "Texture Acc.",
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=12,
    )

for i, task in enumerate(tasks):
    for j, context_criteria in enumerate(contexts):
        ax = axs[i, j]

        # Check if the task and context criteria exist in the data dictionary
        if task not in data_dict or context_criteria not in data_dict[task]:
            if i == 0:
                ax.set_title(f"{context_map[context_criteria]}")
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        valid_models = [
            model for model in models if model in data_dict[task][context_criteria]
        ]
        if not valid_models:
            if i == 0:
                ax.set_title(f"{context_criteria.capitalize()}")
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        shots = sorted(data_dict[task][context_criteria][valid_models[0]].keys())
        if not shots:
            if i == 0:
                ax.set_title(f"{context_criteria.capitalize()}")
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        print(f"Plotting {task} - {context_criteria}: Shots {shots}")

        bar_width = 0.15  # Width of each bar
        spacing = bar_width * (len(models) + 1)  # Space between different shot groups
        x = np.arange(len(shots)) * (
            spacing + bar_width
        )  # X positions for each shot group

        for idx, model in enumerate(models):
            if model not in data_dict[task][context_criteria]:
                continue

            model_data = data_dict[task][context_criteria][model]
            shape_acc = [model_data[shot].get("shape_accuracy", 0) for shot in shots]
            background_acc = [
                model_data[shot].get("texture_accuracy", 0) for shot in shots
            ]

            print(f"{model} - {context_criteria} Shape Acc: {shape_acc}")
            print(f"{model} - {context_criteria} Texture Acc: {background_acc}")

            # Bar positions for shape and background accuracies
            positions = x + idx * bar_width

            # Plot shape accuracy (positive bars)
            ax.bar(
                positions,
                shape_acc,
                bar_width,
                label=f"{model} - Shape Acc." if j == 0 and i == 0 else "",
                color=colors[idx],
                edgecolor="black",
            )
            # Plot background accuracy (negative bars) with more hatching
            ax.bar(
                positions,
                [-val for val in background_acc],
                bar_width,
                label=f"{model} - Texture Acc." if j == 0 and i == 0 else "",
                color=colors[idx],
                edgecolor="black",
                hatch="///",
            )

        if i == 0:
            ax.set_title(f"{context_map[context_criteria]}", fontsize=14, weight="bold")

        # Set y-ticks and convert to positive values for labels
        y_ticks = np.arange(-1.0, 1.05, 0.05)
        y_tick_labels = [
            f"{abs(tick):.2f}" if (idx % 4 == 0) else ""
            for idx, tick in enumerate(y_ticks)
        ]
        print(f"Y-ticks: {y_ticks}")
        print(f"Y-tick labels: {y_tick_labels}")
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)

        # Make the x-axis line bolder
        ax.axhline(0, color="black", linewidth=1.6)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

        # Add random baseline lines for shape and texture
        ax.axhline(
            random_shape_baseline,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Random Shape Baseline" if j == 0 and i == 0 else "",
        )
        ax.axhline(
            -random_texture_baseline,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Random Texture Baseline" if j == 0 and i == 0 else "",
        )

        # Set x-ticks in the middle of the groups of models
        ax.set_xticks(x + bar_width * (len(models) / 2 - 0.5))
        ax.set_xticklabels(shots)

# Super x-label
fig.supxlabel("Number of Shots", fontsize=14, y=0.06, x=0.54)

# Only add the legend once, outside the loop
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(
    handles[: len(models) * 2 + 2],
    labels[: len(models) * 2 + 2],
    loc="lower center",
    ncol=len(models) + 2,
)  # Adjust ncol as needed
plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
plt.show()
