import matplotlib.pyplot as plt
import numpy as np
import itertools
from matplotlib.colors import LinearSegmentedColormap


# Helper to plot training curves
def plot_curve(train_vals, val_vals, title, subject_id, save_path):
    plt.figure()
    plt.plot(train_vals, label=f"Train {title}")
    plt.plot(val_vals, label=f"Val {title}")
    plt.title(f"Subject {subject_id} - {title} vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


# Helper to plot confusion matrix
def plot_confusion_matrix(
        cm,
        save_path,
        class_names,
        title="Confusion Matrix",
        *,
        # ── visual tuning knobs ───────────────────────────────
        cmap = LinearSegmentedColormap.from_list("Blues", 
            ["#E8F5FC", "#99DBFA", "#4693CA"], N=256 # ["#FCF7E8", "#B2E3FA", "#4693CA"]
        ),
        font_sizes=None,       # dict → {"title": 14, "tick": 12, "label": 12, "cell": 10}
        font_colors=None,      # dict → {"title": "black", "tick": "black", "label": "black"}
        cell_text_colors=("black", "black") # ("white", "black"), high-value, low-value cell text colours
):
    """
    Draws and saves a confusion-matrix heat-map with fully controllable
    font sizes/colours and colour-map.

    Parameters
    ----------
    cm : 2-D array (int or float)
        Confusion-matrix counts or averaged scores.
    save_path : str
        File path (incl. extension) where the figure is saved.
    class_names : list[str]
        Axis tick labels.
    title : str, optional
        Figure title.
    cmap : str or mpl Colormap, optional
        Colour-map identifier accepted by Matplotlib.
    font_sizes : dict[str, int], optional
        Keys: "title", "tick", "label", "cell".
    font_colors : dict[str, str], optional
        Keys: "title", "tick", "label".
    cell_text_colors : tuple[str, str], optional
        Colours for cell text when the cell value is above vs. below half
        the matrix maximum (use a single string for a fixed colour).
    """

    # ── sensible defaults ───────────────────────────────────
    default_fs = {"title": 16, "tick": 13, "label": 16, "cell": 16}
    default_fc = {"title": "black", "tick": "black", "label": "black"}

    if font_sizes is not None:
        default_fs.update(font_sizes)
    if font_colors is not None:
        default_fc.update(font_colors)

    high_col, low_col = (cell_text_colors if isinstance(cell_text_colors, tuple)
                         else (cell_text_colors, cell_text_colors))

    # ── plotting ────────────────────────────────────────────
    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, interpolation="nearest", cmap=cmap)
    ax = plt.gca()
    ax.grid(False)                         # turn off any grid
    # for spine in ax.spines.values():      # hide all 4 spines
    #     spine.set_visible(False)

    plt.title(title, fontsize=default_fs["title"], color=default_fc["title"])
    plt.colorbar(im, fraction=.046, pad=.04)

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=default_fs["tick"], color=default_fc["tick"])
    plt.yticks(tick_marks, class_names, fontsize=default_fs["tick"], color=default_fc["tick"])

    plt.ylabel("True label", fontsize=default_fs["label"], color=default_fc["label"])
    plt.xlabel("Predicted label", fontsize=default_fs["label"], color=default_fc["label"])

    fmt = ".1f" if issubclass(cm.dtype.type, np.floating) else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i,
                 format(cm[i, j], fmt),
                 ha="center", va="center",
                 fontsize=default_fs["cell"],
                 color=high_col if cm[i, j] > thresh else low_col)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
