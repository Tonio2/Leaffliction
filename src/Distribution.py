import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt


def render(fruit_type: str, label: list[str], size: list[int]) -> None:
    """ Renders a pie chart and bar chart """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(fruit_type + " class distribution")
    palette = sns.color_palette("Set2")

    ax1.pie(size, labels=label, colors=palette, autopct='%1.1f%%')

    plt.xticks(rotation=60)
    ax2.bar(label, size, color=palette)

    plt.tight_layout()

    plt.show()


def is_dir(dirname: str, d: str) -> bool:
    """ Checks if the given path is a directory """
    return os.path.isdir(os.path.join(dirname, d))


def main(root_dir: str) -> None:
    """
    Analyzes a root directory containing subdirectories for each fruit type,
    extracts class labels and sizes, and renders visualizations.
    """
    fruit_dictionary = {}
    dirnames = [d for d in os.listdir(root_dir) if is_dir(root_dir, d)]
    for dirname in sorted(dirnames):

        fruit = dirname.split("_")[0]
        if fruit not in fruit_dictionary:
            fruit_dictionary[fruit] = {"label": [], "size": []}
        fruit_dictionary[fruit]["label"].append(dirname)
        size = len(os.listdir(os.path.join(root_dir, dirname)))
        fruit_dictionary[fruit]["size"].append(size)

    for fruit in sorted(fruit_dictionary):
        label = fruit_dictionary[fruit]["label"]
        size = fruit_dictionary[fruit]["size"]
        render(fruit, label, size)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python Distribution.py dirname")
        exit(1)
    try:
        main(sys.argv[1])
    except Exception as e:
        print(f"Error processing directory: {e}")
        exit(1)
