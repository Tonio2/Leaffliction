import os
import sys
import matplotlib.pyplot as plt


def is_dir(dirname, d):
    return os.path.isdir(os.path.join(dirname, d))


def main(dirname):
    dirs = [d for d in os.listdir(dirname) if is_dir(dirname, d)]
    dir_size = [len(os.listdir(os.path.join(dirname, d))) for d in dirs]

    fig, axs = plt.subplots(1, 2)
    wedges, texts, autotexts = axs[0].pie(dir_size, autopct="%1.1f%%")
    pie_colors = [w.get_facecolor() for w in wedges]
    axs[1].bar(dirs, dir_size, color=pie_colors)
    axs[1].tick_params(axis='x', rotation=45)

    fig.suptitle("apple class distribution")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python Distribution.py dirname")
        exit(1)
    main(sys.argv[1])
