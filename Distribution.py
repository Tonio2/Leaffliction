import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns


def render(fruit_type, label, size):
	fig, (ax1, ax2) = plt.subplots(1, 2)
	fig.suptitle(fruit_type + " class distribution")
	palette = sns.color_palette("Set2")

	ax1.pie(size, labels=label, colors= palette, autopct='%1.1f%%')

	plt.xticks(rotation=60)
	ax2.bar(label, size, color=palette)

	plt.tight_layout()

	plt.show()

def is_dir(dirname, d):
    return os.path.isdir(os.path.join(dirname, d))


def main(dirname):
    # dirs = [d for d in os.listdir(dirname) if is_dir(dirname, d)]
    # dir_size = [len(os.listdir(os.path.join(dirname, d))) for d in dirs]

	fruit_dictionary = {}

	for root, dirs, files in os.walk(dirname):
		for dir in sorted(dirs):
			if "_" in dir:
				dir_path = os.path.join(root, dir)
				files_in_dir = [item for item in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, item))]
			
				if files_in_dir:
					fruit = dir.split("_")[0]
					if fruit not in fruit_dictionary:
						fruit_dictionary[fruit] = {"label": [], "size": []}
					fruit_dictionary[fruit]["label"].append(dir)
					fruit_dictionary[fruit]["size"].append(len(files_in_dir))

	for fruit in sorted(fruit_dictionary):
		render(fruit, fruit_dictionary[fruit]["label"], fruit_dictionary[fruit]["size"])


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python Distribution.py dirname")
        exit(1)
    main(sys.argv[1])
