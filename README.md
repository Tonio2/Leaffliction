# Leaffliction

## Install

```
python3 -m virtualenv venv
```

```
. venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Part 1: Analysis of the Data Set

```
curl https://cdn.intra.42.fr/document/document/17547/leaves.zip --output ~/goinfre/file
unzip ~/goinfre/file -d ~/goinfre
python Distribution.py <dirname>
```

### Part 2: Data augmentation

You can change which augmentations to use in the code

To transform 1 specific img:
```
python Augmentation.py <img_path>
```

To create the augmented dir:
```
python Augmentation.py --dir <dirname>
```