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
python Distribution.py <dirname>
```

### Part 2: Data augmentation

To transform 1 specific img:
```
python Augmentation.py <img_path>
```

To create the augmented dir:
```
python Augmentation.py --dir <dirname>
```