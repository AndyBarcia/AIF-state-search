# State Space Search

This codebase has a script `aif_lab1.py` to both generate random maps and perform search on a given map. 

# Installation

This scripts requires numpy and pydantic. Both can be installed with

```
pip install -r requirements.txt
```

# Usage

For map generation, the script can be used as follows.

```
python3 aif_lab1.py -s 5 5 --save map.txt
```

This generates a 5x5 map to be saved in map.txt. The seed can also be optionally provided.

```
python3 aif_lab1.py -s 5 5 --seed 42 --save map.txt
```

For search, the script can be used as follows.

```
python3 aif_lab1.py -a bfs -f map.txt
```

This performs search on the `map.txt` file using breadth-first-search. The valid options are:

- `bfs`: breadth-first-search.
- `dfs`: depth-first-search.
- `astr`: A*. With this algorithm, optionally either `--h1` and `--h2` can be set to use a certain heuristic function.

Aditionally, both scripts behaviours can be chained together as follows:

```
python3 aif_lab1.py -s 5 5 | python3 aif_lab1.py -a bfs
```

This generates a random 5x5 map, and then performs search on that map.