# NetGP
NetGP: A Hybrid Framework Combining Genetic Programming and Deep Reinforcement Learning for PDE Solutions


# Simulation
* Solve Single PDE
```
python  analyze.py --DATASET_DIR ./PDEdataset/Advection/1_1D/

```

* Solve ALL PDE
```
python analyze.py 
```

# Parameters

| Argument           | Type    | Default Value                  | Description                                                                                          |
|--------------------|---------|-------------------------------|------------------------------------------------------------------------------------------------------|
| `--DATASET_DIR`    | `str`   | `PDEdataset/**/*`             | Dataset directory. You can specify a folder (e.g., `PDEdataset/Advection/1_1D/`) or use wildcards.   |
| `-results`         | `str`   | `resultsv2`                   | Directory to save results. All output files will be stored here.                                      |
| `-ml`              | `str`   | `netgp`                       | Comma-separated list of ML methods to use (should correspond to a `.py` file name in `methods/`).     |
| `-seed`            | `int`   | `1`                           | Random seed for reproducibility.                                                                      |
| `-op`              | `list`  | `["mul", "add", "sub", "sin", "exp"]` | List of operators allowed in symbolic regression. Example: `mul add sub sin exp`            |






