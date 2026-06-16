# Causality between Principal Components in Protein MD trajectories
This repository contains the codes and scripts required to reproduce the results in....


## Workflow:

Run the scripts in order of numbering. 
The causality calculations are done here with a minimal script to calculate the Information Imbalance (II) and Imbalance Gain (IG) in `imbalance_gain_windows.py` in each directory, but they can also be done using the implementations in [DADApy](https://github.com/sissa-data-science/DADApy).

Each directory has its own sets of scripts and data. The causality data is stored in `iib_data` sub-directories for each system.

The PCs are computed using `sklearn` from the trajectories.

1. Compute the causality between all pairs of the top 5 PCs using the `01*` series of scripts. The ones labelled with `-PC1` or `-PC2` instead compute the causality between PC1/PC2 and all the top 20 PCs.
2. The `Analysis*.ipynb` notebooks are used to analyze the outputs of these calculations.
3. The `02*` series of scripts are to make smoothened IG plots for the paper.
4. The `03` script is to integrate the IG curves.
5. The `04` script is to make a very basic causal graph.
6. The `05` script is to compute the Bruschweiller collectivity index to determine which residues contribute most to each PC.

## Citation:

Please cite: https://arxiv.org/abs/2605.08381

```


```

