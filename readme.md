# I-Repair

This repo contains the I-Repair code for localised repairs of Neural Networks. 

I-repair is used to repair a set of misclassified inputs by iteratively rescaling 
weights connected to individual nodes. In order to determine good repair-node 
candidates, I-Repair requires a small set of correctly classified inputs which is used
to heuristically identify nodes that have a large impact on the repair set and a small 
impact on the correctly classified set.

## Installation

### Pipenv

All dependencies can be installed via pipenv:

```
$ cd <your_repair_tool_path>/
$ pipenv install
```

## Usage

The experiments reported in the paper can be run with the scripts in the
repair_tool/evaluation/benchmark_scripts/ folder; the results are stored in 
repair_tool/benchmark_results.

Remember to enable the pipenv environment before running the library with:

cd <your_path>/repair_tool/
pipenv shell

## Authors

Patrick Henriksen: patrick.henriksen18@imperial.ac.uk  
Francesco Leofante: f.leofante@imperial.ac.uk  
Alessio Lomuscio
