# Baseline Algorithms

This directory contains the implementation of baseline algorithms for the Cooperative Sweep Coverage (CSC) and Multi-Sink Sweep Coverage (MSSC) problems.

## File Structure

- `algorithm.py`: The core library containing data structures (`Graph`, `Vertex`, `Edge`) and algorithm implementations (`CoCycle`, `OSweep`, `MinExpand`, `PDBA`, `SinkCycle`, `SCOPe_M_Solver`).
- `csc.py`: Experiment script for the CSC problem. It compares four algorithms:
  - CoCycle
  - OSweep
  - MinExpand
  - PDBA
- `mssc.py`: Experiment script for the MSSC problem (Multi-Sink scenarios).

## Usage

### Prerequisites
- Python 3.x
- NumPy

### Running the Experiments

Please run the scripts from the **root directory** of the repository to ensure the data paths are correct.

To run the CSC comparison:

```bash
python code/baseline/csc.py
```

To run the MSSC comparison:

```bash
python code/baseline/mssc.py
```

## Data

The scripts expect the data file to be located at `data/points.csv` relative to the execution directory.