# Performance Pay Project




## Directory Structure

### data

Contains cleaned data sets, as well as some derived values.

### old

Previous code

### test

unit tests

### bin

scripts

# Notes
Currently dynamics specification from `[Name].yaml` is not supported. The simplified model is always used.

# Miscellaneous

## IPOPT Solver

The return codes for IPOPT solve status are described [here](https://github.com/coin-or/Ipopt/blob/master/src/Interfaces/IpReturnCodes_inc.h).

## Pytorch-LBFGS

We use the implementation found [here](https://github.com/hjmshi/PyTorch-LBFGS) and vendored under `third_party`.