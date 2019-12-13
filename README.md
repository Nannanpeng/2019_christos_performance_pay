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

## Warnings

*Random state is not managed properly*: complete runs are deterministic, but if you want to resume from a checkpoint, one would need the random state at the time the checkpoint was saved. Currently we don't save that information.