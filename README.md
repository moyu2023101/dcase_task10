# Acoustic Traffic Simulation and Counting

## Purpose of the project

This project contains the source code for the system we submitted for Task 10 in the DCASE2024 Challenge.

## Getting started

To set up the code, you have to do the following:

1.Set up and activate the Python environment:
```
conda create -n traffic python=3.10
conda activate traffic
pip install poetry
poetry install
```
2.Download the dataset from [Zenodo](https://zenodo.org/records/10700792) and decompress `locX.zip` in `real_root/` and `engine-sounds.zip` in `engine_sound_dir/`, respectively.

3.Train and evaluate on the development set.
```
bash pretrain.sh <locX> <gpu_id>
```
### Reproducing the Submitted Results
4.Test on the evaluation set to obtain the test results. Download the evaluation set and extract it to the corresponding locations under the `real_root/` directory by site.

```
poetry run python -m atsc.counting.inference site=<locX> inference.alias=finetune
```

The output of this step is saved at `<work_folder>/counting`

5.Additionally, the modifications needed for our three submitted systems are as follows:
```angular2html
System-1: "atsc/counting/models/baseline.py" -lines_246:self.type=="panns_gat"
System-2: "atsc/counting/models/baseline.py" -lines_246:self.type=="panns_gat" -lines_301:self.encoder = Cnn10(spec_aug=True)
System-3: "atsc/counting/models/baseline.py" -lines_246:self.type=="phase_panns_gat" -lines_301:self.encoder = Cnn10(spec_aug=True)
```
