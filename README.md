# Zero-Shot Domain Adaptation in CT Segmentation by Filtered Back Projection Augmentation
Code release
## Install

```
git clone https://github.com/STNLd2/FBPAug.git
./FBPAug/install.sh
```
Install [surface-distance](https://github.com/deepmind/surface-distance) as well to compute surface metrics.

## Experiment Reproduction
To run a single experiment please follow the steps below:

First, the experiment structure must be created:
```
python -m dpipe build_experiment --config_path "$1" --experiment_path "$2"
```

where the first argument is a path to the `.config` file e.g., 
`"~/miccai_paper/config/expepiments/baseline.config"`
and the second argument is a path to the folder where the experiment
structure will be organized, e.g.
`"~/miccai_paper/baseline"`.

Then, to run an experiment please go to the experiment folder inside the created structure:
```
cd ~/miccai_paper/baseline
```
and call the following command to start the experiment:
```
python -m dpipe run_experiment --config_path "../resources.config"
```
where `resources.config` is the general `.config` file of the experiment.
