# Zero-Shot Domain Adaptation in CT Segmentation by Filtered Back Projection Augmentation
Code release
## Install

```
git clone https://github.com/STNLd2/FBPAug.git
./FBPAug/install.sh
```
Install [surface-distance](https://github.com/deepmind/surface-distance) as well to compute surface metrics.

## Example of using
```python
from fbp_aug import apply_conv_filter
sharper_ct_image = apply_conv_filter(ct_image, a=30, b=3)

```
![example of using](https://https://github.com/STNLd2/FBPAug/blob/master/pics/example_of_transform.png?raw=true)

## Experiment Reproduction
To run a single experiment please follow the steps below:

First, the experiment structure must be created:
```
python -m dpipe build_experiment --config_path "$1" --experiment_path "$2"
```

where the first argument is a path to the `.config` file e.g., 
`"~/FBPAug/configs/expepiments/baseline.config"`
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


