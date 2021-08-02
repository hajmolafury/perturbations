# Perturbations

:exclamation::exclamation: **NOTE**: Here is the [**link**](https://github.com/silverpaths/nodepert) to our new project repository, which is much better documented and contains more comprehensive experiments. The best part is, it's written in **JAX** so it's also much easier to follow!

This repository contains a set of experiments written in Tensorflow (version 1.x) to explore the node perturbation algorithm for training deep fully connected neural networks.


## Usage

Run master.py to train a multi-layer perceptron (with sgd or np).

```python
python master.py -lr 0.1 -update_rule np -n_hl 3 -hl_size 300 -n_epochs 5
```

All are optional arguments:

lr: learning rate

update_rule: either 'np' or 'sgd'

network will have (n_hl) depth and constant width of size (hl_size)
