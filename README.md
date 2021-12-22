# TOM

This repository contains an implementation of the model introduced in [The Traveling Observer Model: Multi-task Learning Through Spatial Variable Embeddings](https://arxiv.org/abs/2010.02354), published with a spotlight at ICLR 2021.

The model is defined at `traveling_observer_model.py`.

A script is also provided, demonstrating how the implementation can be applied:
```
python train_concentric_hyperspheres.py
```

The script is applied to the Concentric Hyperspheres problem, where TOM has shown the most striking advantage over alternative methods.
