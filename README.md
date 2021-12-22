# TOM

This repository contains an implementation of the model introduced in [The Traveling Observer Model: Multi-task Learning Through Spatial Variable Embeddings](https://arxiv.org/abs/2010.02354), published with a spotlight presentation at ICLR 2021. Spotlight presentation: https://iclr.cc/virtual/2021/poster/3007.

The model is defined at `traveling_observer_model.py`.

A script is also provided, demonstrating how the implementation can be applied:
```
python train_concentric_hyperspheres.py
```

The script is applied to the Concentric Hyperspheres problem, where TOM has shown the most striking advantage over alternative methods.

This code was developed with Python version 3.7.12. See `requirements.txt` for the full python environment, and run `pip install -r requirements.txt` to install.
