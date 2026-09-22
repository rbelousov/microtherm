# Microtherm

This repository contains code, examples, and supporting information for the microcaliber theory of thermodiffusion.

## Installation

Get the source code and install using pip, e.g.

```bash
git clone https://git.embl.de/rbelouso/microcaliber.git microcaliber && cd microcaliber
pip install .[examples]
```

To install with test dependencies, replace the last of the above commands with `pip install .[tests]`.

## Supplemental materials

Supplemental materials are deposited here as [sm.pdf](/sm.pdf).


## Usage

Examples of simulations are provided in the directory `examples`, containing a 
collection Jupyter notebooks. These notebooks require optional dependencies from the `examples` group—make sure to install them with `pip install .[examples]`.


## Authors and acknowledgment

Please cite our preprint R. Belousov, J. Elliott, A. Erzberger, Ensemble theory of thermodiffusion, [arXiv:2607.27839](https://doi.org/10.48550/arXiv.2607.27839): 

```bibtex
@misc{belousov2026microtherm,
  doi = {10.48550/ARXIV.2607.27839},
  url = {https://arxiv.org/abs/2607.27839},
  author = {Belousov,  R. and Elliott,  J. and Erzberger,  A.},
  keywords = {Statistical Mechanics (cond-mat.stat-mech),  FOS: Physical sciences},
  title = {Ensemble theory of thermodiffusion},
  publisher = {arXiv},
  year = {2026},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## License

The code is distributed under MIT license (see[LICENSE](https://git.embl.de/rbelouso/microtherm/LICENSE)).
