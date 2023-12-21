# Information-Theoretic Safe Exploration with Gaussian Processes 

PyTorch implementation of the NeurIPS 2022 paper "Information-Theoretic Safe Exploration with Gaussian Processes". 
The paper can be found [here](https://openreview.net/pdf?id=cV03Zw0V-3J). The code allows the users to use our 
implementation of the `ISE` acquisition function, together with others used in the paper experiment section.

**NOTE**: Code for the extension paper "Information-Theoretic Safe Bayesian Optimization" will soon be available in this repo as well.

## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication. It will neither be
maintained nor monitored in any way.

## Setup.

1. Clone the repository and `cd` into it
2. Create a conda environment `ise_exploration` with needed dependencies
```bash
conda env create --file=environment.yaml
```
3. Activate the environment 
4. Install (locally)
 ```bash
conda activate ise_exploration
pip install -e .
```

## Run example

An example is provided in `ise/example.py`. To run it, simply activate the environment where `ISE` has been installed and
execute:

```bash
python ise/example.py
```

## License

Information-Theoretic Safe Exploration is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

