# Layer Ensemble Averaging for Fault Tolerance in Memristive Neural Networks

This repository contains code & data for the following paper:

```
@article{yousuf2024layer,
  title={Layer Ensemble Averaging for Improving Memristor-Based Artificial Neural Network Performance},
  author={Yousuf, Osama and Hoskins, Brian and Ramu, Karthick and Fream, Mitchell and Borders, William A and Madhavan, Advait and Daniels, Matthew W and Dienstfrey, Andrew and McClelland, Jabez J and Lueker-Boden, Martin and Adam, Gina C},
  journal={arXiv preprint arXiv:2404.15621},
  year={2024}
}
```



## Installation

- Clone this repository and navigate inside:

```
git clone https://github.com/ADAM-Lab-GW/layer-ensemble-averaging &&
cd layer-ensemble-averaging
```

- Set up a virtual environment and install dependencies:
```
python3 -m venv env &&
source env/bin/activate &&
pip install -r requirements_full.txt
```

If only plotting scripts need to be run, `requirements_plots.txt` can be used instead of `requirements_full.txt`.

**NOTE:** The scripts have been tested with Python 3.8.10 and Ubuntu 20.04.6 LTS. Minor changes to packages may be required for other Python versions or operating systems.

## Repository Structure

| Directory    | Description |
| -------- | ------- |
| `data`  | Source datasets used for figures in the main text. Sub-directories are grouped by figure and panel numbers.     |
| `plots` | Code for generating figures as presented in the main text.     |
| `generated_plots`    | Output directory where all plots are stored by default. We provide plots for viewing purposes as part of the repository, so that the plotting code does not have to be re-run.    |
| `generated_plots`    | Output directory where all plots are stored by default. We provide plots for viewing purposes as part of the repository, so that the plotting code does not have to be re-run.    |

## Usage


### Generate Plots

Assuming a virtual environment is set up, run the provided `generate_plots.sh` bash script.

```
chmod +x ./generate_plots.sh &&
./generate_plots.sh
```

### Network Training

We also provide code to train the neural network on the multi-task variant of the Yin-Yang dataset, as presented in the paper. This step uses PyTorch, which can be installed by setting up the virtual environment using `requirements_full.txt`.

- To train without using elastic weight consolidation, use the command below. This will train a network that clearly forgets Task 1 after being trained on Task 2. 

```
python3 network/train_multitask_ewc.py --no-ewc
```

- To train with elastic weight consolidation, use the command below. This will train a network that retains its performance on Task 1 even after being trained on Task 2.

```
python3 network/train_multitask_ewc.py
```

**NOTE:** Navigate to directories for more instructions on individual files. 

## Citations

To cite *Layer Ensemble Averaging for Fault Tolerance in Memristive Neural Networks*, use the following BibTeX entry:

```
@article{yousuf2024layer,
  title={Layer Ensemble Averaging for Improving Memristor-Based Artificial Neural Network Performance},
  author={Yousuf, Osama and Hoskins, Brian and Ramu, Karthick and Fream, Mitchell and Borders, William A and Madhavan, Advait and Daniels, Matthew W and Dienstfrey, Andrew and McClelland, Jabez J and Lueker-Boden, Martin and Adam, Gina C},
  journal={arXiv preprint arXiv:2404.15621},
  year={2024}
}
```

To cite this repository, use the following BibTeX entry:

## License

Distributed under the BSD-3 License. See LICENSE.txt for more information.

## Contact

- Osama Yousuf (osamayousuf@gwu.edu)
- Gina C. Adam (ginaadam@gwu.edu)