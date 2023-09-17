# CoIR: Compressive Implicit Radar

This repository provides code for the paper **``CoIR: Compressive Implicit Radar''**, by Sean M. Farrell, Vivek Boominathan, Nathaniel Raymondi, Ashutush Sabharwal, and Ashok Veeraraghavan. Contact: smf5@rice.edu

The paper is available online [[here]](https://ieeexplore.ieee.org/document/10214469)

CoIR is an analysis by synthesis method that leverages the implicit neural network bias in convolutional decoders and compressed sensing to perform high accuracy radar imaging. 

![](/images/main_fig.png)

## Installation

The code is written in python and relies on pytorch. The following is required: 
1. Python >= 3.6
2. Conda
3. Pytorch

First setup a new Conda environment and then install the required python packages in your Conda environments:
```
conda create -n coir python=3.11
conda activate coir
pip install -r requirements.txt
```
Finally, install Pytorch in Conda using a command like:
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```
For details on different Pytorch installations see [[here]](https://pytorch.org/get-started/locally/)

## Processing Experimental Radar Data
The ```sparse_radar_rec.py``` script implements our method CoIR and performs high resolution radar imaging from experimental radar data cube ADC measurements. We used the radar measurements from the ColoRadar data set. The script will generate reconstructions for our propsoed network ComDecoder and competing baselines. Reconstructed images are in polar coordinates and will be saved as .png files and torch matrcies for later viewing and processing.

To convert the recosntructed images from polar to cartesian coordinates run the ```gen_plots_singleScene_results.py``` script. See the ```example_recon_cartesian_run1_frame182``` folder for example reconstructions for all methods.

## Dataset
The experimental data used in this work is from the ColoRadar data set which can be found [[here]](https://arpg.github.io/coloradar/)

Kramer, Andrew, Kyle Harlow, Christopher Williams, and Christoffer Heckman. “ColoRadar: The direct 3D millimeter wave radar dataset.” The International Journal of Robotics Research 41, no. 4 (2022): 351-360.

## Code Release Plan
* Sept. 18 - 22: publish Google Colab notebook that can be used to quickly implement our proposed method CoIR.

* Sept. 25 - 29: publish scripts to post-process raw radar ADC measuremetns from the ColoRadar data set into radar datacubes that can be used with our method.

## Citation
```
@ARTICLE{farrell_coir_2023,
  author={Farrell, Sean M. and Boominathan, Vivek and Raymondi, Nathaniel and Sabharwal, Ashutosh and Veeraraghavan, Ashok},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={CoIR: Compressive Implicit Radar}, 
  year={2023},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TPAMI.2023.3301553}}

```
## Licence
