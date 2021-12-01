# GraphormerIPU  

Implementation of [Graphormer](https://github.com/microsoft/Graphormer) for Graphcore's Intelligence Processing Unit (IPU).  
  
The given dataset is a toy dataset.  
```
# To prepare the actual dataset :
python3 srcIPU/pcq_wrapper.py
```  
This can take several hours.

### Setup Environment
```
# create a new environment
conda create —name graphormerenv python=3.6
pip3 install poptorch 2.3
pip3 install torch==1.9.0+cu111 torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html --upgrade-strategy only-if-needed
pip3 install torch-geometric ogb tqdm torch-sparse torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html --upgrade-strategy only-if-needed
conda install -c rdkit rdkit cython
```

### Run Training
```
# Requires 16 IPUs.
bash train.sh
```  

### Run Test
```
# Requires 16 IPUs.
bash test.sh
```
