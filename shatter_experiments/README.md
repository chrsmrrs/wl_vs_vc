Install Dependencies for GPU (with anaconda):
```
conda create --name env python=3.10
conda activate env
conda install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg
pip install -r requirements.txt
```
Change "+cu117" to "+cpu" if you have no gpu
