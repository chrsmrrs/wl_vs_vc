Install Dependencies for GPU (with anaconda):
```
conda create --name env python=3.10
conda activate env
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
pip install pyg-lib -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install git+https://github.com/pyg-team/pytorch_geometric.git
pip install -r requirements.txt
```
Change "+cu117" to "+cpu" if you have no gpu
