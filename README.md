
 <h1 align="center">Sugar-Binding Predictor</h1>

Sugar binding predictor, a deep learning classifier to predict binding between protein and carbohydrate.  

![overview drawio](https://github.com/Jacky233emm/sugar_binding_predictor/assets/91257855/d781ac20-844f-42fe-bf31-033a57e2bebc)

 <h2 align="left">introduction</h2>


This method  takes the .pdb structure file as input and outputs 1 or 0 to indicate successful or failed binding. Protein-sugar interacting graph is generated from pre-processing the structure file. Then, interacting graph and other information pass to a model included spherical message passing network [Liu et al., 2021], principal neighborhood aggregation layers [Corso et al., 2020], a super-node-based hierarchical graph pooling layer [Ying et al., 2019]. Following by fully connected layers, the classifier returns a binary result.  

![image](https://github.com/Jacky233emm/sugar_binding_predictor/assets/91257855/9d2bb179-db61-4328-8851-bed8abfb9c94)


 <h2 align="left">Installation</h2>


Recommand to use virtual environment for building.

_classifier model part:  _
```sh
conda create -n pred_test python=3.8  pip  
conda activate pred_test  
pip install -r ./requirements_m.txt   
pip install torch torch-sparse torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-1.13.0+cu117.html  
pip install dive-into-graphs
```


_data preprocessing part (PyMOL depended):  _
```sh
conda install -c conda-forge -c schrodinger pymol-bundle  
conda install -c conda-forge libnetcdf==4.7.3  



 <h2 align="left">Running</h2>

Simply run by: 
```sh
python run_predict.py
```
Running option: 

`--complex_pdb` : Load complex .pdb file.

`--file_path` : File path of input file. 

`--with_label` : Label `1` or `0` of input complex, if unknown set `None`. 

`--complex_list` : Load list of complex .pdb file, label can be added after the file name seperated by a tab. 

`--model` : Model parameter for prediction, there are `well-trained` and `general` two options, in default both of them would be used.  



 <h2 align="left">Contact</h2>

 Yijie Luo [Jacky](https://github.com/Jacky233emm) -- yijie.luo@bristol.ac.uk

