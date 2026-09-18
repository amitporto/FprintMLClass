# FprintMLClass
Molecular fingerprint based QSAR modelling using Machine Learning Techniques (KNN,RF,GB,SVM,DT,CB,AD,XGB,ET)

**Usage** <br> 
**Morgan Fingeprints** <br> 
Without chirality:
ECFP4: python fingerprint_tool.py -i LXRdata.csv -m KNN -p knn_grid.csv -f morgan -r 2 <br> 
ECFP6: python fingerprint_tool.py -i LXRdata.csv -m KNN -p knn_grid.csv -f morgan -r 3 <br>
ECFP8: python fingerprint_tool.py -i LXRdata.csv -m KNN -p knn_grid.csv -f morgan -r 4 <br>
FCFP4: python fingerprint_tool.py -i LXRdata.csv -m KNN -p knn_grid.csv -f morgan -r 2 -uf <br>
FCFP6: python fingerprint_tool.py -i LXRdata.csv -m KNN -p knn_grid.csv -f morgan -r 3 -uf <br>
FCFP8: python fingerprint_tool.py -i LXRdata.csv -m KNN -p knn_grid.csv -f morgan -r 4 -uf <br>

**With chirality:** <br> 
Add -uc after the command provided above.

**Other fingerprints** <br> 
RDKIT: python fingerprint_tool.py -i LXRdata.csv -m KNN -p knn_grid.csv -f rdkit <br>
MACCS: python fingerprint_tool.py -i LXRdata.csv -m KNN -p knn_grid.csv -f maccs <br>
Klekora-Roth (Important: add .py and .json files from https://github.com/michal-p-sapa/KRFingerprints in the main folder):<br>
python fingerprint_tool.py -i LXRdata.csv -m KNN -p knn_grid.csv -f kr <br>

**Additional information**<br> 
use 'python fingerprint_tool.py -h' command for additional options such as test set fraction (-ts), random state (-rs), cv-fold (-cv), model saving (-sm) and 
training/test set file saving (-sm).<br> 
usage: fingerprint_tool.py [-h] -i INPUT -m MODEL -p PARAMETER [-fp FINGERPRINT] [-r RADIUS] [-nb NBITS] [-uf] [-uc]
                           [-ts TESTSIZE] [-rs RANDOMSTATE] [-cv CROSSV] [-sm] [-sl SMILESLABEL] [-sf] <br> 

options:<br> 
  -h, --help            show this help message and exit <br> 
  -i, --input INPUT     .csv file with SMILES notations <br> 
  -m, --model MODEL     machine learning tool (KNN/RF/SVM/GB/AB/MLP/XGB/ET/CB/DT)<br> 
  -p, --parameter PARAMETER .csv file with ML parameters <br> 
  -fp, --fingerprint FINGERPRINT morgan/maccs/rdkit/kr (default=morgan) <br> 
  -r, --radius RADIUS   radius for Morgan (default=4) <br> 
  -nb, --nBits NBITS    nBits for Morgan (default=1024) <br> 
  -uf, --useFeatures    Use features for Morgan, default=False <br> 
  -uc, --useChirality   Use Chirality for Morgan, default=False <br> 
  -ts, --testSize TESTSIZE test set size, default=0.2<br> 
  -rs, --randomState RANDOMSTATE random_state, default=42 <br> 
  -cv, --crossV CROSSV  cross_validation fold, default=5 <br> 
  -sm, --saveModel      Model saving, default=True <br> 
  -sl, --smilesLabel SMILESLABEL label for SMILES column, default=SMILES <br> 
  -sf, --saveFiles      Training/Test data saving, default=True <br> 
