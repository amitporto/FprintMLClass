# FprintMLClass
Molecular fingerprint based QSAR modelling using Machine Learning Techniques (KNN,RF,GB,SVM,DT,CB,AD,XGB,ET)

**Usage** <br> 
**Morgan Fingeprints** <br> 
Without chirality:
ECFP4: python fingerprint_tool.py -i LXRdata.csv -m KNN -p knn_grid.csv -f morgan <br> 
ECFP6: python fingerprint_tool.py -i LXRdata.csv -m KNN -p knn_grid.csv -f morgan -r 3 <br>
ECFP8: python fingerprint_tool.py -i LXRdata.csv -m KNN -p knn_grid.csv -f morgan -r 4 <br>
FCFP4: python fingerprint_tool.py -i LXRdata.csv -m KNN -p knn_grid.csv -f morgan -uf <br>
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
use 'python fingerprint_tool.py -h' command for additional options.<br> 
