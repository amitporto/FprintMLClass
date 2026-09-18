# FprintMLClass
Molecular fingerprint based QSAR modelling using Machine Learning Techniques (KNN,RF,GB,SVM,DT,CB,AD,XGB,ET)

# Usage
# Morgan Fingeprints
Without chirality:
ECFP4: python fingerprint_tool.py -i LXRdata.csv -m KNN -p knn_grid.csv -f morgan 
ECFP6: python fingerprint_tool.py -i LXRdata.csv -m KNN -p knn_grid.csv -f morgan -r 3
ECFP8: python fingerprint_tool.py -i LXRdata.csv -m KNN -p knn_grid.csv -f morgan -r 4
FCFP4: python fingerprint_tool.py -i LXRdata.csv -m KNN -p knn_grid.csv -f morgan -uf 
FCFP6: python fingerprint_tool.py -i LXRdata.csv -m KNN -p knn_grid.csv -f morgan -r 3 -uf 
FCFP8: python fingerprint_tool.py -i LXRdata.csv -m KNN -p knn_grid.csv -f morgan -r 4 -uf

With chirality:
Add -uc after the command provided above.

# Other fingerprints
RDKIT: python fingerprint_tool.py -i LXRdata.csv -m KNN -p knn_grid.csv -f rdkit
MACCS: python fingerprint_tool.py -i LXRdata.csv -m KNN -p knn_grid.csv -f maccs
Klekora-Roth (Important: add .py and .json files from https://github.com/michal-p-sapa/KRFingerprints in the main folder):
python fingerprint_tool.py -i LXRdata.csv -m KNN -p knn_grid.csv -f kr

# Additional information
use 'python fingerprint_tool.py -h' command for additional options.
