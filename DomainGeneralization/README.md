
## Get Started

### Step1: Prepare Environment

``` pip install -r requirements.txt ```

### Step2: Process Illusion-IN data into the required DG format

Please specify the path to your generated Illusion-IN in ```root```.

``` python clean_sin.py ```


### Run Domain Generalization

'''
bash run.sh
'''

### Acknowledgement
This codebase is based on  DomainBed (https://github.com/facebookresearch/DomainBed) and DPLCLIP (https://github.com/shogi880/DPLCLIP). We thank the authors of DomainBed and DPLCLIP for their great codebase.
