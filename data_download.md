# Data Download

To generate our benchmark BeCoS you need to download 6 of the 7 datasets. TOSCA is already placed in the `data/` folder. If you have downloaded some dataset before and want to reuse them, make sure that they are complete and in their original form (not remeshed). If there are any problems, please create an issue.

## Automatic Download
Use the `automatic_download.py` file to download and preprocess the KIDS and SHREC20 dataset. 

## Manual Download
Some Datasets need to be downloaded manually. 


### FAUST Dataset
The FAUST dataset can be downloaded [here](https://faust-leaderboard.is.tuebingen.mpg.de/download). You need to create an account to download the data. Place the downloaded `MPI-FAUST` folder in the `data/` folder.

### SMAL Dataset
The SMAL dataset can be downloaded [here](https://smal.is.tue.mpg.de/index.html). You need to create an account to download the data. If you are logged in you can use the direct download links for [big_cats](https://download.is.tue.mpg.de/download.php?domain=smal&resume=1&sfile=big_cats_results.tgz), [dogs](https://download.is.tue.mpg.de/download.php?domain=smal&resume=1&sfile=dogs_results.tgz), [horses](https://download.is.tue.mpg.de/download.php?domain=smal&resume=1&sfile=horses_results.tgz), [cows](https://download.is.tue.mpg.de/download.php?domain=smal&resume=1&sfile=cows_results.tgz) and [hippos](https://download.is.tue.mpg.de/download.php?domain=smal&resume=1&sfile=hippos_results.tgz). Please download all of them and place them in the currently empty `data/smal` folder. The folder structure should look like `data/smal/big_cats`, `data/smal/horses`, .... If the direct download links do not work, you can find all these shapes on the [SMAL download website](https://smal.is.tue.mpg.de/download.php).

## DT4D Dataset
Fill out the form [here](https://docs.google.com/forms/d/e/1FAIpQLSckMLPBO8HB8gJsIXFQHtYVQaTPTdd-rZQzyr9LIIkHA515Sg/viewform). You will get the link for the dataset via Email afterwards. Download the dataset and place it in the `data/` folder. Use our `preprocess_dt4d.py` to generate the meshes used in the BeCoS benchmark.


### SCAPE Dataset
The SCAPE dataset has been made available for research purposes [on this webpage](https://graphics.soe.ucsc.edu/private/data/SCAPE). Please email Prof. James Davis (**davis at soe.ucsc.edu**) to get the password. Place the data under `SCAPE/Meshes`.