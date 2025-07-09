# [BeCoS](https://nafieamrani.github.io/BeCoS/)
Official repository for the SGP 2025 paper: Beyond Complete Shapes: A quantitative Evaluation of 3D Shape Matching Algorithms by Viktoria Ehm, Nafie El Amrani, Yizheng Xie, Lennart Bastian, Maolin Gao, Weikang Wang, Sang Lu, Dongliang Cao, Tobias Weißberg, Zorah Lähner, Daniel Cremers and Florian Bernard.
## ⚙️ Installation 
```
conda create -n becos python=3.8
conda activate becos
pip install -r requirements.txt
```

## 📝 Dataset 
To generate the partial dataset for all given datasets, download all data under the ```./data``` folder (see `data_download.md` for more information) or adjust the path in the ```config/general_config.conf``` to the corresponding path. 
For demo purposes we provide the TOSCA dataset and will generate shape pairs only for this dataset.

## 🧑‍💻️‍  Usage

### Demo: Generation of subset of BeCoS
In this demo, you can generate a subset of BeCoS using TOSCA shapes only. TOSCA shapes are included in the `data` folder. Simply run: 
```
python main.py --config demo_config
```


### Generate Partial Shapes
The generation is tested on **Linux only**.
To start the partial-to-partial shape generation of the benchmark, run 
```
python main.py
```

### Generate Partial-to-Full and Full-to-Full Pairs
For the other datasets you can adjust the setting of `config/general_config.conf` to `full_to_full` or `partial_to_full`.

### Generation of different versions of the benchmark
To control the options of the generation of BeCoS, you can adjust the settings of `config/general_config.conf`. It stores all options for the generation. Note that changing the default settings will result in generating a set of shapes that are different from the ones used in the benchmark.

### Runtimes for generation of the benchmark
All runtimes reported below are on a linux machine with: AMD Ryzen 9 5950X 16-Core Processor and 64GB of RAM.

|    Setting    | Full-to-full | Partial-to-full | Partial-to-partial |
|:-------------:|:------------:|:---------------:|--------------------|
| Runtime (min) | 31.52        | 63.23           | 33.52              |

## 🎓 Attribution
```
@article{ehm2025becos,
    journal = {Computer Graphics Forum},
    title = {{Beyond Complete Shapes: A Benchmark for Quantitative Evaluation of 3D Shape Matching Algorithms}},
    author = {Ehm, Viktoria and El Amrani, Nafie and Xie, Yizheng and Bastian, Lennart and Gao, Maolin and Wang, Weikang and Sang, Lu and Cao, Dongliang and Weißberg, Tobias and L{\"a}hner, Zorah and Cremers, Daniel and Bernard, Florian},
    year = {2025},
    publisher = {The Eurographics Association and John Wiley & Sons Ltd.},
    ISSN = {1467-8659},
    DOI = {10.1111/cgf.70186}
}
```

Please make sure that if using BeCoS you also attribute the underlying datasets. 
#### FAUST
```
@inproceedings{bogo2014faust,
  title={FAUST: Dataset and evaluation for 3D mesh registration},
  author={Bogo, Federica and Romero, Javier and Loper, Matthew and Black, Michael J},
  booktitle={CVPR},
  pages={3794--3801},
  year={2014}
}
```
#### SCAPE
```
@article{anguelov2005scape,
  title={Scape: shape completion and animation of people},
  author={Anguelov, Dragomir and Srinivasan, Praveen and Koller, Daphne and Thrun, Sebastian and Rodgers, Jim and Davis, James},
  journal={ACM SIGGRAPH 2005 Papers},
  pages={408--416},
  year={2005}
}
```
#### SMAL
```
@inproceedings{Zuffi:CVPR:2017,
        title = {{3D} Menagerie: Modeling the {3D} Shape and Pose of Animals},
        author = {Zuffi, Silvia and Kanazawa, Angjoo and Jacobs, David and Black, Michael J.},
        booktitle = {CVPR},
        month = jul,
        year = {2017},
        month_numeric = {7}
      }
```
### DT4D
```
@inproceedings{li20214dcomplete,
  title={4dcomplete: Non-rigid motion estimation beyond the observable surface},
  author={Li, Yang and Takehara, Hikari and Taketomi, Takafumi and Zheng, Bo and Nie{\ss}ner, Matthias},
  booktitle={ICCV},
  pages={12706--12716},
  year={2021}
} 
```
### TOSCA
```
@book{bronstein2008numerical,
author = {Bronstein, Alexander and Bronstein, Michael and Kimmel, Ron},
title = {Numerical Geometry of Non-Rigid Shapes},
year = {2008},
isbn = {0387733000},
publisher = {Springer Publishing Company, Incorporated},
edition = {1},
}
```
### KIDS
```
@inproceedings{rodola2014dense,
  title={Dense non-rigid shape correspondence using random forests},
  author={Rodola, Emanuele and Rota Bulo, Samuel and Windheuser, Thomas and Vestner, Matthias and Cremers, Daniel},
  booktitle={CVPR},
  pages={4177--4184},
  year={2014}
}
```
### SHREC20
```
@article{dyke2020shrec,
  title={SHREC’20: Shape correspondence with non-isometric deformations},
  author={Dyke, Roberto M and Lai, Yu-Kun and Rosin, Paul L and Zappala, Stefano and Dykes, Seana and Guo, Daoliang and Li, Kun and Marin, Riccardo and Melzi, Simone and Yang, Jingyu},
  journal={Computers \& Graphics},
  volume={92},
  pages={28--43},
  year={2020},
  publisher={Elsevier}
}
```

## 🚀 License
This repo is licensed under MIT licence.