# Bienvenue sur le référentiel DeepLearning 

<img src="https://raw.githubusercontent.com/rbizoi/DeepLearning/refs/heads/main/images/style_transfer_result_at_iteration_19.png" width="768">


# Installation 

## 01 <b></b><a href="https://www.anaconda.com/download/success">Installation Anaconda</a></b>
<a href="https://www.anaconda.com/download/success"><img src="https://raw.githubusercontent.com/rbizoi/FHU_TARGET_ScientificDays_2025/refs/heads/main/images/anaconda.png" width="256"></a><

## 04 Mise à jour des librairies de l’environnement <b>base</b>

```
conda activate root
conda update --all
python -m pip install --upgrade pip
```

## 05 Création de l’environnement <b>keras</b>

## 05.1 <b>Windows</b>

```
conda create -n keras -c conda-forge  python==3.12 ipython ipython-sql jupyter notebook numpy pandas pyarrow matplotlib seaborn portpicker biopython flatbuffers redis colour pydot pygraphviz pyyaml pyspark folium scikit-image scikit-learn plotly imgaug tifffile imagecodecs
```

## 05.2 <b>Linux</b>

```
conda create -p /home/razvan/anaconda3/envs/keras -c conda-forge  python==3.12 ipython ipython-sql jupyter notebook numpy pandas pyarrow matplotlib seaborn portpicker biopython flatbuffers redis colour pydot pygraphviz pyyaml pyspark folium scikit-image scikit-learn plotly imgaug tifffile imagecodecs
```

## 06 Configuration de l’environnement <b>keras</b>

```
conda activate keras

pip install keras tensorflow --upgrade
```