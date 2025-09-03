# Bienvenue sur le référentiel DeepLearning 

<img src="https://raw.githubusercontent.com/rbizoi/DeepLearning/refs/heads/main/images/style_transfer_result_at_iteration_19.png" width="768">

# Installation 

## 01 <b></b><a href="https://www.anaconda.com/download/success">Installation Anaconda <br>download(Windows, Linux, MacOs)</a></b>
<a href="https://www.anaconda.com/download/success"><img src="https://raw.githubusercontent.com/rbizoi/FHU_TARGET_ScientificDays_2025/refs/heads/main/images/anaconda_setup.png" width="256"></a>

<i>Pour plus d'aide vous avez trois vidéos suivant votre système d’exploitation :<br></i>
>> <i><a href="https://www.anaconda.com/docs/getting-started/anaconda/install#windows-installation">Windows</a><br></i>
>> <i><a href="https://www.anaconda.com/docs/getting-started/anaconda/install#linux-installer">Linux</a><br></i>
>> <i><a href="https://www.anaconda.com/docs/getting-started/anaconda/install#macos-graphical-installer">MacOs</a><br></i>

## 02 Mise à jour des librairies de l’environnement <b>base</b>
Ouvrir un environnement de commandes (une fenêtre shell pour Mac ou Linux).
>> <b>Windows et environnement graphique MacOs</b>
<img src="https://raw.githubusercontent.com/rbizoi/FHU_TARGET_ScientificDays_2025/refs/heads/main/images/conda_prompt.png" width="256">

>> <b>Linux</b>
<img src="https://raw.githubusercontent.com/rbizoi/FHU_TARGET_ScientificDays_2025/refs/heads/main/images/terminal.png" width="128">

>> Exécutez les commandes suivantes dans l'environnement de commandes <br><b>(utilisez copier/coller)</b>

```
conda activate root
conda update --all
python -m pip install --upgrade pip
```

## 03 Création de l’environnement <b>keras</b>

>> <b>Attention aux systèmes d’exploitation</b>

## 03.1 <b>Windows</b>

```
conda create -n keras -c conda-forge  python==3.12 ipython ipython-sql jupyter notebook numpy pandas pyarrow matplotlib seaborn portpicker biopython flatbuffers redis colour pydot pygraphviz pyyaml pyspark folium scikit-image scikit-learn plotly imgaug tifffile imagecodecs
```

## 03.2 <b>Linux ou MacOs</b>

```
conda create -p /home/nom_utilisateur/anaconda3/envs/keras -c conda-forge  python==3.12 ipython ipython-sql jupyter notebook numpy pandas pyarrow matplotlib seaborn portpicker biopython flatbuffers redis colour pydot pygraphviz pyyaml pyspark folium scikit-image scikit-learn plotly imgaug tifffile imagecodecs
```

## 04 Configuration de l’environnement <b>keras</b>

```
conda activate keras

pip install keras tensorflow --upgrade
```