# AMT-2022-72
Code and data required to reproduce the data and models in the paper:

# Train Convolutionnal Neural Networks

DS_training_data.zip and PIP_training_data.zip need to be extracted. 
They contain the padded images required to run cnncode_2DS.py and cnncode_PIP.py which can be used to train new models on the corresponding datasets.

# Models shown in the AMT article:

PIP_model.h5py.zip and DS_model.h5py.zip need to be extracted and can be loaded using keras.models.load_model().
For instance:
PIP_model=keras.models.load_model(path_to_Gitclone+'PIP_model.h5py')

# Random inspections results:

Contains 2 subfolders, one for each probe. Inside the actual images from the random inspections are distributed as in the confusion matrices shown in subsection 4.2, respectively in Figure 5(b) for the PIP and Figure 6(b) for the 2DS.
