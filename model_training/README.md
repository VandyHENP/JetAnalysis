# Retraining Model:
Feed datasets from Pythia script into model_training and specify outpath for the new model (training script credits to Umar)

# Testing Models:
Load whichever model and datasets needed and test how the models predicitons on different datasets.

# Models:
1. model.pth is untrained model (I believe)
2. model_unrotated.pth is trained on unrotated sets
3. model_rotated.pth is trained on rotated sets

ACCRE does not have Torch installed so these have to be run on Rithya's Machine. I edited the scripts using "nano testing_models.py" and then ran it using "python3 testing_models.py", but there's probably a way to open the files in an IDE I just did not. 
