"""Custom Model Module."""

import cloudpickle

# model_file, model_path, model_file_path

def load_model(file_path):
    with open(file_path, "rb") as model_file:
        model = cloudpickle.load(model_file)
    return model
