from pathlib import Path
import pandas as pd
import hydra
from hydra.utils import to_absolute_path
import tensorflow as tf
from src.generator import ImageSequence

import numpy as np

@hydra.main(config_path="src", config_name="config")
def main(cfg):
    # the path to test meta file
    csv_path = Path(to_absolute_path(__file__)).parent.joinpath("meta", f"FairFace.csv")
    test = pd.read_csv(str(csv_path))

    # get generator
    test_gen = ImageSequence(cfg, test, "test")
    
    # load weights
    model_filename = "_".join([cfg.model.model_name,
                            str(cfg.model.img_size),
                            'all',
                            "weights.hdf5"])

    checkpoint_dir = Path(to_absolute_path(__file__)).parent.joinpath("checkpoint")

    model =  tf.keras.models.load_model(str(checkpoint_dir) + "/" + model_filename)

    predictions = []
    groundtruth = []

    # calculate the prediction and ground truth
    for image_batch, labels_batch in test_gen:
        labels_batch = np.array(labels_batch).T.tolist()
        res = model.predict(image_batch)
        pred_genders, pred_ages, pred_races = res
        for ind, pred_gender in enumerate(pred_genders):
            pred_gender = np.argmax(pred_gender)
            pred_age = np.argmax(pred_ages[ind])
            pred_race = np.argmax(pred_races[ind])
            predictions.append([pred_gender, pred_age, pred_race])
        groundtruth.extend(labels_batch)
    
    predictions = np.array(predictions)
    groundtruth = np.array(groundtruth)
    
    df = pd.DataFrame()
    
    df["gender predictions"] = predictions.T[0]
    df["gender groundtruth"] = groundtruth.T[0]
    df["age predictions"] = predictions.T[1]
    df["age groundtruth"] = groundtruth.T[1]
    df["race predictions"] = predictions.T[2]
    df["race groundtruth"] = groundtruth.T[2]
    
    df.to_csv("./FairFace_prediction.csv", index=False)

if __name__ == '__main__':
    main()
