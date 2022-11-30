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
    csv_path = Path(to_absolute_path(__file__)).parent.joinpath("meta", f"FairFace_test.csv")
    test = pd.read_csv(str(csv_path))

    # get generator
    test_gen = ImageSequence(cfg, test, "test")

    # load weights
    gender_filename = "_".join([cfg.model.model_name,
                            str(cfg.model.img_size),
                            "gender",
                            "weights.hdf5"])

    male_filename = "_".join([cfg.model.model_name,
                        str(cfg.model.img_size),
                        "male",
                        "weights.hdf5"])

    female_filename = "_".join([cfg.model.model_name,
                        str(cfg.model.img_size),
                        "female",
                        "weights.hdf5"])

    checkpoint_dir = Path(to_absolute_path(__file__)).parent.joinpath("checkpoint")

    # create model
    gender_model =  tf.keras.models.load_model(str(checkpoint_dir) + "/" + gender_filename)
    male_model =  tf.keras.models.load_model(str(checkpoint_dir) + "/" + male_filename)
    female_model =  tf.keras.models.load_model(str(checkpoint_dir) + "/" + female_filename)

    predictions = []
    groundtruth = []

    # evaluate
    for image_batch, labels_batch in test_gen:
        labels_batch = np.array(labels_batch).T.tolist()
        pred_genders = gender_model.predict(image_batch)
        pred_male_ages, pred_male_races = male_model(image_batch)
        pred_female_ages, pred_female_races = female_model(image_batch)
        for ind, pred_gender in enumerate(pred_genders):
            male_prob= pred_gender[0]
            if male_prob > 0.5:
                gender_label = 0
                pred_age = np.argmax(pred_male_ages[ind])
                pred_race = np.argmax(pred_male_races[ind])
            else:
                gender_label = 1
                pred_age = np.argmax(pred_female_ages[ind])
                pred_race = np.argmax(pred_female_races[ind])
            predictions.append([gender_label, pred_age, pred_race])
        groundtruth.extend(labels_batch)
    
    predictions = np.array(predictions)
    groundtruth = np.array(groundtruth)
    
    # define cross entropy
    def cross_entropy(predictions, targets, epsilon=1e-12):
        """
        Computes cross entropy between targets (encoded as one-hot vectors)
        and predictions. 
        Input: predictions (N, k) ndarray
            targets (N, k) ndarray        
        Returns: scalar
        """
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        N = predictions.shape[0]
        ce = -np.sum(targets*np.log(predictions+1e-9))/N
        return ce
    
    # calculate the loss
    gender_loss = cross_entropy(predictions.T[0], groundtruth.T[0])
    age_loss = cross_entropy(predictions.T[1], groundtruth.T[1])
    race_loss = cross_entropy(predictions.T[2], groundtruth.T[2])
    
    loss = gender_loss + age_loss + race_loss
    
    print(f"The gender loss: {gender_loss}")
    print(f"The age loss: {age_loss}")
    print(f"The race loss: {race_loss}")
    print(f"Total loss: {loss}")
    
    # calculate the accuracy
    gender_acc = np.mean(predictions.T[0] == groundtruth.T[0])
    age_acc = np.mean(predictions.T[1] == groundtruth.T[1])
    race_acc = np.mean(predictions.T[2] == groundtruth.T[2])

    print(f"The gender accuracy: {gender_acc}")
    print(f"The age accuracy: {age_acc}")
    print(f"The race accuracy: {race_acc}")


if __name__ == '__main__':
    main()
