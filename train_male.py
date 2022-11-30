from pathlib import Path
import multiprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
import hydra
from hydra.utils import to_absolute_path
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from src.factory import get_model, get_optimizer, get_scheduler, get_chained_model
from src.generator import AgeRaceImageSequence


@hydra.main(config_path="src", config_name="config")
def main(cfg):
    if cfg.wandb.project:
        import wandb
        from wandb.keras import WandbCallback
        wandb.init(project=cfg.wandb.project)
        male_callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5), WandbCallback()]
    else:
        male_callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]

    csv_path = Path(to_absolute_path(__file__)).parent.joinpath("meta", "FairFace_male.csv")
    df = pd.read_csv(str(csv_path))
    train, val = train_test_split(df, random_state=42, test_size=0.1)
    train_gen = AgeRaceImageSequence(cfg, train, "train")
    val_gen = AgeRaceImageSequence(cfg, val, "val")

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        _, male_model, _ = get_chained_model(cfg)
        opt = get_optimizer(cfg)
        scheduler = get_scheduler(cfg)
        male_model.compile(optimizer=opt,
                      loss=["sparse_categorical_crossentropy", "sparse_categorical_crossentropy"],
                      metrics=['accuracy'])

    checkpoint_dir = Path(to_absolute_path(__file__)).parent.joinpath("checkpoint")
    checkpoint_dir.mkdir(exist_ok=True)
    male_filename = "_".join([cfg.model.model_name,
                         str(cfg.model.img_size),
                         "male",
                         "weights.hdf5"])

    male_callbacks.extend([
        LearningRateScheduler(schedule=scheduler),
        ModelCheckpoint(str(checkpoint_dir) + "/" + male_filename,
                        monitor="val_loss",
                        verbose=1,
                        save_best_only=True,
                        mode="auto")
    ])

    with strategy.scope():
        male_model.fit(train_gen, epochs=cfg.train.epochs, callbacks=male_callbacks, validation_data=val_gen,
                workers=multiprocessing.cpu_count())            


if __name__ == '__main__':
    main()
