# Dependency
The testing environment is Python 3.9

The dependencies are included in `requirements.txt`
```bash
pip install -r requirements.txt
```

It is suggested to have a cuda environment. You could install the cuda and cudnn using conda
```bash
conda install cudatoolkit cudnn
```

# Data
The datasets and checkpoints could be downloaded [here](https://drive.google.com/drive/folders/1Xh6eARKz3w9qVVuGqDPffnL_7kTbvC7N?usp=sharing). It includes `checkpoint`, `data`, `meta`, and `pretrained_models`. Please put those folders in the same path of the project.

# Demo
To predict the pictures, you could use the `demo.py`:
```bash
python demo.py
```
If your computer doesn't have qt supported, you could comment 159 line in `demo.py`
```python
cv2.imshow("results", img)
```
and uncomment the 161 line:
```python
cv2.imwrite(f"results/result_{l}_image_cnt}.jpg", img)
```
The results will then saved into `results` folder.

# Train
We have two tasks:
- age-gender estimation
- age-gender-race estimation
To train the age-gender estimation, use
```bash
python train-age-gender.py
```

To train the age-gender-race model, run
```bash
python train.py
```
The default configuration is using ResNet152V2 as pretrained model and using FairFace as dataset. You could change the database to `UTKface` or `test`. The `test` is the FairFace test set.

The available models could be found [here](https://keras.io/api/applications/).

## Check training curve
The training logs can be easily visualized via wandb by:

1. create account from [here](https://app.wandb.ai/login?signup=true)
2. create new project in wandb (e.g. "age-gender-estimation")
3. run `wandb login` on terminal and authorize
4. run training script with `wandb.project=age-gender-estimation` argument
5. check dashboard!

## Submodels
In this assignment, we trained lots of submodels including the model purely for prediction gender, the model predicts age and race only for male, and the model predicts age and race only for female. Those models are trained my different python files.

For the gender model:
```bash
python train_gender.py
```

For the male model:
```bash
python train_male.py
```

For the female model:
```bash
python train_female.py
```

# Estimation
To evaluate the model all in one, run
```bash
python test.py data.db=test
```

To evaluate the two-stage model, run
```bash
python test_chained.py data.db=test
```

To evaluate the quality of the datasets, which means the balanceness of the prediction, we can use the `test_FairFace.py`and `test_UTK.py` to get the prediction and the groundtruth. They will be saved into outputs as `csv` and thus we can apply further analysis.
```bash
python test_FairFace.py
```
and 
```bash
python test_UTK.py
```
