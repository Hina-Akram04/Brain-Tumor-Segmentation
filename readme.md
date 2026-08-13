# Brain Tumor Segmentation (BraTS)

A U-Net model that segments brain tumors from MRI slices into three sub-regions — necrotic core (NCR/NET), edema, and enhancing tumor. Built and trained on CPU using a 2,300-slice subset of the BraTS2020 dataset. Includes a Streamlit app for uploading a slice and viewing the segmentation.

## How it works

Each MRI slice has 4 channels (T1, T1ce, T2, FLAIR) and a ground-truth mask. The model takes the 4-channel image and outputs a 4-class pixel map:

* Background
* NCR/NET
* Edema
* Enhancing tumor

The model is trained using a combined Dice + cross-entropy loss.

Only a subset of the full dataset (2,300 of 57,195 slices) was used because training was performed on CPU without a GPU. The subset is weighted toward tumor-positive slices so the model learns tumor boundaries instead of mostly predicting background.

## Project Structure

```text
BrainTumorSegmentation/
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   └── slices/
│
├── models/
│   └── best_model.pth
│
├── outputs/
│   ├── last_prediction.png
│   ├── logs.csv
│   └── sample_result.png
│
├── src/
│   ├── dataset.py
│   ├── metrics.py
│   ├── model.py
│   ├── predict.py
│   └── train.py
│
├── app.py
├── notebook.ipynb
├── requirements.txt
├── readme.md
└── .gitignore
```

## Setup

Create and activate a virtual environment, then install the required packages:

```bash
pip install -r requirements.txt
```

The dataset is from Kaggle:

`awsaf49/brats2020-training-data`

The dataset should be downloaded and placed inside the `data/` directory.

## Running the Project

### Train the model

```bash
python src/train.py
```

The best model is saved to:

```text
models/best_model.pth
```

### Predict on a single slice

```bash
python src/predict.py --file data/slices/<filename>.h5 --save outputs/result.png
```

### Run the Streamlit app

```bash
streamlit run app.py
```

The app allows you to upload an MRI slice and view the predicted tumor segmentation.

## Results

On the held-out validation split after 20 epochs:

| Metric           | Score |
| ---------------- | ----: |
| Dice coefficient |  0.78 |
| IoU              |  0.68 |
| Pixel accuracy   | 99.3% |

Model size: approximately 1.9M parameters.

## Notes

* The model is trained on 2D MRI slices rather than full 3D volumes. This keeps training CPU-feasible but does not capture cross-slice context.
* The input contains four MRI modalities: T1, T1ce, T2 and FLAIR.
* The three tumor mask channels are mutually exclusive per pixel and are therefore represented as a single 4-class label map.
* The application expects `.h5` slices in the same format as the training data.
* Raw DICOM or NIfTI files are not directly supported by the current application.
