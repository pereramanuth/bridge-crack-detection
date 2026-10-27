# 🌉 Bridge Crack Detection developing phase

Deep learning project that detects cracks in bridge surface images and shows where they are. A pretrained **ResNet18** is fine-tuned in **PyTorch** to classify image patches as crack or no crack. **Grad-CAM** and a **sliding-window heatmap** then highlight the cracked regions.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet18-orange)
![Colab](https://img.shields.io/badge/Run%20on-Google%20Colab-yellow)

## 🎯 Goal

Inspecting bridges by eye is slow and subjective. This project builds a model that flags cracks automatically and also shows which part of the image it based its decision on, so the result can be checked by a person.

## 📦 Dataset

**Bridge Crack Image (DBCC)** from Kaggle: [yidazhang07/bridge-cracks-image](https://www.kaggle.com/datasets/yidazhang07/bridge-cracks-image)

- 🖼️ 50,000 training and 5,000 validation images in `.jpeg` format
- 🏷️ Labels come from the filenames: `crack123.jpeg` is a crack, `no123.jpeg` is not
- <img width="1589" height="575" alt="image" src="https://github.com/user-attachments/assets/d1e040fd-4605-424b-8463-e4f1da617361" />

- ✂️ A test set of 5,000 images (10%) is held out from the training images, so final scores are never measured on data used for tuning

The dataset is not included in this repo. Download it from Kaggle.

## 🧠 Method

- **Model:** ResNet18 pretrained on ImageNet, with a new 2-class output layer
- **Input:** images resized to 128 x 128 and loaded into memory once for fast GPU training
- **Training in two stages:**
  1. Train only the new output layer for 2 epochs (learning rate 1e-3)
  2. Fine-tune the whole network (learning rate 1e-4) with a learning rate scheduler and early stopping
- **Class-weighted loss** to handle any imbalance between crack and no-crack images
- **Augmentation:** random horizontal and vertical flips
- **Decision threshold:** chosen on the validation set to maximize F1, then applied to the test set
- **Localization:**
  - 🔥 **Grad-CAM** shows which pixels drove the crack prediction
  - 🪟 **Sliding window** runs the classifier over tiles of a large image and averages the results into a crack probability map

## 📓 Notebooks

| Notebook | What it does |
|---|---|
| `01_data_exploration.ipynb` | Reads the dataset, labels images from filenames, builds the train, validation and test split, shows sample images |
| `02_train_model.ipynb` | Loads the images, trains ResNet18 in two stages, saves the best model, plots loss and F1 curves |
| `03_evaluate_and_localize.ipynb` | Test metrics, confusion matrix, ROC curve, Grad-CAM examples, sliding-window heatmaps |

## 📊 Results

Evaluated on the held-out test set of 5,000 images (4,400 no crack, 600 crack), using the decision threshold of 0.925 chosen on the validation set.

| Metric | Value |
|---|---|
| Accuracy | 99.12% |
| Precision (crack) | 93.85% |
| Recall (crack) | 99.17% |
| F1 (crack) | 96.43% |
| ROC AUC | 0.9982 |

The model catches almost every crack (recall 99.2%). Its main weakness is false alarms: precision on the crack class is lower because the test set has far more no-crack images than crack images, so even a small false positive rate adds up.

⚠️ The test set is drawn from the same pool as the training images, so scores on real bridge photos from other sources may be lower.

<img width="1989" height="490" alt="image" src="https://github.com/user-attachments/assets/b55872f2-2cec-4e91-8f6c-19d4a5926c56" />

<img width="1479" height="511" alt="image" src="https://github.com/user-attachments/assets/9d61d8b3-3ae1-4fe3-a3bc-fc7af9634fa3" />


## 🚀 How to run

The notebooks were run on **Google Colab** with a T4 GPU.

1. Upload `archive.zip` (the Kaggle download) to a Google Drive folder named `computer_vision`
2. Upload the three notebooks to the same folder and open them with Google Colaboratory
3. Set **Runtime, Change runtime type, T4 GPU**
4. Run the notebooks in order: 01, then 02, then 03

Each notebook starts with a setup cell that mounts Drive and unzips the bridge dataset. Files such as `manifest.csv`, the trained model and the threshold are saved in the Drive folder and shared between notebooks.

## 🛠️ Tech stack

Python, PyTorch, torchvision, scikit-learn, pandas, NumPy, Matplotlib, Google Colab

## 🔮 Possible improvements

- Pixel-level crack segmentation with a U-Net instead of patch classification
- Test on real bridge photos taken outside the dataset
- Try larger backbones such as ResNet50 or EfficientNet
- Add stronger augmentation for lighting and weather changes

## 📄 License and credits

Dataset credit goes to its original authors on Kaggle. Check the dataset page for its license before reusing the data.
