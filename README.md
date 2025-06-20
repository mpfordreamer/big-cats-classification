# 🐾 Big Cats Classifier with CNN
![3 Cats CNN](https://github.com/user-attachments/assets/28fceb7d-6e40-4154-97d1-7282ef5656ac)

This project implements a Convolutional Neural Network (CNN) to classify images of three big cats: **Cheetah**, **Lion**, and **Tiger**. The project covers the entire workflow, from model training and evaluation to converting the model into various formats for deployment.

---

## 📁 Project Structure

```
├── checkpoints/              # Model checkpoints (.keras) for val_accuracy and val_loss
├── datasets/                # Structured dataset (train/val/test)
│   ├── train/
│   ├── val/
│   └── test/
├── saved_model/             # Model in TensorFlow SavedModel format
├── tfjs_model/              # Model in TensorFlow.js format
├── tflite/                  # Model in TensorFlow Lite format
│   ├── model_loss.tflite
│   └── labels.txt
├── notebook.ipynb           # Main project notebook
├── requirements.txt         # Python dependency list
└── README.md                # This project documentation
```

---

## 🚀 Project Overview

-   A CNN model was trained using a dataset of big cat images, split into **train**, **val**, and **test** sets.
-   The training process used the following strategies:
    -   `EarlyStopping` to prevent overfitting.
    -   `ReduceLROnPlateau` to adjust the learning rate.
    -   Checkpointing based on `val_accuracy` and `val_loss`.

> ⏱️ **Best Checkpoints**
>
> -   **val_loss** (5th epoch): `val_loss = 0.2668` | `val_accuracy = 95.58%`
> -   **val_accuracy** (11th epoch): `val_accuracy = 95.98%` | `val_loss = 0.3180`

---

## 📊 Model Evaluation

The model was tested on a set of 126 images across 3 classes:

| Model                     | Accuracy   | Average F1-Score |
| ------------------------- | ---------- | ---------------- |
| 🏆 **Best `val_loss`**    | **95.24%** | **0.9524**       |
| 🎯 Best `val_accuracy`    | 94.44%     | 0.9450           |

The `val_loss` model demonstrated **more stable predictions and better generalization**, making it the chosen model for conversion and deployment.

---

## 🔁 Model Formats

The best model (based on `val_loss`) has been converted into the following formats:

| Format       | Location                          | Description                       |
| ------------ | --------------------------------- | --------------------------------- |
| `SavedModel` | `saved_model/bigcats_loss_model/` | For deployment via TensorFlow     |
| `TFLite`     | `tflite/model_loss.tflite`        | For mobile/embedded devices       |
| `TF.js`      | `tfjs_model/`                     | For deployment in the browser (web) |

---

## 💡 How to Run

1.  **Clone the repository and install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Notebook**
    Open `notebook.ipynb` to view the complete workflow, from training and evaluation to model conversion.

---

## 📦 Dataset

The dataset is structured into the following directories:

```bash
datasets/
├── train/    # Images for training
├── val/      # Images for validation
└── test/     # Images for testing
```

---

## 📈 Training Visualization

Training logs are saved in the `logs/` directory and can be visualized with TensorBoard:

```bash
tensorboard --logdir=logs/
```

---

## 📝 Key Takeaways

-   The `val_loss` model was selected as the best because it showed the **lowest prediction error** and **superior generalization** on the test data.
-   The evaluation demonstrates **stable and accurate performance** across all classes, making the model ready for use in various image classification scenarios.
