# 🐶 Dog Breed Classification using VGG16 & Streamlit

An interactive, deep-learning-powered web application that classifies images of dogs into 10 popular breeds with 92.5% accuracy. Built using **TensorFlow/Keras**, **VGG16 Transfer Learning**, and **Streamlit**.

---

## 🚀 Features

* **High Accuracy Predictions:** Uses a fine-tuned VGG16 model achieving ~92.50% test accuracy.
* **Interactive UI:** Dynamic progress bars displaying full class probability distributions for every uploaded image.
* **Multiple Image Formats:** Supports `.jpg`, `.jpeg`, `.png`, and `.jfif` image uploads.
* **Responsive Dark Mode UI:** Customized CSS theme featuring gradient styling, metrics, and cards.

---

## 🐕 Supported Dog Breeds

The model can accurately distinguish between the following 10 breeds:

* 🐶 Beagle
* 🐶 Boxer
* 🐶 Bulldog
* 🐶 Dachshund
* 🐶 German Shepherd
* 🐶 Golden Retriever
* 🐶 Labrador Retriever
* 🐶 Poodle
* 🐶 Rottweiler
* 🐶 Yorkshire Terrier

---

## 🛠️ Model Architecture & Details

| Parameter | Value |
| :--- | :--- |
| **Base Model** | VGG16 (Pre-trained on ImageNet) |
| **Input Shape** | 224 × 224 × 3 |
| **Output Classes** | 10 |
| **Preprocessing** | `tf.keras.applications.vgg16.preprocess_input` |
| **Framework** | TensorFlow / Keras |
| **UI Framework** | Streamlit |

---

## 💻 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Tamil-6534/dog-breed-classifier-vgg16.git
```


2. Install Required Dependencies
Make sure you have Python 3.10+ installed, then run:

```bash
pip install streamlit tensorflow pillow numpy
```


3. Model Weight File
Ensure your trained model file vgg16_dog_breeds.h5 is placed in the root directory of the project.

🎈 How to Run the App
Launch the Streamlit web application by executing:
```bash
python -m streamlit run app.py
```

👨‍💻 Author
Tamil Arasan

Deep Learning & Computer Vision Project
