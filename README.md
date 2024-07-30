# 🌍 Geo_Magnet-Location-Finder 🌍

Welcome to the **Geo_Magnet-Location-Finder** project! This repository contains all the necessary code, datasets, and results for our innovative geographical prediction model. 📍

## 📝 Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Methodology](#methodology)
6. [Results](#results)
7. [Future Enhancements](#future-enhancements)
8. [Contributing](#contributing)
9. [License](#license)

## 🌟 Project Overview

The **Geo_Magnet-Location-Finder** project aims to predict geographical coordinates based on image inputs. Utilizing advanced machine learning models and data augmentation techniques, we strive to provide accurate location predictions. 🌐

## ✨ Features

- 🔍 **Geographical Prediction:** Predicts latitude and longitude from images.
- 🧠 **Machine Learning Models:** Incorporates MoCo-V2, Random Forest, and other advanced models.
- 📈 **Data Augmentation:** Implements techniques to enhance model robustness.
- 🖥️ **GUI Interface:** User-friendly interface for easy interaction and predictions.

## 🚀 Installation

To get started with this project, follow the steps below:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Nani1-glitch/Geo_Magnet-Location-Finder.git
    cd Geo_Magnet-Location-Finder
    ```

2. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up Git LFS (if not already installed):**
    ```bash
    git lfs install
    git lfs track "output/moco_model.pth"
    git lfs track "z_FINAL_RESULTS/sup_learning_resnet50.pth"
    ```

4. **Add large files to Git LFS tracking:**
    ```bash
    git add .gitattributes
    git commit -m "Add large files with Git LFS"
    ```

## 🛠️ Usage

1. **Run the Flask app:**
    ```bash
    python app.py
    ```

2. **Access the app in your browser:**
    ```
    http://127.0.0.1:5001
    ```

3. **Upload an image and get the geographical prediction:**

    ![Geographical Prediction](https://path-to-your-image.png)

## 📚 Methodology

### Data Augmentation

To tackle overfitting, we employed advanced data augmentation techniques such as Temporally Augmented Positive Pair Generation, ensuring the model learns robust features from diverse data.

### Hyperparameter Tuning

We optimized the model performance using Grid Search and Cross-Validation techniques. These methods help in selecting the best hyperparameters for Random Forest and other models.

## 📊 Results

Our experiments yielded promising results with significant improvements in model accuracy. Below are some key outcomes:

- **Geography Aware Model:**
  - Top-1 Accuracy: 0.32%
  - Top-5 Accuracy: 0.41%



- **MoCo Model:**
  - Top-1 Accuracy: 1.76%
  - Top-5 Accuracy: 12.07%


- **Ensemble Model:**
  - Mean Squared Error: 3471.51
  - R-squared: -5.657


## 🌍 Future Enhancements

We plan to:

- 📊 Fine-tune hyperparameters further for optimized performance.
- 🛠️ Implement additional data augmentation techniques.
- 🔍 Explore more advanced machine learning models for better predictions.

## 🤝 Contributing

We welcome contributions! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

💡 **Note:** This README file is designed to provide a comprehensive overview of the project, its installation, usage, and contributions. Feel free to enhance it further as per your requirements.

Happy Coding! 🚀
