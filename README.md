# Hand-written-classification-neural-network---MNIST-dataset

## Overview
This project involves building, training, and evaluating a neural network to classify handwritten digits from the MNIST dataset using PyTorch. The goal is to achieve high accuracy in recognizing digits (0-9) from 28x28 grayscale images. The project includes data preprocessing, model definition, training, validation, testing, and model improvement through hyperparameter tuning.

## Project Structure
- **File**: `MNIST_Handwritten_Digits-STARTER.ipynb`
- **Dataset**: MNIST dataset, accessed via `torchvision.datasets.MNIST`
- **Framework**: PyTorch
- **Environment**: Python 3.7.6
- **Dependencies**: Listed in `requirements.txt` (e.g., `ipywidgets`, `torch`, `torchvision`, `matplotlib`, `numpy`, `scikit-learn`)

## Setup Instructions
1. **Install Dependencies**:
   - Run the following command to install required packages:
     ```bash
     python -m pip install -r requirements.txt
     ```
   - Restart the Jupyter kernel after installation to ensure the packages are loaded.

2. **Download Dataset**:
   - The MNIST dataset is automatically downloaded when the `torchvision.datasets.MNIST` object is instantiated with `download=True`.

3. **Hardware**:
   - The code checks for GPU availability using `torch.cuda.is_available()` and moves data/models to the GPU if available for faster computation.

## Data Preprocessing
- **Transforms**:
  - `transforms.ToTensor()`: Converts images to PyTorch tensors, enabling compatibility with neural network operations.
  - `transforms.Normalize((0.5,), (0.5,))`: Normalizes pixel values to the range [-1, 1] by subtracting the mean (0.5) and dividing by the standard deviation (0.5). This stabilizes and accelerates training.

- **Dataset Splitting**:
  - The full training dataset (60,000 images) is split into training (48,000 images) and validation (12,000 images) sets using `train_test_split` from `scikit-learn` with a test size of 20% and stratified sampling to maintain class distribution.
  - The test dataset contains 10,000 images.
  - DataLoaders are created with a batch size of 64 for training, validation, and testing.

## Model Architecture
- The neural network (`net`) is a custom-defined model using `torch.nn.Module`.
- Architecture details (not fully visible in the provided notebook excerpt) typically include convolutional layers, fully connected layers, and activation functions suited for image classification.
- The model is designed to output probabilities for 10 classes (digits 0-9).

## Training
- **Optimizer**: `torch.optim` (specific optimizer not shown in the excerpt, but likely Adam or SGD).
- **Loss Function**: `torch.nn.CrossEntropyLoss` (inferred from typical usage in classification tasks).
- **Epochs**: Initially trained for an unspecified number of epochs, later revised to 15 epochs with a learning rate scheduler.
- **Training Loop**:
  - Computes training and validation accuracy/loss per epoch.
  - Uses GPU if available for faster computation.
  - Stores loss history for visualization.

## Evaluation
- **Initial Test Accuracy**: 98.05% on the test set, indicating strong performance but room for improvement to approach state-of-the-art results (e.g., 99.65% from Ciresan et al., 2011).
- **Visualization**:
  - The `show5` function displays five sample images from the dataset with their labels.
  - Training and validation loss curves are plotted using `matplotlib`.

## Revisions and Improvements
To improve the model's test accuracy, the following revision was made:
- **Learning Rate Scheduler**:
  - Added a `ReduceLROnPlateau` scheduler to dynamically reduce the learning rate when the validation loss plateaus.
  - Configuration:
    - `mode='min'`: Minimizes validation loss.
    - `factor=0.1`: Reduces learning rate by a factor of 0.1.
    - `patience=5`: Waits 5 epochs for improvement before reducing the learning rate.
    - `verbose=True`: Prints a message when the learning rate is reduced.
  - **Impact**:
    - The scheduler was introduced to address potential stagnation in learning, as observed in the initial training (validation loss not decreasing significantly).
    - After retraining for 15 epochs, the test accuracy improved to **98.38%**, a modest increase from 98.05%.
    - The training and validation accuracy improved over epochs, with training accuracy reaching 99.45% and validation accuracy reaching 98.42% by epoch 15.

## Final Results
- **Final Test Accuracy**: 98.38% after incorporating the learning rate scheduler.
- **Loss Trends**:
  - Training loss decreased from 0.04420 (epoch 1) to 0.01682 (epoch 15).
  - Validation loss decreased from 0.06793 (epoch 1) to 0.06151 (epoch 15), with the learning rate reduction occurring after epoch 10.
- The model was saved as `saved_model.pth` using `torch.save(net.state_dict(), "saved_model.pth")` for future use.

## Future Improvements
- **Increase Epochs**: Training for more epochs (e.g., 20–30) could further improve accuracy.
- **Model Architecture**:
  - Experiment with deeper convolutional neural networks (CNNs) or architectures like LeNet or ResNet.
  - Add dropout layers to reduce overfitting.
- **Hyperparameter Tuning**:
  - Adjust batch size, initial learning rate, or optimizer (e.g., try AdamW or RMSprop).
  - Explore other schedulers like `StepLR` or `CosineAnnealingLR`.
- **Data Augmentation**: Apply random rotations, translations, or flips to increase dataset diversity and improve generalization.
- **Batch Normalization**: Add batch normalization layers to stabilize training and improve convergence.

## How to Run
1. Open `MNIST_Handwritten_Digits-STARTER.ipynb` in a Jupyter Notebook environment.
2. Ensure all dependencies are installed (see `requirements.txt`).
3. Run the cells sequentially, ensuring the kernel is restarted after installing dependencies.
4. To visualize sample images, use the `show5` function with a DataLoader.
5. Train the model using the provided training loop and evaluate using the test DataLoader.
6. Save the trained model for future use.
