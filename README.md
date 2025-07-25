
# Semantic Communications with ViT and CNN

This repository contains the code for the paper "On the Role of ViT and CNN in Semantic Communications: Analysis and Prototype Validation" (IEEE Access, 2023). The model proposed in the paper has been adapted and enhanced to handle larger input image sizes, specifically 512x512, instead of the original 32x32. This modification increases the model's complexity and improves its robustness for handling high-resolution data.

## Key Differences:

- **Input Image Size**: The original model supported 32x32 images. In this version, the model has been upgraded to support 512x512 images, improving its performance in semantic communications tasks.
  
- **Increased Model Size**: The model has been scaled up from 52.7 MB (13.8 million parameters) to 270.5 MB (70 million parameters), significantly enhancing its learning capacity and enabling it to process more complex data.
  
- **Semantic Communication**: The model combines Vision Transformers (ViT) and Convolutional Neural Networks (CNN) for tasks in semantic communications, offering a robust framework for real-world applications.

## Requirements

Before running the code, ensure that the following dependencies are installed:

- Python 3.8 or higher
- TensorFlow 2.x (or compatible versions)
- tensorflow_compression (for GDN layers)
- numpy
- other necessary libraries as listed in `requirements.txt`

## Setup Instructions

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-repository-link.git
   cd your-repository-folder


2. **Install the dependencies**:

   Ensure you have Python 3.8+ installed. Then, install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

The model expects high-resolution images (512x512). You can use any dataset, but ensure the following format:

* **Training Data**: A directory with images and corresponding labels.
* **Test Data**: A separate directory for testing the model.

## Training the Model

To train the model, run the following command:

```bash
python train_dist.py --experiment_name "SemViT_512_run" --epochs 300 --gpu "0"
```

### Training Arguments:

* **experiment\_name**: The name of your training experiment (e.g., `SemViT_512_run`).
* **epochs**: Number of training epochs (default is 300).
* **gpu**: GPU device ID to use for training (if applicable). For example, `--gpu "0"` uses the first GPU.

The model will start training and save checkpoints periodically.

## Evaluation

Once the model is trained, you can evaluate its performance using the following:

```bash
python evaluate.py --checkpoint_path "path/to/checkpoint"
```

This will load the checkpoint and perform evaluation on the test dataset, displaying metrics like PSNR (Peak Signal-to-Noise Ratio) and other relevant performance measures.

## Model Architecture

The model combines Vision Transformers (ViT) and Convolutional Neural Networks (CNN) to create a robust framework for semantic communication tasks. Here's a breakdown of the architecture:

* **Encoder**: Utilizes blocks of ViT layers to extract spatial features from the input image.

* **Channel Layer**: Processes the data through different communication channels (AWGN, Rayleigh, Rician).

* **Decoder**: Reconstructs the processed features back into the final output using a combination of ViT and CNN layers.

The model architecture is flexible, allowing for different configurations of block types, filters, and repetitions.

## Results

You can track the training progress using TensorBoard:

```bash
tensorboard --logdir=logs/
```

This will launch a web interface to visualize metrics such as loss, accuracy, and PSNR over the course of training.

## References

Yoo, Hanju, Dai, Linglong, Kim, Songkuk, and Chae, Chan-Byoung, "On the Role of ViT and CNN in Semantic Communications: Analysis and Prototype Validation," *IEEE Access*, vol. 11, 2023, pp. 71528-71541. DOI: [10.1109/ACCESS.2023.3291405](https://doi.org/10.1109/ACCESS.2023.3291405)


