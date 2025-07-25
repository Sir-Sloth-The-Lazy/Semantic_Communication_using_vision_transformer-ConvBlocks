import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import numpy as np
from PIL import Image

from models.model import SemViT
from utils.datasets import dataset_generator

def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def prepare_dataset():
    AUTO = tf.data.experimental.AUTOTUNE
    test_ds = dataset_generator('/dataset/CIFAR10/test/')  # <-- Ensure this dir has 512×512-ready data

    normalize = tf.keras.layers.Rescaling(1./255)

    test_ds = (
        test_ds.map(lambda x, y: (normalize(x), y))
               .map(lambda x, _: (x, x))  # identity target
               .cache()
               .prefetch(AUTO)
    )
    return test_ds

def float32_to_rgb_manual(arr):
    h, w, c = arr.shape
    result = []
    for i in range(h):
        row = []
        for j in range(w):
            pixel = []
            for k in range(c):
                val = arr[i][j][k]
                int_val = int(min(max(val * 255, 0), 255))
                pixel.append(int_val)
            row.append(pixel)
        result.append(row)
    return np.array(result, dtype='uint8')

def save_reconstructed_images(model, test_ds, output_dir, limit=100):
    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for batch in test_ds:
        images, _ = batch
        preds = model.predict(images, verbose=0)

        for i in range(images.shape[0]):
            if count >= limit:
                return
            original = images[i].numpy()
            recon = preds[i]

            original_img = float32_to_rgb_manual(original)
            recon_img = float32_to_rgb_manual(recon)

            Image.fromarray(original_img).save(f"{output_dir}/original_{count:03d}.png")
            Image.fromarray(recon_img).save(f"{output_dir}/recon_{count:03d}.png")
            count += 1

def main():
    block_types = "CCVVCC"
    filters = [256, 256, 256, 256, 256, 256]
    repetitions = [1, 1, 3, 3, 1, 1]
    gdn = False
    data_size = 2048  # more symbols for larger images
    snrdB = 15
    channel = "AWGN"
    experiment_name = "CCVVCC_15dB"
    epoch_to_test = 250
    recon_dir = f"reconstructions/epoch_{epoch_to_test}"

    model = SemViT(
        block_types=block_types,
        filters=filters,
        repetitions=repetitions,
        has_gdn=gdn,
        num_symbols=data_size,
        snrdB=snrdB,
        channel=channel
    )

    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=[psnr]
    )

    model.build(input_shape=(None, 512, 512, 3))
    model.summary()

    ckpt_path = f"./ckpt/{experiment_name}_{epoch_to_test}"
    print(f"🔍 Loading checkpoint: {ckpt_path}")
    model.load_weights(ckpt_path)

    test_ds = prepare_dataset()

    loss, test_psnr = model.evaluate(test_ds)
    print(f"✅ Test PSNR (Epoch {epoch_to_test}): {test_psnr:.2f}")

    print(f"📸 Saving reconstructed images to: {recon_dir}")
    save_reconstructed_images(model, test_ds, recon_dir, limit=100)

if __name__ == "__main__":
    main()
