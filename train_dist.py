# %%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import tensorflow as tf
import argparse

from models.model import SemViT
from utils.datasets import dataset_generator

def main(args):
  if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

  # Load CIFAR-10 dataset
  train_ds, test_ds = prepare_dataset()

  EXPERIMENT_NAME = args.experiment_name
  print(f'Running {EXPERIMENT_NAME}')

  # strategy = tf.distribute.MultiWorkerMirroredStrategy()
  # with strategy.scope():

  model = SemViT(
    args.block_types,
    args.filters,
    args.repetitions,
    has_gdn=args.gdn,
    num_symbols=args.data_size,
    snrdB=args.train_snrdB,
    channel=args.channel_types
  )
  # model = DeepJSCC(
  #   snrdB=args.train_snrdB,
  #   channel=args.channel_types
  # )

  def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1)
  
  model.compile(
      loss='mse',
      optimizer=tf.keras.optimizers.legacy.Adam(
          learning_rate=1e-4
      ),
      metrics=[
          psnr
      ]
  )

    # Add this after model creation and before model.build()
  try:
      model.build(input_shape=(None, 512, 512, 3))
      print("Model built successfully!")
      model.summary()
  except Exception as e:
      print(f"Error building model: {e}")
      print("Trying with concrete batch size...")
      # Try building with concrete batch size first
      model.build(input_shape=(1, 512, 512, 3))
      print("Model built with concrete batch size!")
      model.summary()

  # Also add some debugging for your dataset
  def debug_dataset(ds, name):
      print(f"\n=== {name} Dataset Debug ===")
      for batch in ds.take(1):
          if isinstance(batch, tuple):
              x, y = batch
              print(f"Input shape: {x.shape}, Output shape: {y.shape}")
              print(f"Input dtype: {x.dtype}, Output dtype: {y.dtype}")
              print(f"Input range: [{tf.reduce_min(x):.3f}, {tf.reduce_max(x):.3f}]")
          else:
              print(f"Batch shape: {batch.shape}")
      print("=" * 30)

  # Add this after prepare_dataset() in main function:
  debug_dataset(train_ds, "Training")
  debug_dataset(test_ds, "Test")

  model.build(input_shape=(None, 512, 512, 3))
  model.summary()
  
  if args.ckpt is not None:
    model.load_weights(args.ckpt)

  save_ckpt = [
      tf.keras.callbacks.ModelCheckpoint(
          filepath=f"./ckpt/{EXPERIMENT_NAME}_" + "{epoch}",
          save_best_only=True,
          monitor="val_loss",
          save_weights_only=True,
          options=tf.train.CheckpointOptions(
              experimental_io_device=None, experimental_enable_async_checkpoint=True
          )
      )
  ]

  tensorboard = tf.keras.callbacks.TensorBoard(log_dir=f'logs/{EXPERIMENT_NAME}')
  history = model.fit(
      train_ds,
      initial_epoch=args.initial_epoch,
      epochs=args.epochs,
      callbacks=[tensorboard, save_ckpt],
      validation_data=test_ds,
  )

  model.save_weights(f"{EXPERIMENT_NAME}_" + f"{args.epochs}")


def prepare_dataset():
  AUTO = tf.data.experimental.AUTOTUNE
  test_ds = dataset_generator('/content/dataset/' , target_size=(512 , 512))
  train_ds = dataset_generator('/content/dataset/' , target_size=(512,512)).cache()

  normalize = tf.keras.layers.Rescaling(1./255)
  augment_layer = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.RandomFlip("horizontal"),
  ])

  def normalize_and_augment(image, training):
    image = augment_layer(image, training=training)
    return image

  train_ds = (
    train_ds.shuffle(50000, reshuffle_each_iteration=True)
            .map(lambda x, y: (normalize_and_augment(x, training=True), y), num_parallel_calls=AUTO)
            .map(lambda x, _: (x, x))
            .prefetch(AUTO)
  )
  test_ds = (
    test_ds.map(lambda x, y: (normalize(x), y))
           .map(lambda x, _: (x, x))
           .cache()
           .prefetch(AUTO)
  )

  return train_ds, test_ds


if __name__ == "__main__":
  class Args:
    data_size = 8192
    channel_types = 'AWGN'
    train_snrdB = 15
    block_types = 'CCVVCC'
    experiment_name = 'SemViT_512_run'
    epochs = 300
    filters = [256, 256, 256, 256, 256, 256]
    repetitions = [1, 1, 3, 3, 1, 1]
    gdn = False
    initial_epoch = 0
    ckpt = None
    gpu = '0'

  args = Args()
  main(args)

