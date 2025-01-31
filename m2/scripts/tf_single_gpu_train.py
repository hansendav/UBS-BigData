# Imports 
import os
import tensorflow as tf 
import keras 
from keras import layers
import pyarrow as pa
import pyarrow.fs as fs 
import pyarrow.parquet as pq
import pandas as pd 
import re 
import rasterio
import numpy as np 
import argparse 
import time

# import segmentation models 
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm

# Label encodings 
label_encodings = {
    111: 0, 112: 1, 121: 2, 122: 3, 123: 4, 124: 5, 131: 6, 132: 7, 133: 8, 141: 9, 142: 10,
    211: 11, 212: 12, 213: 13, 221: 14, 222: 15, 223: 16, 231: 17, 241: 18, 242: 19, 243: 20,
    244: 21, 311: 22, 312: 23, 313: 24, 321: 25, 322: 26, 323: 27, 324: 28, 331: 29, 332: 30,
    333: 31, 334: 32, 335: 33, 411: 34, 412: 35, 421: 36, 422: 37, 423: 38, 511: 39, 512: 40,
    521: 41, 522: 42, 523: 43, 999: 44
}


# Define utility classes and functions ----------------------------------------
class TimeLoggingCallback(tf.keras.callbacks.Callback):
    """
    A custom Keras callback to log the duration of each epoch during training.
    
    Methods
    -------
    on_epoch_begin(epoch, logs=None)
        Records the start time at the beginning of the epoch.
    on_epoch_end(epoch, logs=None)
        Calculates the time taken for the epoch and logs it.
    """
    
    def on_epoch_begin(self, epoch, logs=None):
        # Record the start time at the beginning of the epoch
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        # Calculate the time taken for the epoch
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - self.epoch_start_time

        # Optionally, you can log this information into logs
        logs['epoch_duration'] = epoch_duration


def get_dataframes(path=None, subsample=None):
    """
    Load a parquet file containing dataset paths, process it to extract the training and testing dataframes, 
    and optionally subsample the data.
    Args:
        path (str, optional): The path to the parquet file. If None, a default S3 path is used.
        subsample (float, optional): Fraction of the dataset to sample. If None, the entire dataset is used.
    Returns:
        tuple: A tuple containing two DataFrames (train_df, test_df).
    """
    # load parquet to get file paths 
    if path is None:
        path = 's3://ubs-cde/home/e2405193/bigdata/meta_with_image_paths.parquet'
    
    df = pq.read_table(path).to_pandas()
    print(f'Loaded parquet file from {path}')

    df = df[['split', 's1_path', 's2_path', 'label_path']]

    df['paths'] = df['s1_path'] + ' ' + df['s2_path'] + ' ' + df['label_path']
    
    #  drop paths that are not needed anymore 
    df = df.drop(columns=['s1_path', 's2_path', 'label_path'])

    df_train = df[df['split'] == 'train']
    df_test = df[df['split'] == 'test']

    if subsample is not None: 
        df_train = df_train.sample(frac=subsample, random_state=42)
        df_test = df_test.sample(frac=subsample, random_state=42)

    print(f'Train size: {len(df_train)}')
    print(f'Test size: {len(df_test)}')

    return df_train, df_test


# function to read image band from s3
def read_img_with_rasterio(file, label=False):
    """
    Reads an image file using rasterio and converts it to a TensorFlow tensor.
    Args:
        file (str): Path to the image file to be read.
        label (bool, optional): If True, the image is treated as a label and 
                                encoded using label_encodings. Defaults to False.
    Returns:
        tf.Tensor: The image data as a TensorFlow tensor with channels last and 
                   cast to float32.
    """
    with rasterio.open(file) as src:
        image = src.read()
    if label == True: 
        image = np.vectorize(label_encodings.get)(image)
        tensor = tf.convert_to_tensor(image) # convert numpy array to tensor
        tensor = tf.transpose(tensor, perm=[1, 2, 0]) # change channel pos to last
        tensor = tf.cast(tensor, tf.float32) # cast to float32 for training
    else:        
        tensor = tf.convert_to_tensor(image) # convert numpy array to tensor
        tensor = tf.transpose(tensor, perm=[1, 2, 0]) # change channel pos to last
        tensor = tf.cast(tensor, tf.float32) # cast to float32 for training
        
    return tensor # return the tensor

def get_s1_bands(input_string):
    vv = re.sub(r'([^/]+)/$', r'\1/\1_VV.tif', input_string)
    vh = re.sub(r'([^/]+)/$', r'\1/\1_VH.tif', input_string)
    return [vv, vh]

def get_s2_bands(input_string):
    b02 = re.sub(r'([^/]+)/$', r'\1/\1_B02.tif', input_string)
    b03 = re.sub(r'([^/]+)/$', r'\1/\1_B03.tif', input_string)
    b04 = re.sub(r'([^/]+)/$', r'\1/\1_B04.tif', input_string)
    b08 = re.sub(r'([^/]+)/$', r'\1/\1_B08.tif', input_string)
    return [b02, b03, b04, b08]

def get_label_band(input_string):
    return [re.sub(r'([^/]+)/$', r'\1/\1_reference_map.tif', input_string)]

def fetch_image_paths_from_s3(band_paths_string):
    band_paths_list = band_paths_string.split(' ')
    s1 = get_s1_bands(band_paths_list[0])
    s2 = get_s2_bands(band_paths_list[1])
    label = get_label_band(band_paths_list[2])

    return s1, s2, label

def build_and_compile_model(): 
    """
    Builds and compiles a U-Net model with a ResNet34 backbone.
    The model is configured with the following specifications:
    - Input shape: (None, None, 6)
    - Number of classes: 44
    - Activation function: softmax
    - Encoder weights: None (training from scratch, not using ImageNet weights)
    - Optimizer: Adam
    - Loss function: Jaccard Loss
    - Metrics: Intersection over Union (IoU) score
    - JIT compile: Disabled (set to False)
    Returns:
        keras.Model: A compiled U-Net model ready for training.
    """
    model = sm.Unet('resnet34', input_shape=(None, None, 6), classes=44, activation='softmax', encoder_weights=None) # need to train from scratch! not using iamgenet weights
    
    model.compile(
        'Adam',
        loss=sm.losses.jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )
    
    # set jit compile to False see: https://keras.io/guides/migrating_to_keras_3/
    model.jit_compile = False 

    return model 


def map_function(i):
    i = i.numpy().decode()
    s1, s2, label = fetch_image_paths_from_s3(i)
    label = tf.image.resize(read_img_with_rasterio(label[0], label=True), (128, 128)) # resize label to 128x128
    s1_bands = [tf.image.resize(read_img_with_rasterio(s1_band), (128, 128)) for s1_band in s1]
    s2_bands = [tf.image.resize(read_img_with_rasterio(s2_band), (128, 128)) for s2_band in s2]
    image = tf.concat(s1_bands + s2_bands, axis=-1)
    return image, label

def _fixup_shape(x, y):
        x.set_shape([None, None, 6])
        y.set_shape([None, None, 1])
        return x, y

def get_data_set_from_df(df, global_batch_size, strategy):
    dataset = tf.data.Dataset.from_generator(lambda: list(df['paths']), output_signature=tf.TensorSpec(shape=(), dtype=tf.string))
    dataset = dataset.map(lambda i: tf.py_function(map_function, [i], Tout=(tf.float32, tf.float32)), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(_fixup_shape, tf.data.AUTOTUNE) # to match first layer input shape
    dataset = dataset.repeat(2).batch(global_batch_size,  num_parallel_calls=tf.data.AUTOTUNE).prefetch(1).cache() # changed from number of replicas
    return dataset


def main(PATH_TO_PARQUET, SUBSAMPLE_RATIO, BATCH_SIZE, CHECKPOINT_NAME, LOGS_NAME):
    
    # clear previous sessions
    keras.backend.clear_session()
    
    # get dataframes
    df_train, df_test = get_dataframes(PATH_TO_PARQUET, SUBSAMPLE_RATIO)

    # get dataset 
    data_train = get_data_set_from_df(df_train, BATCH_SIZE, None)
    data_test = get_data_set_from_df(df_test, BATCH_SIZE, None)

    for batch in data_train.take(1):
        print(f'Batch shape: {batch[0].shape}')
        print(f'Label shape: {batch[1].shape}')
        print(f'Batch type: {type(batch[0])}')
        print(f'Label type: {type(batch[1])}')

    # Instantiate custom timelogging callback 
    time_callback = TimeLoggingCallback()

    # Instantiate checkpoint callback 

    if not os.path.exists('./model_checkpoints'):
        os.makedirs('./model_checkpoints')

    save_checkpoint_path = f'./model_checkpoints/{CHECKPOINT_NAME}.keras'
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath= save_checkpoint_path,
        save_weights_only=False,
        monitor='train_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )

    # instantiate model
    model = build_and_compile_model()

    start = time.time()
    hist = model.fit(data_train,
        epochs=2,
        #validation_data=data_test,
        verbose=1,
        #steps_per_epoch=steps_per_epoch, 
        callbacks=[time_callback, checkpoint_callback]
    )
    end = time.time()
    print(f'Training time: {end - start}')



    # save history to parquet
    save_log_path = f's3://ubs-cde/home/e2405193/bigdata/experiments/m2/logs/hists/{LOGS_NAME}.parquet'
    hist_df = pd.DataFrame(hist.history)
    hist_df = pa.Table.from_pandas(hist_df)
    pq.write_table(hist_df, save_log_path)

    print(f'History saved to {save_log_path}')
    print(f'Model saved to {save_checkpoint_path}')
    print(f'Training completed')

    print(f'Training completed')

    start = time.time()
    loss, iou = model.evaluate(data_test)#, steps=validation_steps)
    end = time.time()
    print(f'Testing time: {end - start}')

    print(f'Testing completed')
    print(f'Validation loss: {loss}')
    print(f'Validation IoU: {iou}')
    print(f'Finished training and testing')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_parquet', type=str, required=False)
    parser.add_argument('--subsample_ratio', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--checkpoint_name', type=str, required=True)
    parser.add_argument('--logs_name', type=str, required=True) 
    args = parser.parse_args()

    main(args.path_to_parquet, args.subsample_ratio, args.batch_size, args.checkpoint_name, args.logs_name)