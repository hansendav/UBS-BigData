# -----------------------------------------------------------------------------
# ### SCRIPT: CONCATENATE PATCH BANDS - BIGEARTHNETV2 -> SAVE TO S3 BUCKET 
# ### UBS-BigData Class M2: Geodata Science 
# ### Subproject: M2 - Distributed DeepLearning on BigEarthNetv2 
# -----------------------------------------------------------------------------
# Author: David Hansen 
# Date: 2025 - 01 - 08 
# Description: This script is used prior the principle task to concatenate the
# image bands to a single array prior training using tensorflow. It uses the 
# custom metadata parquet file (from M1; to befind on S3://ubs-cde/home/e2405193/bigdata/) 
# and materializeses all patches in the training and testsplit of BigEarthNetv2  
# with together with their labels on S3 (s3://ubs-cde/home/e2405193/bigdata/m2/dataset/)
# This script uses PySpark and should be executed with the bashfile 
# exec_concatenate_patch_bands.sh to be easily adaptable to the given processing 
# environment 
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# ### Libraries import 
# -----------------------------------------------------------------------------
from pyspark.sql import SparkSession 
from pyspark.sql.types import * # import all datatypes
import pyarrow.fs as fs # used to list files on S3
from pyspark.sql.functions import col, udf # need python-UDF
from pyspark.sql import functions as f
import rasterio # read bands as .tif
import re 
import numpy as np
import argparse
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# ### Wrapper functions 
# -----------------------------------------------------------------------------
def log_runtime(task_name):
    """
    Decorator to log the runtime of any given wrapped function. 
    Logs and prints the runtime of the wrapped function to the stdout. 
    Times are logged in local time and the format '%H:%M:%S'.
    
    Args: 
        task_name (str): Name of task used for printing statement when logging. 
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_time_formatted = time.strftime("%H:%M:%S", time.localtime(start_time))
            print(f"{task_name} started at {start_time_formatted}")
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_time_formatted = time.strftime("%H:%M:%S", time.localtime(end_time))
            print(f"{task_name} finished at {end_time_formatted}")
            
            runtime = end_time - start_time  # Runtime calculation in seconds 
            hours, rem = divmod(runtime, 3600) # convert seconds to hours and remaining
            minutes, seconds = divmod(rem, 60) # convert remaining seconds to minutes and seconds 
            runtime_formatted = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
            print(f"Runtime of task {task_name}: {runtime_formatted}")
            
            return result 
        return wrapper
    return  decorator


# -----------------------------------------------------------------------------
# ### Custom PySpark Schemas -> move this to an external file 
# -----------------------------------------------------------------------------
band_schema = StructType(StructField('band_data', BinaryType(), nullable=False))
# -----------------------------------------------------------------------------
# ### Functions 
# -----------------------------------------------------------------------------
def prepare_cu_metadata(metadata):
    """
    Function to prepare the custom metadata dataframe for the BigEarthNetv2 
    dataset.Uses as input cols the paths to the respective S1, S2 and label directories. 
    Removes the 's3://' part of the patch paths.Returns the metadata dataframe 
    with an additional column that holds the paths 
    to all patch related images as an array.

    Args: 
        metadata (pyspark.sql.DataFrame): Metadata dataframe of the CUSTOM 
                                          BigEarthNetv2 dataframe. Thus, needs 
                                          have the s1_path, s2_path and label_path
                                          columns.
    Returns: 
        metadata (pypsark.sql.DataFrame): DataFrame with updated path columns
                                          and one additonal array column holding
                                          the paths to all subbands of the patch.
    """
    metadata = metadata \
    .withColumn('s1_path', f.split(f.col('s1_path'), 's3://').getItem(1))\
    .withColumn('s2_path', f.split(f.col('s2_path'), 's3://').getItem(1))\
    .withColumn('label_path', f.split(f.col('label_path'), 's3://').getItem(1))\
    .withColumn( # return column with array that holds all the changed paths.
        'paths_array',
        f.array(
            f.col('s1_path'),
            f.col('s2_path'),
            f.col('label_path')
        )
    )

    return metadata


def get_paths_from_meta(patch_path_array):
    """
    Extracts the respective paths to the S1, S2 and label bands for each a given
    patch. 

    Args: 
        patch_path_array (array): Array holding the path to the S1, S2 and 
                                  label bands.
    
    Returns: 
        s1_path, s2_path, label_path (str): Paths to the patch respective bands
                                            on the storage system (S3).

    """
    s1_path = patch_path_array[0]
    s2_path = patch_path_array[1]
    label_path = patch_path_array[2]

    return s1_path, s2_path, label_path


def get_band_paths(patch_path, is_s2=False):
    """
    Extracts image band paths from a given directory path. 
    ---
    Example: Input: path to S2 directory holding all band tifs 
    Output: list of paths to all bands
    """

    # Uses pyarrow here to get list of files in the S3 directories
    filesystem = fs.S3FileSystem()

    if is_s2 == True: # different for s2 bands -> only B02, B03, B04, B08
        files_info = filesystem.get_file_info(fs.FileSelector(patch_path, recursive=True))
        file_paths = ['s3://' + file.path for file in files_info if file.is_file and re.search(r'_B(0[2348]).tif$', file.path)]
    else: 
        files_info = filesystem.get_file_info(fs.FileSelector(patch_path, recursive=True))
        file_paths = ['s3://' + file.path for file in files_info if file.is_file] 
    
    return file_paths


def read_band(band_path): 
    """Read a single band from the given path."""
    try:
        with rasterio.open(band_path) as src:
            band = src.read()
        return band
    except Exception as e:
        logger.error(f"Error reading band from path {band_path}: {e}")
        return None

def read_bands(band_paths):
    """Read multiple bands from the given paths."""
    bands = []
    for band_path in band_paths:
        band = read_band(band_path)
        if band is not None:
            bands.append(band)
        else:
            logger.warning(f"Skipping band at path {band_path} due to read error.")
    return bands

def concatenate_patch_bands(patch_path_array): 
    """Concatenate image bands from the given patch paths."""
    try:
        # get paths to band directories
        s1_paths, s2_paths, _ = get_paths_from_meta(patch_path_array)

        # get band paths from band directories
        s2_band_paths = get_band_paths(s2_paths, is_s2=True)
        s1_band_paths = get_band_paths(s1_paths)
        
        # read needed image bands 
        patch_bands_s2 = read_bands(s2_band_paths)
        patch_bands_s1 = read_bands(s1_band_paths)

        # returned stacked array of input bands
        patch_bands_conc = np.concatenate([patch_bands_s1, patch_bands_s2], axis=0)
        return patch_bands_conc.tobytes()
    except Exception as e:
        logger.error(f"Error concatenating bands for patch {patch_path_array}: {e}")
        return None

def get_patch_label(patch_path_array):
    """Get the label for the given patch paths."""
    try:
        _, _, label_path = get_paths_from_meta(patch_path_array)
        label_band_paths = get_band_paths(label_path)
        patch_label = read_bands(label_band_paths)

        return patch_label[0].tobytes() if patch_label else None
    except Exception as e:
        logger.error(f"Error getting label for patch {patch_path_array}: {e}")
        return None


# -----------------------------------------------------------------------------
# ### Set UDFs 
# -----------------------------------------------------------------------------
# Create python User defined functions based on the functions 
# created above
create_patch_band_array = udf(concatenate_patch_bands, band_schema)
create_patch_label_array = udf(get_patch_label, band_schema)


# -----------------------------------------------------------------------------
# ### MAIN DEFINITION
# -----------------------------------------------------------------------------
@log_runtime('Main - Preprocessing M2') 
def main(partitions, output_path): 
    # Create spark session - all other configurations in the bashfile  
    spark = SparkSession.builder\
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")\
        .getOrCreate()
    logger.info('Spark session created')
    
    #create schema to only read what you need 
    meta_schema = StructType([
        StructField('patch_id', StringType(), False),
        StructField('split', StringType(), False),
        StructField('labels', ArrayType(StringType()), False),
        StructField('s1_path', StringType(), False),
        StructField('s2_path', StringType(), False),
        StructField('label_path', StringType(), False)
    ])

    # read CUSTOM metadata parquet file from S3
    meta = spark.read\
        .schema(meta_schema)\
            .parquet('s3://ubs-cde/home/e2405193/bigdata/meta_with_image_paths.parquet')\
            .filter(f.col('split').isin(['train', 'test']))

    meta = meta.limit(1)
    meta.show(1)

    # Repartition meta-parquet before processing
    # based on calculation num_executors*cores_per_executor*3
    meta = meta.repartition(partitions)

    # prepare metadata paths 
    meta = prepare_cu_metadata(meta) # adapt directory paths to be used with pyarrow
    # only keep necessary columns
    meta = meta.select('patch_id', 'split', 'labels', 'paths_array') 

    meta.show(1)

    # get image array 
    meta = meta.withColumn('bands', create_patch_band_array('paths_array'))\
        .withColumn('label_map', create_patch_label_array('paths_array'))\
        .select('patch_id', 'split', 'bands', 'label_map')
    
    meta.printSchema()
    meta.show(1)
    
    
    #write metadata df as parquet on S3 
    meta.write.mode('overwrite').parquet(output_path)

    spark.stop()
    logger.info('Spark session stopped')
# -----------------------------------------------------------------------------
# ### RUN MAIN 
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--partitions',
        type=int,
        required=True,
        help='Number of partitions for repartitioning the metadataframe.'
        )
    parser.add_argument(
        '--output_path',
        type=str, 
        required=True, 
        help='String of path to store the resulting parquet file (on S3).'
    )
    args = parser.parse_args()

    main(args.partitions, args.output_path)
# -----------------------------------------------------------------------------
# ### END OF SCRIPT