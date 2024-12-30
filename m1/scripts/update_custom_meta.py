from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyarrow as pa 
import pyarrow.parquet as pq
import pyspark.sql.functions as f
import pyarrow.fs as fs


def create_spark_session(master_url, app_name, cores):
    """
    Create and return a Spark session with necessary configurations.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .master(master_url) \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
    return spark

def create_filesystem():
    return fs.S3FileSystem()

def read_metadata(spark, metadata_file):
    """
    Read metadata file and return a Spark DataFrame.
    """
    schema = StructType([
        StructField('patch_id', StringType(), True),
        StructField('labels', StringType(), True),
        StructField('split', StringType(), True),
        StructField('country', StringType(), True),
        StructField('s1_name', StringType(), True),
        StructField('s2v1_name', StringType(), True),
        StructField('contains_seasonal_snow', BooleanType(), True),
        StructField('contains_cloud_or_shadow', BooleanType(), True),
        StructField('patch_id_path', StringType(), True),
        StructField('patch_id_path_s1', StringType(), True),
        StructField('s1_path', StringType(), True),
        StructField('s2_path', StringType(), True),
        StructField('label_path', StringType(), True)
    ])


    meta = pq.read_table(metadata_file).to_pandas()
    meta = spark.createDataFrame(meta, schema=schema)
    return meta

def process_metadata(spark, meta_df):
    meta = meta_df \
        .withColumn('s1_path', f.split(f.col('s1_path'), 's3://').getItem(1)) \
        .withColumn('s2_path', f.split(f.col('s2_path'), 's3://').getItem(1)) \
        .withColumn('label_path', f.split(f.col('label_path'), 's3://').getItem(1)) \
        .withColumn(
            'patch_path_array',
            f.array(
                f.col('s1_path'),
                f.col('s2_path'),
                f.col('label_path'),
                f.col('patch_id')
            )
        )\
        .select('patch_id', 'split', 'patch_path_array')
    return meta

def write_metadata(spark, filesystem, meta_df, output_path):
    """
    Write metadata to a parquet file.
    """
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
    meta = meta_df.toPandas()
    table = pa.Table.from_pandas(pandas_df)
    pq.write_table(meta_df, output_path, filesystem=filesystem)


def main():
    master_url = "local[*]"
    app_name = "MetadataFileProcessing"
    cores = "4"
    metadata_file = 's3://ubs-cde/home/e2405193/bigdata/meta_with_image_paths.parquet'

    spark = create_spark_session(master_url, app_name, cores)
    s3 = create_filesystem()
    meta = read_metadata(spark, metadata_file)
    print(meta.rdd.getNumPartitions())
    meta = process_metadata(spark, meta)
    print(meta.rdd.getNumPartitions())

    output_path = 'ubs-cde/home/e2405193/bigdata/meta_for_preprocessing.parquet'
    write_metadata(spark, s3, meta, output_path)

if __name__ == "__main__":
    main()
