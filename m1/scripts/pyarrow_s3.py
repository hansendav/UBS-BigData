# Create custom metadata.parquet file for BigEarthNetv2 dataset
# Author: David Hansen 
#------------------------------------------------------------------------------
# Description:
# This script reads the metadata.parquet file from the BigEarthNetv2 dataset
# and adds the paths to the S1, S2 and label images per patch.
# The updated metadata is then saved to a new parquet file in S3
# Usage: 
# Before execution make sure to have the following environment variables set:
# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN
# Adapth the bucket and parquet file paths to the respective locations in S3
# Run the script with the following command:
# python scripts/pyarrow_s3.py 


#------------------------------------------------------------------------------
# BEGINNING OF THE SCRIPT
#------------------------------------------------------------------------------
# Import libraries
import pyarrow as pa
import pyarrow.fs as fs
import pyarrow.parquet as pq
import pandas as pd
import re 

# Set up S3 
s3 = fs.S3FileSystem()
bigearthnet_bucket = "s3://ubs-datasets/bigearthnet/"


# Read parquet file to pandas df
parquet_path = "s3://ubs-datasets/bigearthnet/metadata.parquet"
meta = pq.read_table(parquet_path).to_pandas()

meta['patch_id_path'] = meta['patch_id'].apply(lambda x: re.match(r'(.*)_[0-9]+_[0-9]+$', x).group(1))
meta['patch_id_path_s1'] = meta['s1_name'].apply(lambda x: re.match(r'(.*)_[A-Z0-9]+_[0-9]+_[0-9]+$', x).group(1))

meta['s1_path'] = bigearthnet_bucket + 'BigEarthNet-S1/'  + meta['patch_id_path_s1'] + '/' + meta['s1_name'] + '/'
meta['s2_path'] = bigearthnet_bucket + 'BigEarthNet-S2/'  + meta['patch_id_path'] + '/' + meta['patch_id'] +  '/' 
meta['label_path'] = bigearthnet_bucket + 'Reference_Maps/' + meta['patch_id_path'] + '/' + meta['patch_id'] + '/'

# Write to S3 
table = pa.Table.from_pandas(meta)
pq.write_table(table, 'ubs-cde/home/e2405193/bigdata/meta_with_image_paths.parquet', filesystem=s3)

print(
        f"Saved updated meta.parquet to:"
        f"\n s3://ubs-cde/home/e2405193/bigdata/meta_with_image_paths.parquet"
)