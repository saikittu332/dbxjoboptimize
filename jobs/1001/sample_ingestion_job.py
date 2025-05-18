from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, current_timestamp, lit
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ingestion-job')

def read_source_data(spark, source_path, format_type="csv"):
    """Read data from source location."""
    logger.info(f"Reading data from {source_path}")
    
    # This could be optimized for large datasets
    if format_type == "csv":
        return spark.read.option("header", "true").csv(source_path)
    elif format_type == "parquet":
        return spark.read.parquet(source_path)
    elif format_type == "json":
        return spark.read.json(source_path)
    else:
        raise ValueError(f"Unsupported format: {format_type}")

def transform_data(df):
    """Apply transformations to the data."""
    logger.info("Applying transformations")
    
    # Example transformations - these could be optimized
    transformed_df = df \
        .withColumn("processed_timestamp", current_timestamp()) \
        .withColumn("data_date", to_timestamp(col("date"), "yyyy-MM-dd"))
    
    # Simulate a non-optimized transformation that could be improved
    # This creates a cross join which is very inefficient
    df_small = df.limit(10).select("region")
    
    # This cross join is deliberately inefficient for demo purposes
    # The optimizer should suggest changing this to a broadcast join
    result_df = transformed_df.crossJoin(df_small)
    
    # More transformations
    # Inefficient UDF - could be replaced with built-in functions
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType
    
    @udf(returnType=StringType())
    def capitalize_string(s):
        if s is None:
            return None
        return s.upper()
    
    # Using a UDF instead of the more efficient built-in upper() function
    result_df = result_df.withColumn("CATEGORY_UPPERCASE", capitalize_string(col("category")))
    
    # Inefficient way to filter data - could be pushed down to the data source
    result_df = result_df.filter(col("value") > 100)
    
    return result_df

def write_data(df, target_path, mode="overwrite"):
    """Write data to target location."""
    logger.info(f"Writing data to {target_path}")
    
    # Writing with poor partitioning scheme - optimizer should catch this
    df.write.mode(mode).parquet(target_path)

def run_ingestion_job(source_path, target_path):
    """Main function to run the ingestion job."""
    logger.info("Starting ingestion job")
    
    # Create Spark session
    spark = SparkSession.builder \
        .appName("SampleIngestionJob") \
        .getOrCreate()
    
    try:
        # Set log level to minimize verbose output
        spark.sparkContext.setLogLevel("WARN")
        
        # Read data
        source_df = read_source_data(spark, source_path)
        
        # Transform data
        transformed_df = transform_data(source_df)
        
        # Unnecessary cache - data is used only once
        transformed_df.cache()
        
        # Write data - poor partitioning
        write_data(transformed_df, target_path)
        
        # Count is forcing evaluation but result isn't used
        row_count = transformed_df.count()
        logger.info(f"Processed {row_count} rows")
        
        # Unnecessary second action
        distinct_count = transformed_df.select("category").distinct().count()
        logger.info(f"Found {distinct_count} distinct categories")
        
        logger.info("Ingestion job completed successfully")
        
    except Exception as e:
        logger.error(f"Error in ingestion job: {e}")
        raise
    finally:
        # Stop SparkSession
        spark.stop()

if __name__ == "__main__":
    # For local testing
    run_ingestion_job("./data/sales_data.csv", "./data/processed/sales")
