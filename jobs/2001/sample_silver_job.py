from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, desc, date_format, count, when, lit
from pyspark.sql.window import Window
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('silver-job')

def read_bronze_data(spark, bronze_path):
    """Read data from bronze layer."""
    logger.info(f"Reading bronze data from {bronze_path}")
    # Read all parquet files - not optimized for large datasets 
    return spark.read.parquet(bronze_path)

def enrich_with_region_data(spark, df, region_path):
    """Enrich data with region information."""
    logger.info(f"Enriching with region data from {region_path}")
    
    # Read region dimension table
    region_df = spark.read.option("header", "true").csv(region_path)
    
    # Inefficient join - no broadcast hint despite small dimension table
    enriched_df = df.join(
        region_df,
        df["region_id"] == region_df["id"],
        "left"
    )
    
    return enriched_df

def aggregate_data(df):
    """Create aggregated views of the data."""
    logger.info("Creating aggregations")
    
    # Inefficient - doesn't repartition by key before aggregation
    daily_agg = df.groupBy("data_date", "region") \
                  .agg(
                      sum("value").alias("total_value"),
                      avg("value").alias("avg_value"),
                      count("*").alias("transaction_count")
                  )
    
    # Multiple expensive sorts 
    daily_agg = daily_agg.orderBy(desc("total_value"))
    
    # Inefficient window function - no partitioning hint
    window_spec = Window.orderBy("data_date")
    
    daily_agg = daily_agg.withColumn(
        "running_total", 
        sum("total_value").over(window_spec)
    )
    
    # Unnecessarily computing multiple aggregations separately
    # Could be combined in a single operation
    category_agg = df.groupBy("category").count()
    category_sum = df.groupBy("category").agg(sum("value").alias("category_total"))
    
    # Another inefficient join without broadcast
    category_metrics = category_agg.join(category_sum, "category")
    
    return {
        "daily_agg": daily_agg,
        "category_metrics": category_metrics
    }

def apply_business_rules(df):
    """Apply business rules to the data."""
    logger.info("Applying business rules")
    
    # Inefficient multiple filter operations - could be combined
    high_value = df.filter(col("value") > 1000)
    priority_regions = df.filter(col("region").isin("NORTH", "SOUTH"))
    
    # Calculating the same expression multiple times
    df = df.withColumn("tax_rate", 
                      when(col("region") == "NORTH", 0.08)
                      .when(col("region") == "SOUTH", 0.07)
                      .when(col("region") == "EAST", 0.09)
                      .when(col("region") == "WEST", 0.095)
                      .otherwise(0.10))
    
    df = df.withColumn("tax_amount", col("value") * col("tax_rate"))
    df = df.withColumn("value_with_tax", col("value") + col("tax_amount"))
    
    # Inefficient string formatting
    df = df.withColumn("month", date_format(col("data_date"), "MMMM"))
    df = df.withColumn("year", date_format(col("data_date"), "yyyy"))
    
    # Simulating a slow operation by adding a delay
    # This is just for demo purposes to show a performance bottleneck
    time.sleep(2)
    
    return df

def write_silver_data(df, target_paths):
    """Write data to silver layer."""
    logger.info(f"Writing silver data")
    
    # Write main dataset - poor partitioning strategy
    df.write.mode("overwrite").parquet(target_paths["main"])
    
    # Inefficient - reads data back that was just processed
    aggregations = aggregate_data(df)
    
    # Write aggregations
    aggregations["daily_agg"].write.mode("overwrite").parquet(target_paths["daily_agg"])
    aggregations["category_metrics"].write.mode("overwrite").parquet(target_paths["category_metrics"])
    
    # Unnecessary repartition before writing
    df.repartition(1).write.mode("overwrite").parquet(target_paths["single_file"])
    
    return target_paths

def run_silver_job(bronze_path, region_path, target_paths):
    """Main function to run the silver job."""
    logger.info("Starting silver job")
    
    # Create Spark session with suboptimal configuration
    spark = SparkSession.builder \
        .appName("SampleSilverJob") \
        .config("spark.sql.shuffle.partitions", "200")  # Default value, not optimized
        .getOrCreate()
    
    try:
        # Read bronze data
        bronze_df = read_bronze_data(spark, bronze_path)
        
        # Unnecessary persistence at default storage level
        bronze_df.cache()
        bronze_df.count()  # Force caching
        
        # Enrich data
        enriched_df = enrich_with_region_data(spark, bronze_df, region_path)
        
        # Apply business rules
        silver_df = apply_business_rules(enriched_df)
        
        # Write silver data
        result_paths = write_silver_data(silver_df, target_paths)
        
        logger.info("Silver job completed successfully")
        return result_paths
        
    except Exception as e:
        logger.error(f"Error in silver job: {e}")
        raise
    finally:
        # Stop SparkSession
        spark.stop()

if __name__ == "__main__":
    # For local testing
    target_paths = {
        "main": "./data/silver/main",
        "daily_agg": "./data/silver/daily_agg",
        "category_metrics": "./data/silver/category_metrics",
        "single_file": "./data/silver/single_file"
    }
    
    run_silver_job(
        "./data/processed/sales",
        "./data/reference/regions.csv",
        target_paths
    )
