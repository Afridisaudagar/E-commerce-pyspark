
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, count

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"

spark = (
    SparkSession.builder
    .appName("ECommerceSalesPipeline")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    # Read CSV files
    orders = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(str(DATA_DIR / "sample-orders.csv"))
    )

    customers = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(str(DATA_DIR / "sample-customers.csv"))
    )

    products = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(str(DATA_DIR / "sample-products.csv"))
    )

    print("\n--- Orders ---")
    orders.show(5, truncate=False)

    # Data cleaning
    orders = orders.dropna(
        subset=[
            "order_id", "customer_id", "product",
            "quantity", "total", "date"
        ]
    )

    orders = orders.filter(
        (col("quantity") > 0) & (col("total") >= 0)
    )

    # Check customer and product CSV column names
    print("Customer columns:", customers.columns)
    print("Product columns:", products.columns)

    # Join customers
    sales = orders.join(
        customers, on="customer_id", how="inner"
    )

    # Join products only if both datasets have a product column
    if "product" in products.columns:
        sales = sales.join(products, on="product", how="left")
    else:
        print("WARNING: products.csv has no 'product' column.")
        print("Skipping product lookup join.")

    # In this dataset, total already represents order value
    sales = sales.withColumnRenamed("total", "revenue")

    print("\n--- Processed Sales ---")
    sales.show(10, truncate=False)

    # City-wise report
    city_report = (
        sales.groupBy("city")
        .agg(
            spark_sum("revenue").alias("total_revenue"),
            count("order_id").alias("total_orders")
        )
        .orderBy(col("total_revenue").desc())
    )

    print("\n--- City-wise Sales ---")
    city_report.show(truncate=False)

    # Product-wise report
    product_report = (
        sales.groupBy("product")
        .agg(
            spark_sum("quantity").alias("units_sold"),
            spark_sum("revenue").alias("total_revenue")
        )
        .orderBy(col("total_revenue").desc())
    )

    print("\n--- Product-wise Sales ---")
    product_report.show(truncate=False)

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sales.write.mode("overwrite").parquet(
        str(OUTPUT_DIR / "processed_sales")
    )

    city_report.write.mode("overwrite").option(
        "header", True
    ).csv(str(OUTPUT_DIR / "city_report"))

    product_report.write.mode("overwrite").option(
        "header", True
    ).csv(str(OUTPUT_DIR / "product_report"))

    print("\nPipeline completed successfully!")
    print("Output folder:", OUTPUT_DIR)

finally:
    spark.stop()
