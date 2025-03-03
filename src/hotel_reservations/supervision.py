"""Monitoring module."""

import time

from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import MonitorInferenceLog, MonitorInferenceLogProblemType
from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from hotel_reservations.feature_lookup_model import FeatureLookUpModel
from hotel_reservations.serving import FeatureLookupServing
from hotel_reservations.utility import is_databricks

if is_databricks():
    spark = SparkSession.builder.getOrCreate()


class MonitoringManager:
    """A class for monitoring and interacting with model serving endpoints."""

    def __init__(self, model: FeatureLookUpModel, serving: FeatureLookupServing) -> None:
        """Initialize the Monitor with a model and serving object.

        :param model: The model to be monitored
        :param serving: The serving object to interact with endpoints
        """
        self.model = model
        self.serving = serving
        self.inference_table_fullname = (
            f"{self.model.catalog_name}.{self.model.schema_name}.`hotel-reservations-model-fe-serving-dev_payload`"
        )
        self.monitoring_table_fullname = f"{self.model.catalog_name}.{self.model.schema_name}.model_monitoring"

    def create_or_refresh_monitoring(self) -> None:
        """Manage the creation and refreshing of monitoring tables and quality monitors."""
        # Create or refresh the Lakehouse monitoring table
        inference_data = self._process_inference_data()
        self._create_monitor_table(inference_data=inference_data)

        try:
            self.serving.workspace.quality_monitors.get(self.monitoring_table_fullname)
            self.serving.workspace.quality_monitors.run_refresh(table_name=self.monitoring_table_fullname)
            logger.info("Lakehouse monitoring table exist, refreshing.")
        except NotFound:
            self._create_quality_monitor()
            self._verify_quality_monitor()
            logger.info("Lakehouse monitoring table is created.")

    def _process_inference_data(self) -> DataFrame:
        inference_table = spark.sql(f"SELECT * FROM {self.inference_table_fullname}")
        inference_table.display()

        request_schema = StructType(
            [
                StructField(
                    "dataframe_records",
                    ArrayType(
                        StructType(
                            [
                                StructField("booking_id", StringType(), False),
                                StructField("no_of_adults", IntegerType(), False),
                                StructField("no_of_children", IntegerType(), False),
                                StructField("no_of_weekend_nights", IntegerType(), False),
                                StructField("no_of_week_nights", IntegerType(), False),
                                StructField("type_of_meal_plan", StringType(), False),
                                StructField("required_car_parking_space", IntegerType(), False),
                                StructField("room_type_reserved", StringType(), False),
                                StructField("lead_time", IntegerType(), False),
                                StructField("avg_price_per_room", DoubleType(), False),
                                StructField("no_of_special_requests", IntegerType(), False),
                                StructField("date_of_arrival", TimestampType(), False),
                                StructField("date_of_booking", TimestampType(), False),
                            ]
                        ),
                        True,
                    ),
                )
            ]
        )

        inference_table_parsed = inference_table.withColumn(
            "parsed_request", F.from_json(F.col("request"), request_schema)
        )
        inference_table_parsed.display()
        logger.info("parsed request completed")

        response_schema = StructType(
            [
                StructField("predictions", ArrayType(LongType()), True),
                StructField(
                    "databricks_output",
                    StructType(
                        [
                            StructField("trace", StringType(), True),
                            StructField("databricks_request_id", StringType(), True),
                        ]
                    ),
                    True,
                ),
            ]
        )

        inference_table_parsed = inference_table_parsed.withColumn(
            "parsed_response", F.from_json(F.col("response"), response_schema)
        )
        inference_table_parsed.display()
        logger.info("parsed response completed")

        df_exploded = inference_table_parsed.withColumn("record", F.explode(F.col("parsed_request.dataframe_records")))
        df_exploded.display()
        logger.info("exploded records completed")

        df_final = df_exploded.select(
            F.from_unixtime(F.col("timestamp_ms") / 1000).cast("timestamp").alias("timestamp"),
            "timestamp_ms",
            "databricks_request_id",
            "execution_time_ms",
            F.col("record.booking_id").alias("booking_id"),
            F.col("record.no_of_adults").alias("no_of_adults"),
            F.col("record.no_of_children").alias("no_of_children"),
            F.col("record.no_of_weekend_nights").alias("no_of_weekend_nights"),
            F.col("record.no_of_week_nights").alias("no_of_week_nights"),
            F.col("record.type_of_meal_plan").alias("type_of_meal_plan"),
            F.col("record.required_car_parking_space").alias("required_car_parking_space"),
            F.col("record.room_type_reserved").alias("room_type_reserved"),
            F.col("record.lead_time").alias("lead_time"),
            F.col("record.avg_price_per_room").alias("avg_price_per_room"),
            F.col("record.no_of_special_requests").alias("no_of_special_requests"),
            F.col("record.date_of_arrival").alias("date_of_arrival"),
            F.col("record.date_of_booking").alias("date_of_booking"),
            F.col("parsed_response.predictions")[0].alias("prediction"),
            F.lit("hotel_reservations_model_fe").alias("model_name"),
        )
        df_final.display()
        logger.info("final dataframe created")

        return df_final

    def _create_monitor_table(self, inference_data: DataFrame) -> None:
        # merge the predictions with the ground truth data
        test_set = spark.table(f"{self.model.catalog_name}.{self.model.schema_name}.test_set")
        inference_set = spark.sql(f"SELECT * FROM {self.model.catalog_name}.{self.model.schema_name}.inference_set")

        df_final_with_status = (
            inference_data.join(test_set.select("booking_id", "booking_status"), on="booking_id", how="left")
            .withColumnRenamed("booking_status", "booking_status_test")
            .join(inference_set.select("booking_id", "booking_status"), on="booking_id", how="left")
            .withColumnRenamed("booking_status", "booking_status_inference")
            .select(
                "*", F.coalesce(F.col("booking_status_test"), F.col("booking_status_inference")).alias("booking_status")
            )
            .drop("booking_status_test", "booking_status_inference")
            .withColumn("booking_status", F.col("booking_status").cast("integer"))
            .withColumn("prediction", F.col("prediction").cast("integer"))
            .dropna(subset=["booking_status", "prediction"])
        )

        logger.info("inference data updated with ground truth for booking_status")

        hotel_features = spark.table(self.model.feature_table_name)
        df_final_with_features = df_final_with_status.join(hotel_features, on="booking_id", how="left")

        # Convert the dataframe to delta table and append to the monitoring table
        df_final_with_features.write.format("delta").mode("append").saveAsTable(self.monitoring_table_fullname)

        # Important to update monitoring
        spark.sql(
            f"ALTER TABLE {self.monitoring_table_fullname} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

    def _create_quality_monitor(self) -> None:
        # we can also add 1) baseline_table_name and 2) slicing_exprs to enhance the implementation.
        self.serving.workspace.quality_monitors.create(
            table_name=self.monitoring_table_fullname,
            assets_dir=f"/Workspace/Shared/lakehouse_monitoring/{self.monitoring_table_fullname}",
            output_schema_name=f"{self.model.catalog_name}.{self.model.schema_name}",
            inference_log=MonitorInferenceLog(
                problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
                prediction_col="prediction",
                timestamp_col="timestamp",
                granularities=["30 minutes"],
                model_id_col="model_name",
                label_col="booking_status",
            ),
        )

    def _verify_quality_monitor(self, max_retries: int = 20, retry_interval: int = 30) -> bool:
        """Verify the quality monitor's status and wait for it to become active.

        Repeatedly checks the status of the quality monitor for a specified table,
        waiting for it to become active. If the monitor becomes active within the
        given number of retries, it returns True. Otherwise, it returns False.

        :param max_retries: Maximum number of retry attempts
        :param retry_interval: Time interval between retries in seconds
        :return: True if the monitor becomes active, False otherwise
        """
        # https://www.perplexity.ai/search/you-are-an-expert-in-databrick-EYVnMWq4SrONbrHbl7zxRw
        for _ in range(max_retries):
            try:
                # Fetch the monitor details
                monitor = self.serving.workspace.quality_monitors.get(table_name=self.monitoring_table_fullname)

                # Check if the monitor status is ACTIVE
                if monitor.status == "MONITOR_STATUS_ACTIVE":
                    print(f"Quality monitor for {self.monitoring_table_fullname} is active and serving.")
                    return True
                else:
                    print(f"Monitor status: {monitor.status}. Retrying in {retry_interval} seconds...")
                    time.sleep(retry_interval)

            except Exception as e:
                print(f"Error checking monitor status: {str(e)}. Retrying in {retry_interval} seconds...")
                time.sleep(retry_interval)

        print(f"Quality monitor for {self.monitoring_table_fullname} failed to activate after {max_retries} attempts.")
        return False
