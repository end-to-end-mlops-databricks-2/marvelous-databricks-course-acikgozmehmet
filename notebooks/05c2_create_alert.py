# Databricks notebook source
# MAGIC %md
# MAGIC ### Create a query that checks the accuracy_score being less than 0.70

# COMMAND ----------
import time

from loguru import logger

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql


# COMMAND ----------
workspace = WorkspaceClient()

sources = workspace.data_sources.list()

alert_query= """
SELECT accuracy_score
FROM mlops_dev.acikgozm.model_monitoring_profile_metrics
WHERE window.start = (SELECT MAX(window.start) FROM mlops_dev.acikgozm.model_monitoring_profile_metrics)"""

query = sql.CreateQueryRequestQuery(display_name=f"hotel-reservations-alert-query-accuracy-score-{time.time_ns()}",
                                    warehouse_id=sources[0].warehouse_id,
                                    description="Alert on hotel reservations accuracy score",
                                    query_text=alert_query)

query = workspace.queries.create(query=query)


alert = sql.CreateAlertRequestAlert(
        display_name=f'"hotel-reservations-accuracy_alert_{time.time_ns()}',
        query_id=query.id,
        condition=sql.AlertCondition(
            operand=sql.AlertConditionOperand(
                column=sql.AlertOperandColumn(name="accuracy")
            ),
            op=sql.AlertOperator.LESS_THAN,
            threshold=sql.AlertConditionThreshold(
                value=sql.AlertOperandValue(double_value=0.70)
            )
        ))

alert = workspace.alerts.create(alert=alert)

logger.info(f"Alert created with ID: {alert.id}")

# COMMAND ----------
# clean up
workspace.queries.delete(id=query.id)
workspace.alerts.delete(id=alert.id)
