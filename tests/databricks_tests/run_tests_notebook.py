# Databricks notebook source
!pip install /Volumes/mlops_dev/acikgozm/packages/credit_default-latest-py3-none-any.whl
# COMMAND ----------

%restart_python

# COMMAND ----------
from credit_default import __version__
print(__version__)

# COMMAND ----------

import os
import pathlib
import sys

import pytest

# COMMAND ----------

# wd = os.getcwd()
wd = pathlib.Path().resolve()
print(f"Current working directory: {wd}")
unit_test_folder = (wd / "../unit_tests").as_posix()
# COMMAND ----------


# dbutils.widgets.text(
#     "p_test_folder", unit_test_folder
# )  # noqa: F821 # pylint: disable=undefined-variable
# v_test_folder = dbutils.widgets.get("p_test_folder")  # noqa: F821
# print(f"Test folder path from widget: {v_test_folder}")

# COMMAND ----------
# Ensure pytest-json-report is installed (uncomment if needed)
# !pip install pytest-json-report --quiet

# COMMAND ----------

try:
    # Prevent Python from writing .pyc files
    sys.dont_write_bytecode = True

    # Change directory to the test folder
    os.chdir(unit_test_folder)
    print(f"Changed working directory to: {unit_test_folder}")

    # Run pytest with JSON report generation
    result = pytest.main(["-v", "--json-report", f"--json-report-file={wd}/test_run_results.json"])

    if result == 0:
        print("Tests ran successfully.")
    else:
        print(f"Some tests failed. Exit code: {result}")

except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"An error occurred while running tests: {e}")


# COMMAND ----------


# COMMAND ----------
