# Databricks notebook source
!pip install /Volumes/mlops_dev/acikgozm/packages/hotel_reservations-latest-py3-none-any.whl
# COMMAND ----------

%restart_python

# COMMAND ----------
from hotel_reservations import __version__
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
# unit_test_folder = (wd / "../unit_tests").resolve().as_posix()
unit_test_folder = (wd / "../unit_tests/code_under_test").resolve().as_posix()
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
