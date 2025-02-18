"""Unit tests for model_serving."""

import pytest

from hotel_reservations.serving import ModelServing
from hotel_reservations.utility import call_endpoint, is_databricks


@pytest.mark.skipif(not is_databricks(), reason="Only runs on Databricks")
def test_model_serving_fixture_as_expected(model_serving: ModelServing) -> None:
    """Test the basic model fixture to ensure it meets expected criteria.

    This function checks various attributes and configurations of the basic model fixture.

    :param model_serving: The basic model fixture to be tested
    """
    assert isinstance(model_serving, ModelServing)
    assert model_serving.model_name == "hotel_reservations_model_basic"
    assert isinstance(model_serving.served_entities, list)
    assert len(model_serving.served_entities) == 1
    assert model_serving.served_entities[0].entity_name == model_serving.model_name
    assert model_serving.served_entities[0].scale_to_zero_enabled is True
    assert model_serving.served_entities[0].workload_size == "Small"


@pytest.mark.skipif(not is_databricks(), reason="Only runs on Databricks")
def test_model_serving_get_latest_model_version(model_serving: ModelServing) -> None:
    """Test the _get_latest_model_version method in ModelServing.

    This function retrieves the latest model version registered in MLflow.

    :param model_serving: The ModelServing fixture to be tested
    """
    latest_version = model_serving._get_latest_model_version()
    assert isinstance(latest_version, int)


@pytest.mark.skipif(not is_databricks(), reason="Only runs on Databricks")
def test_model_serving_deploy_or_update_serving_endpoint_with_retry(deployed_model_serving: ModelServing) -> None:
    """Test the deploy_or_update_serving_endpoint_with_retry method in ModelServing.

    This function deploys or updates a Databricks serving endpoint with a retry mechanism.

    :param deployed_model_serving: The ModelServing fixture to be tested
    """
    endpoint = deployed_model_serving.workspace.serving_endpoints.get(name=deployed_model_serving.endpoint_name)
    assert endpoint.state.ready.value == "READY", "Endpoint is not alive"


@pytest.mark.skipif(not is_databricks(), reason="Only runs on Databricks")
def test_model_serving_api_call(deployed_model_serving: ModelServing) -> None:
    """Test the model serving API call.

    This test function sends a sample dataframe record to the deployed model serving endpoint
    and verifies the response.

    :param deployed_model_serving: The deployed model serving object
    """
    dataframe_records = [
        {
            "no_of_adults": 2,
            "no_of_children": 0,
            "no_of_weekend_nights": 0,
            "no_of_week_nights": 3,
            "required_car_parking_space": 0,
            "lead_time": 8,
            "repeated_guest": 0,
            "no_of_previous_cancellations": 0,
            "no_of_previous_bookings_not_canceled": 0,
            "avg_price_per_room": 87.78,
            "no_of_special_requests": 1,
            "type_of_meal_plan": "Meal Plan 1",
            "room_type_reserved": "Room_Type 4",
        }
    ]

    status_code, response_text = call_endpoint(
        endpoint_name=deployed_model_serving.endpoint_name, record=dataframe_records
    )

    assert status_code == 200
    assert response_text == '{"predictions": [0]}'
