from pathlib import Path

import pytest

from conftest import ALL_INSTANCE_PATHS
from environment.data_converter import convert_instance

REQUIRED_RETAILER_COLUMNS = {
    "id", "x_cord", "y_cord", "initial_inventory", "max_capacity",
    "min_capacity", "demand", "holding_cost",
}
REQUIRED_SUPPLIER_KEYS = {"id", "x_cord", "y_cord", "initial_inventory", "production_rate", "holding_cost"}


@pytest.mark.parametrize("instance_path", ALL_INSTANCE_PATHS, ids=lambda p: Path(p).stem)
def test_convert_instance_structure(instance_path):
    """Every benchmark instance file parses into the shape `IRPEnv` expects."""
    data = convert_instance(instance_path)
    assert set(data.keys()) == {"parameters", "supplier", "retailers"}

    params = data["parameters"]
    assert isinstance(params["num_nodes"], int) and params["num_nodes"] > 1
    assert isinstance(params["episode_length"], int) and params["episode_length"] > 0
    assert params["vehicle_capacity"] > 0

    supplier = data["supplier"]
    assert REQUIRED_SUPPLIER_KEYS.issubset(supplier.keys())
    assert supplier["initial_inventory"] >= 0
    assert supplier["production_rate"] >= 0
    assert supplier["holding_cost"] >= 0

    retailers = data["retailers"]
    num_retailers = params["num_nodes"] - 1
    assert len(retailers) == num_retailers
    assert REQUIRED_RETAILER_COLUMNS.issubset(retailers.columns)


@pytest.mark.parametrize("instance_path", ALL_INSTANCE_PATHS, ids=lambda p: Path(p).stem)
def test_retailer_values_are_physically_sane(instance_path):
    """
    Bounds `IRPEnv` relies on without re-checking itself: capacities and costs
    non-negative, min <= max, and initial stock within capacity — verified
    to hold for every instance currently in the benchmark set.
    """
    retailers = convert_instance(instance_path)["retailers"]

    assert (retailers["min_capacity"] >= 0).all()
    assert (retailers["max_capacity"] >= retailers["min_capacity"]).all()
    assert (retailers["initial_inventory"] >= 0).all()
    assert (retailers["initial_inventory"] <= retailers["max_capacity"]).all()
    assert (retailers["demand"] >= 0).all()
    assert (retailers["holding_cost"] >= 0).all()
    assert retailers["id"].is_unique


@pytest.mark.parametrize("instance_path", ALL_INSTANCE_PATHS, ids=lambda p: Path(p).stem)
def test_instance_name_matches_retailer_count(instance_path):
    """The `n<N>` in each filename (e.g. `abs1n10.dat` -> 10) matches the parsed node count."""
    stem = Path(instance_path).stem
    expected_num_retailers = int(stem.rsplit("n", 1)[-1])
    params = convert_instance(instance_path)["parameters"]
    assert params["num_nodes"] - 1 == expected_num_retailers
