import pytest
from app.models.vehicle import Car, Motorcycle
from app.models.errors import VehicleValidationError
from app.services.vehicle_db import VehicleDatabase


@pytest.fixture
def db():
    return VehicleDatabase()


@pytest.fixture
def car():
    return Car(vehicle_id="c1", name="Honda Civic", quantity=10, price=25000.0)


def test_create_and_read(db, car):
    db.create(car)
    result = db.read("c1")
    assert result.name == "Honda Civic"


def test_create_duplicate_raises(db, car):
    db.create(car)
    with pytest.raises(VehicleValidationError, match="already exists"):
        db.create(car)


def test_read_nonexistent_raises(db):
    with pytest.raises(VehicleValidationError, match="not found"):
        db.read("nonexistent")


def test_delete(db, car):
    db.create(car)
    db.delete("c1")
    with pytest.raises(VehicleValidationError):
        db.read("c1")


def test_sell_reduces_quantity(db, car):
    db.create(car)
    db.sell("c1", 3)
    assert db.read("c1").quantity == 7


def test_sell_insufficient_raises(db, car):
    db.create(car)
    with pytest.raises(VehicleValidationError, match="Insufficient"):
        db.sell("c1", 100)


def test_restock_increases_quantity(db, car):
    db.create(car)
    db.restock("c1", 5)
    assert db.read("c1").quantity == 15


def test_list_all_sorted(db):
    db.create(Car(vehicle_id="z1", name="Z Car", quantity=1, price=100.0))
    db.create(Car(vehicle_id="a1", name="A Car", quantity=1, price=100.0))
    result = db.list_all(sort_by="vehicle_id")
    assert result[0].vehicle_id == "a1"


def test_car_invalid_num_doors():
    with pytest.raises(VehicleValidationError):
        Car(vehicle_id="c1", name="Test", quantity=1, price=100.0, num_doors=0)


def test_motorcycle_invalid_engine_cc():
    with pytest.raises(VehicleValidationError):
        Motorcycle(vehicle_id="m1", name="Test", quantity=1, price=100.0, engine_cc=-100)
