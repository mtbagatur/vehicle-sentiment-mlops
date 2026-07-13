from typing import Optional
from app.models.vehicle import Vehicle
from app.models.errors import VehicleValidationError


class VehicleDatabase:
    """In-memory vehicle inventory database."""

    def __init__(self):
        self._vehicles: dict[str, Vehicle] = {}

    def create(self, vehicle: Vehicle) -> None:
        if vehicle.vehicle_id in self._vehicles:
            raise VehicleValidationError(f"Vehicle '{vehicle.vehicle_id}' already exists")
        self._vehicles[vehicle.vehicle_id] = vehicle

    def read(self, vehicle_id: str) -> Vehicle:
        if vehicle_id not in self._vehicles:
            raise VehicleValidationError(f"Vehicle '{vehicle_id}' not found")
        return self._vehicles[vehicle_id]

    def update(self, vehicle_id: str, **kwargs) -> None:
        vehicle = self.read(vehicle_id)
        for key, value in kwargs.items():
            if not hasattr(vehicle, key):
                raise VehicleValidationError(f"'{key}' is not a valid field")
            setattr(vehicle, key, value)
        vehicle.__post_init__()

    def delete(self, vehicle_id: str) -> None:
        if vehicle_id not in self._vehicles:
            raise VehicleValidationError(f"Vehicle '{vehicle_id}' not found")
        del self._vehicles[vehicle_id]

    def list_all(self, sort_by: str = "vehicle_id") -> list[Vehicle]:
        vehicles = list(self._vehicles.values())
        if vehicles and not hasattr(vehicles[0], sort_by):
            raise VehicleValidationError(f"Cannot sort by '{sort_by}'")
        return sorted(vehicles, key=lambda v: getattr(v, sort_by))

    def sell(self, vehicle_id: str, quantity: int) -> None:
        vehicle = self.read(vehicle_id)
        if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
            raise VehicleValidationError("Quantity must be a positive integer")
        if vehicle.quantity < quantity:
            raise VehicleValidationError(
                f"Insufficient stock: {vehicle.quantity} available, {quantity} requested"
            )
        vehicle.quantity -= quantity

    def restock(self, vehicle_id: str, quantity: int) -> None:
        vehicle = self.read(vehicle_id)
        if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
            raise VehicleValidationError("Quantity must be a positive integer")
        vehicle.quantity += quantity

    @property
    def count(self) -> int:
        return len(self._vehicles)
