from dataclasses import dataclass
from app.models.errors import VehicleValidationError


@dataclass
class Vehicle:
    vehicle_id: str
    name: str
    quantity: int
    price: float

    def __post_init__(self):
        if not isinstance(self.vehicle_id, str) or not self.vehicle_id.strip():
            raise VehicleValidationError("vehicle_id must be a non-empty string")
        if not isinstance(self.name, str) or not self.name.strip():
            raise VehicleValidationError("name must be a non-empty string")
        if not isinstance(self.quantity, int) or isinstance(self.quantity, bool) or self.quantity < 0:
            raise VehicleValidationError("quantity must be a non-negative integer")
        if not isinstance(self.price, (int, float)) or isinstance(self.price, bool) or self.price < 0:
            raise VehicleValidationError("price must be a non-negative number")


@dataclass
class Car(Vehicle):
    num_doors: int = 4
    is_electric: bool = False

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.num_doors, int) or isinstance(self.num_doors, bool) or self.num_doors <= 0:
            raise VehicleValidationError("num_doors must be a positive integer")
        if not isinstance(self.is_electric, bool):
            raise VehicleValidationError("is_electric must be a boolean")


@dataclass
class Motorcycle(Vehicle):
    engine_cc: int = 600
    has_sidecar: bool = False

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.engine_cc, int) or isinstance(self.engine_cc, bool) or self.engine_cc <= 0:
            raise VehicleValidationError("engine_cc must be a positive integer")
        if not isinstance(self.has_sidecar, bool):
            raise VehicleValidationError("has_sidecar must be a boolean")
