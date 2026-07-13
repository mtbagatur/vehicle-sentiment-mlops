from fastapi import APIRouter, HTTPException
from app.models.vehicle import Car, Motorcycle
from app.models.errors import VehicleValidationError
from app.schemas import (
    CarCreate, MotorcycleCreate, VehicleResponse,
    SellRequest, RestockRequest
)
from app.api.deps import get_vehicle_db
from prometheus_client import Counter, Histogram
import time

router = APIRouter(prefix="/vehicles", tags=["vehicles"])

VEHICLE_OPS = Counter("vehicle_operations_total", "Vehicle operations", ["operation", "status"])
VEHICLE_LATENCY = Histogram("vehicle_operation_duration_seconds", "Vehicle operation latency", ["operation"])


def _to_response(vehicle) -> VehicleResponse:
    return VehicleResponse(
        vehicle_id=vehicle.vehicle_id,
        name=vehicle.name,
        quantity=vehicle.quantity,
        price=vehicle.price,
        type=type(vehicle).__name__.lower()
    )


@router.post("/cars", response_model=VehicleResponse, status_code=201)
def create_car(payload: CarCreate):
    db = get_vehicle_db()
    start = time.time()
    try:
        car = Car(
            vehicle_id=payload.vehicle_id,
            name=payload.name,
            quantity=payload.quantity,
            price=payload.price,
            num_doors=payload.num_doors,
            is_electric=payload.is_electric,
        )
        db.create(car)
        VEHICLE_OPS.labels(operation="create_car", status="success").inc()
        return _to_response(car)
    except VehicleValidationError as e:
        VEHICLE_OPS.labels(operation="create_car", status="error").inc()
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        VEHICLE_LATENCY.labels(operation="create_car").observe(time.time() - start)


@router.post("/motorcycles", response_model=VehicleResponse, status_code=201)
def create_motorcycle(payload: MotorcycleCreate):
    db = get_vehicle_db()
    start = time.time()
    try:
        moto = Motorcycle(
            vehicle_id=payload.vehicle_id,
            name=payload.name,
            quantity=payload.quantity,
            price=payload.price,
            engine_cc=payload.engine_cc,
            has_sidecar=payload.has_sidecar,
        )
        db.create(moto)
        VEHICLE_OPS.labels(operation="create_motorcycle", status="success").inc()
        return _to_response(moto)
    except VehicleValidationError as e:
        VEHICLE_OPS.labels(operation="create_motorcycle", status="error").inc()
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        VEHICLE_LATENCY.labels(operation="create_motorcycle").observe(time.time() - start)


@router.get("/{vehicle_id}", response_model=VehicleResponse)
def get_vehicle(vehicle_id: str):
    db = get_vehicle_db()
    try:
        return _to_response(db.read(vehicle_id))
    except VehicleValidationError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/", response_model=list[VehicleResponse])
def list_vehicles(sort_by: str = "vehicle_id"):
    db = get_vehicle_db()
    try:
        return [_to_response(v) for v in db.list_all(sort_by=sort_by)]
    except VehicleValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{vehicle_id}", status_code=204)
def delete_vehicle(vehicle_id: str):
    db = get_vehicle_db()
    try:
        db.delete(vehicle_id)
    except VehicleValidationError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{vehicle_id}/sell", status_code=200)
def sell_vehicle(vehicle_id: str, payload: SellRequest):
    db = get_vehicle_db()
    try:
        db.sell(vehicle_id, payload.quantity)
        return {"message": f"Sold {payload.quantity} units of '{vehicle_id}'"}
    except VehicleValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{vehicle_id}/restock", status_code=200)
def restock_vehicle(vehicle_id: str, payload: RestockRequest):
    db = get_vehicle_db()
    try:
        db.restock(vehicle_id, payload.quantity)
        return {"message": f"Restocked '{vehicle_id}' with {payload.quantity} units"}
    except VehicleValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
