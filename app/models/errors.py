class VehicleValidationError(ValueError):
    """Raised when vehicle data validation fails."""
    pass


class ModelNotReadyError(Exception):
    """Raised when sentiment model is not yet trained."""
    pass
