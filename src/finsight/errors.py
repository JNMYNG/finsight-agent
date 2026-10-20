"""Public exceptions; transport errors never include authenticated URLs."""


class FinSightError(Exception):
    """Base error for configuration, API, and data failures."""


class ConfigurationError(FinSightError):
    pass


class DartTransportError(FinSightError):
    pass


class DartAPIError(FinSightError):
    def __init__(self, status: str, message: str) -> None:
        self.status = status
        self.message = message
        super().__init__(f"OpenDART [{status}]: {message}")


class NoDataError(DartAPIError):
    pass


class InvalidResponseError(FinSightError):
    pass


class CompanyNotFoundError(FinSightError):
    pass


class AmbiguousCompanyError(FinSightError):
    pass


class AgentModelError(FinSightError):
    """Sanitized model API/protocol failure."""


class DisclosureError(FinSightError):
    """Invalid, unsupported, or unavailable disclosure source/index."""


class EmbeddingError(FinSightError):
    """Sanitized embedding API failure."""
