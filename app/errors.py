class AppError(Exception):
    """Error terkontrol dengan status HTTP dan pesan aman untuk klien."""

    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
