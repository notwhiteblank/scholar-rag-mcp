class ScholarRagError(Exception):
    code: str = "internal_error"

class ConfigError(ScholarRagError):
    code = "config_error"

class KbNotFoundError(ScholarRagError):
    code = "kb_not_found"

class KbExistsError(ScholarRagError):
    code = "kb_exists"

class DocNotFoundError(ScholarRagError):
    code = "doc_not_found"

class DocExistsError(ScholarRagError):
    code = "doc_exists"

    def __init__(self, doc_id: str, message: str | None = None) -> None:
        super().__init__(message or f"document already exists: {doc_id}")
        self.doc_id = doc_id

class InvalidFilterError(ScholarRagError):
    code = "invalid_filter"

class ConfirmTokenError(ScholarRagError):
    code = "confirm_token_invalid"

class JobNotFoundError(ScholarRagError):
    code = "job_not_found"

class ServiceUnavailableError(ScholarRagError):
    code = "service_unavailable"

class PipelineStageError(ScholarRagError):
    code = "pipeline_stage_failed"

    def __init__(self, message: str, *, stage: str | None = None, cause: str | None = None) -> None:
        super().__init__(message)
        self.stage = stage
        self.cause = cause
