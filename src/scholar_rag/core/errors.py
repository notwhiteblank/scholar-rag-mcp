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
