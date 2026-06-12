"""Feature-engineering constants (formerly hardcoded in scripts/Preprocessing.py).

Centralized so deployment-specific values (work hours, the organization's email
domain, URL classification lists) live in one reviewable place instead of being
buried in the preprocessing module. ueba.features.preprocessing re-exports all
of these for backward compatibility.
"""

# Bare default output directory for preprocessing artifacts (the version-scoped
# absolute path lives in ueba.config.DEFAULT_OUTPUT_DIR).
DEFAULT_OUTPUT_DIR = "processed_datasets"

# Population-level fallback work hours; v6 derives per-user envelopes from logon
# history (10th/90th percentile) and persists them to user_work_hours.parquet.
WORK_HOURS = (9, 17)

# Columns read from each raw CERT log file.
USECOLS_MAP = {
    "logon":  ["date", "user", "pc", "activity"],
    "file":   ["date", "user", "pc", "activity", "filename"],
    "device": ["date", "user", "pc", "activity"],
    "email":  ["date", "user", "pc", "to", "attachments"],
    "http":   ["date", "user", "pc", "url", "activity"],
}

DTYPE_MAP = {
    "user":     "category",
    "pc":       "category",
    "activity": "category",
}

TIMESTAMP_FORMAT = "%m/%d/%Y %H:%M:%S"

# CERT sources too large to load eagerly — processed in chunks.
LARGE_FILE_SOURCES = {"email", "http", "file"}

# Organization domain for external-email detection.
INTERNAL_EMAIL_DOMAIN = "dtaa.com"

# URLs longer than this are flagged (possible encoded-exfiltration indicator).
LONG_URL_THRESHOLD = 90

# HTTP URL classification lists.
JOB_DOMAINS = {
    "careerbuilder.com",
    "indeed.com",
    "monster.com",
    "simplyhired.com",
    "linkedin.com",
    "www.linkedin.com",
}

CLOUD_STORAGE_DOMAINS = {
    "dropbox.com",
    "www.dropbox.com",
    "drive.google.com",
    "docs.google.com",
    "yousendit.com",
    "www.yousendit.com",
}

SUSPICIOUS_DOMAINS = {
    "wikileaks.org",
    "www.wikileaks.org",
}
