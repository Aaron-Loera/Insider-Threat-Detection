import glob
import os
import numpy as np
import pandas as pd
from urllib.parse import urlparse

# Layer A

# Constants
DEFAULT_OUTPUT_DIR = "processed_datasets"
WORK_HOURS = (9, 17)

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

LARGE_FILE_SOURCES = {"email", "http"}
DEFAULT_SAMPLE_LOG_ROWS = 100_000

INTERNAL_EMAIL_DOMAIN = "dtaa.com"
LONG_URL_THRESHOLD = 90

JOB_DOMAINS = {
    "careerbuilder.com",
    "indeed.com",
    "monster.com",
    "simplyhired.com",
    "linkedin.com",
    "www.linkedin.com"
}

CLOUD_STORAGE_DOMAINS = {
    "dropbox.com",
    "www.dropbox.com",
    "drive.google.com",
    "docs.google.com",
    "yousendit.com",
    "www.yousendit.com"
}

SUSPICIOUS_DOMAINS = {
    "wikileaks.org",
    "www.wikileaks.org"
}

# Functions
def load_raw_logs(cert_path: str, sample_n: int | None = None) -> dict:
    """
    Loads the raw CERT log files needed for preprocessing. Small files are loaded eagerly
    whereas large files are represented as {path: ..., chunked: True} for downstream chunked
    processing.
    
    Args:
        cert_path: The base path containing the CERT dataset
        sample_n: If provided, loads only the first sample_n rows from each source.
        
    Returns:
        dict: {file_name: DataFrame | {"path": str, "chunked": True, "sample_n": int | None}}
    """
    file_map = {
        "logon":  "logon.csv",
        "file":   "file.csv",
        "device": "device.csv",
        "email":  "email.csv",
        "http":   "http.csv",
    }
    
    logs = {}
    missing_files = []
    
    for source_name, filename in file_map.items():
        full_path = os.path.join(cert_path, filename)
        # Takes note of missing file paths
        if not os.path.exists(full_path):
            missing_files.append(filename)
            continue
        # Defers large files for later chunked loading
        if source_name in LARGE_FILE_SOURCES:
            logs[source_name] = {
                "path": full_path,
                "chunked": True,
                "sample_n": sample_n,
            }
        # Loads small files with optimized dtypes
        else:
            applicable_dtype = {col: DTYPE_MAP[col] for col in USECOLS_MAP[source_name] if col in DTYPE_MAP}
            logs[source_name] = pd.read_csv(
                full_path,
                usecols=USECOLS_MAP[source_name],
                dtype=applicable_dtype,
                nrows=sample_n,
            )
            
    if missing_files:
        raise FileNotFoundError("Missing required CERT files: " + ", ".join(missing_files))
    
    return logs


def load_ldap_profiles(ldap_path: str, users: set[str] | None = None) -> pd.DataFrame:
    """
    Loads monthly LDAP snapshot CSVs and optionally filters to the users active in a sampled log subset.

    Args:
        ldap_path: Directory containing LDAP snapshot CSVs.
        users: Optional set of lowercase user IDs to filter LDAP rows to only those users.

    Returns:
        pd.DataFrame: Concatenated LDAP snapshot rows with lowercase user IDs and snapshot metadata.
    """
    ldap_files = sorted(glob.glob(os.path.join(ldap_path, "*.csv")))
    if not ldap_files:
        raise FileNotFoundError(f"No LDAP CSVs found in: {ldap_path}")

    frames = []
    for path in ldap_files:
        snapshot = os.path.basename(path).replace(".csv", "")
        df = pd.read_csv(path)
        df["_snapshot"] = pd.to_datetime(snapshot)
        df["user"] = df["user_id"].astype(str).str.lower()
        if users is not None:
            df = df[df["user"].isin(users)]
        frames.append(df)

    ldap_raw = pd.concat(frames, ignore_index=True)
    print(
        f"Loaded {len(ldap_files)} LDAP snapshots — {ldap_raw['user'].nunique()} unique users, {len(ldap_raw)} total rows"
    )
    return ldap_raw


def build_user_profiles(ldap_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a per-user profile table from LDAP snapshot rows.

    Args:
        ldap_raw: Raw LDAP rows containing a lowercase `user` column and `_snapshot` datetime.

    Returns:
        pd.DataFrame: User profile table with only the most recent snapshot and an `is_active` flag.
    """
    latest_snapshot = ldap_raw["_snapshot"].max()

    user_profiles = (
        ldap_raw
        .sort_values("_snapshot")
        .groupby("user", as_index=False)
        .last()
        [["user", "employee_name", "role", "department", "functional_unit", "supervisor", "_snapshot"]]
    )

    user_profiles["is_active"] = user_profiles["_snapshot"] == latest_snapshot
    user_profiles = user_profiles.drop(columns="_snapshot")
    return user_profiles


def normalize_shared_columns(df: pd.DataFrame, remove_cols: list=["id"], sort: bool=True) -> pd.DataFrame:
    """
    Normalizes CERT log files across commonly shared columns. Additionally drops columns that are deemed irrelevant.

    Args:
        df: The raw CERT dataframe (logon, file, device, or email)
        remove_cols: The columns to drop from the original CERT file
        sort: When False, skips the final (user, pc, timestamp) sort.

    Returns:
        pd.Dataframe: A normalized dataframe with consistent identifiers and fields
    """
    # Standardizing column names
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    
    # Renaming date column
    if "date" in df.columns:
        df.rename(columns={"date": "timestamp"}, inplace=True)
        
    if "timestamp" not in df.columns:
        raise KeyError("Expected a 'date' or 'timestamp' column in DataFrame.")
        
    # Converting timestamp to datetime.
    df["timestamp"] = pd.to_datetime(df["timestamp"], format=TIMESTAMP_FORMAT, errors="coerce")
    
    # Dropping rows with invalid timestamps
    df.dropna(axis=0, subset=["timestamp"], inplace=True)
    
    # Creating a 'day' aggregation key column
    df["day"] = df["timestamp"].dt.floor("D")
    
    # Normalizing identifiers
    df["user"] = df["user"].astype(str).str.lower().str.strip().astype("category")
    df["pc"] = df["pc"].astype(str).str.lower().str.strip().astype("category")
    
    # Dropping unusable columns
    remove_cols = [col.lower().strip() for col in remove_cols]
    cols_to_drop = [col for col in remove_cols if col in df.columns]
    df.drop(axis=1, columns=cols_to_drop, inplace=True)
    
    # Sorting rows if specified
    if sort:
        df.sort_values(by=["user", "pc", "timestamp"], inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df


def validate_normalized_files(logs: dict[str, pd.DataFrame], required_cols: set={"user", "pc", "timestamp", "day"}) -> None:
    """
    A sanity-check for normalized CERT logs before feature extraction is performed.
    Chunked sources (email, http) are skipped as they are validated per-chunk during extraction.
    
    Args:
        logs: A dictionary of the format: {log_name: DataFrame | {"path": str, "chunked": True}}
        required_cols: A set of required columns for each CERT log
        
    Returns:
        None:
    """
    print("Validating normalized files")
    for name, df in logs.items():
        # Case where file is marked as large file
        if isinstance(df, dict):
            print(f"Skipped {name}.csv (validated per chunk during extraction)")
            continue
        # Raises error if required column is missing
        if not required_cols.issubset(df.columns):
            missing = required_cols.difference(df.columns)
            raise ValueError(f"Normalized {name} log is missing column(s): {sorted(missing)}")
        # Raises error if required column is in invalid format
        if str(df["day"].dtype) != "datetime64[ns]":
            raise TypeError(f"Normalized {name} log has invalid 'day' format: {df["day"].dtype}")
        
    print("All CERT logs were properly normalized.")
    
    
def extract_domain(url: str) -> str:
    """
    Extracts a lowercased domain from a URL-like value.
    
    Args:
        url: Raw url string
        
    Returns:
        str: The domain found within the URL
    """
    if pd.isna(url):
        return ""
    
    url = str(url).strip().lower()
    if not url:
        return ""
    
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
        
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""
    
    
def load_log_in_chunks(
    filepath: str,
    usecols: list,
    dtype: dict,
    chunksize: int = 50_000,
    max_rows: int | None = None
) -> pd.io.parsers.readers.TextFileReader:
    """
    Returns an iterator consisting of DataFrames for processing large CSV files in fixed-size chunks.
    
    Args:
        filepath: Absolute path to the CSV file
        usecols: The columns to load in from the CSV file
        dtype: Dict of dtype assignments for specific columns
        chunksize: Number of rows per chunk
        max_rows: If provided, limits the total rows read from the file
        
    Returns:
        pd.io.parsers.TextFileReader: An iterable of DataFrames, one per chunk
    """
    applicable_dtype = {col: dtype[col] for col in usecols if col in dtype}
    return pd.read_csv(
        filepath,
        usecols=usecols,
        dtype=applicable_dtype,
        chunksize=chunksize,
        nrows=max_rows,
    )


def combine_partial_aggregations(partial_list: list, merge_cols: list) -> pd.DataFrame:
    """
    Concatenates a list of per-chunk aggregated DataFrames and sums all count columns across groups.
    
    Args:
        partial_list: A list containing the chunked groupby-aggregated DataFrames with additive count columns
        merge_cols: The groupby key columns such as (user, pc, day)
        
    Returns:
        pd.DataFrame: Combined aggregation with counts correctly summed across all chunks
    """
    combined = pd.concat(partial_list, ignore_index=True)
    return combined.groupby(merge_cols, as_index=False, observed=True, sort=False).sum()


def build_unique_count(identity_frames: list, merge_cols: list, value_col: str, output_col: str, batch_size: int=20) -> pd.DataFrame:
    """
    Computes an exact per-group nunique count from accumulated per-chunk DataFrames.
    
    Args:
        identity_frames: Per-chunk deduplicated DataFrames
        merge_cols: The groupby key columns (user, pc, day)
        value_col: The column whose distinct values are being counted
        output_col: Name for the resulting count column
        batch_size: Number of identity frames to concat per batch before deduplication
        
    Returns:
        pd.DataFrame: (merge_cols + output_col) with exact nunique counts
    """
    deduped_batches = []
    for i in range(0, len(identity_frames), batch_size):
        batch = pd.concat(identity_frames[i:i + batch_size], ignore_index=True).drop_duplicates()
        deduped_batches.append(batch)
    combined = pd.concat(deduped_batches, ignore_index=True).drop_duplicates()
    return (
        combined.groupby(merge_cols, observed=True, sort=False)[value_col]
        .nunique()
        .reset_index()
        .rename(columns={value_col: output_col})
    )
    
    
def extract_logon_features(norm_df: pd.DataFrame, work_hours: tuple=(9, 17)) -> pd.DataFrame:
    """
    Extracts daily authentication behavior features from logon events to create an aggregated feature table.
    
    Args:
        norm_df: The normalized logon dataframe
        work_hours: The range describing regular work hours based on a 24-hour format
        
    Returns:
        pd.DataFrame: Aggregated logon behavior features per (user, pc, day)
    """
    hour = norm_df["timestamp"].dt.hour
    off_hours = (hour < work_hours[0]) | (hour > work_hours[1])
    is_logon = (norm_df["activity"] == "Logon")
    is_logoff = (norm_df["activity"] == "Logoff")

    df = norm_df.assign(
        is_logon=is_logon,
        is_logoff=is_logoff,
        off_hours_logon_flag=(is_logon & off_hours),
    )

    features = (
        df.groupby(["user", "pc", "day"], observed=True, sort=False)
          .agg(
              logon_count=("is_logon", "sum"),
              logoff_count=("is_logoff", "sum"),
              off_hours_logon=("off_hours_logon_flag", "sum"),
          )
          .reset_index()
    )

    return features


def extract_file_features(
    norm_df: pd.DataFrame,
    work_hours: tuple = (9, 17),
    return_identity_frame: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts daily file access behavior features to create an aggregated feature table.

    Args:
        norm_df: Normalized file activity dataframe
        work_hours: The range describing regular work hours based on a 24-hour format
        return_identity_frame: When True, returns a deduplicated (user, day, filename) DataFrame

    Returns:
        pd.DataFrame: Aggregated file behavior features per (user, pc, day).
        If `return_identity_frame` is True, returns a (features, identity_frame) tuple.
    """
    hour = norm_df["timestamp"].dt.hour
    off_hours = (hour < work_hours[0]) | (hour > work_hours[1])
    activity = norm_df["activity"]

    df = norm_df.assign(
        is_open=(activity == "File Open"),
        is_write=(activity == "File Write"),
        is_copy=(activity == "File Copy"),
        is_delete=(activity == "File Delete"),
        off_hours=off_hours,
    )

    features = (
        df.groupby(["user", "pc", "day"], observed=True, sort=False)
          .agg(
              file_open_count=("is_open", "sum"),
              file_write_count=("is_write", "sum"),
              file_copy_count=("is_copy", "sum"),
              file_delete_count=("is_delete", "sum"),
              unique_files_accessed=("filename", "nunique"),
              off_hours_files_accessed=("off_hours", "sum"),
          )
          .reset_index()
    )

    # Returns an additional frame consisting (user, day, filename) granularity
    if return_identity_frame:
        identity_frame = df[["user", "day", "filename"]].drop_duplicates()
        return features, identity_frame

    return features


def extract_device_features(norm_df: pd.DataFrame, work_hours: tuple=(9, 17)) -> pd.DataFrame:
    """
    Extracts daily removable media (USB) behavior features to create an aggregated feature table.
    
    Args:
        norm_df: Normalized file activity dataframe
        work_hours: The range describing regular work hours based on a 24-hour format
        
    Returns:
        pd.DataFrame: Aggregated removable media behavior features per (user, pc, day)
    """
    hour = norm_df["timestamp"].dt.hour
    off_hours = (hour < work_hours[0]) | (hour > work_hours[1])
    activity = norm_df["activity"]

    df = norm_df.assign(
        is_connect=(activity == "Connect"),
        is_disconnect=(activity == "Disconnect"),
        off_hours=off_hours,
    )

    features = (
        df.groupby(["user", "pc", "day"], observed=True, sort=False)
          .agg(
              usb_insert_count=("is_connect", "sum"),
              usb_remove_count=("is_disconnect", "sum"),
              off_hours_usb_usage=("off_hours", "sum"),
          )
          .reset_index()
    )

    return features


def extract_email_features_chunked(
    filepath: str,
    work_hours: tuple = (9, 17),
    chunksize: int = 50_000,
    max_rows: int | None = None,
    return_identity_frame: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Memory-efficient email feature extraction via chunked CSV reading.

    Additive counts are accumulated per chunk then combined via `combine_partial_aggregations`.
    Unique recipients are tracked via `build_unique_count`.

    Args:
        filepath: Absolute path to email.csv
        work_hours: The range describing regular work hours based on a 24-hour format
        chunksize: Number of rows per chunk
        return_identity_frame: When True, returns a deduplicated (user, day, to) DataFrame

    Returns:
        pd.DataFrame: Aggregated email behavior features per (user, pc, day).
        If return_identity_frame is True, returns a (features, identity_frame) tuple.
    """
    MERGE_COLS = ["user", "pc", "day"]
    partial_aggs = []
    identity_frames = []

    for i, chunk in enumerate(load_log_in_chunks(filepath, USECOLS_MAP["email"], DTYPE_MAP, chunksize, max_rows), start=1):
        print(f"  Email chunk {i}...")
        chunk = normalize_shared_columns(chunk, sort=False)

        # Defining off-hour emails
        hour = chunk["timestamp"].dt.hour
        chunk["off_hours"] = (hour < work_hours[0]) | (hour > work_hours[1])

        # External email heuristic
        chunk["external_emails_sent"] = ~chunk["to"].str.contains(INTERNAL_EMAIL_DOMAIN, na=False)

        # Pre-computing attachment presence
        chunk["has_attachment"] = chunk["attachments"].notnull()

        # Deriving email-related features per chunk
        partial = chunk.groupby(MERGE_COLS, observed=True, sort=False).agg(
            emails_sent=("to", "count"),
            external_emails_sent=("external_emails_sent", "sum"),
            attachments_sent=("has_attachment", "sum"),
            off_hours_emails=("off_hours", "sum"),
        ).reset_index()

        # Accumulating deduplicated tuples (user, pc, day, to) for unique recipient counting
        identity_frames.append(chunk[MERGE_COLS + ["to"]].drop_duplicates())
        partial_aggs.append(partial)
        del chunk, partial

    # Computes aggregation and unique count operations across all chunks
    print(f"  Combining {len(partial_aggs)} email chunks...")
    combined = combine_partial_aggregations(partial_aggs, MERGE_COLS)
    unique_recipients = build_unique_count(identity_frames, MERGE_COLS, "to", "unique_recipients")
    features = combined.merge(unique_recipients, on=MERGE_COLS, how="left")

    # Returns an additional frame consisting of (user, day, to) granularity    
    if return_identity_frame:
        # Collapses identity frames
        email_identity = (
            pd.concat(identity_frames, ignore_index=True)[["user", "day", "to"]]
            .drop_duplicates()
        )
        return features, email_identity

    return features


def extract_http_features_chunked(
    filepath: str,
    work_hours: tuple = (9, 17),
    chunksize: int = 50_000,
    max_rows: int | None = None,
    return_identity_frame: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Memory-efficient HTTP feature extraction via chunked CSV reading.

    Additive counts are accumulated per chunk then combined via `combine_partial_aggregations`.
    Unique domains visited are tracked via `build_unique_count`.

    Args:
        filepath: Absolute path to http.csv
        work_hours: The range describing regular work hours based on a 24-hour format
        chunksize: Number of rows per chunk
        return_identity_frame: When True, also returns a deduplicated (user, day, domain) DataFrame

    Returns:
        pd.DataFrame: Aggregated web browsing features per (user, pc, day).
        If return_identity_frame is True, returns a (features, identity_frame) tuple.
    """
    MERGE_COLS = ["user", "pc", "day"]
    partial_aggs = []
    identity_frames = []

    for i, chunk in enumerate(load_log_in_chunks(filepath, USECOLS_MAP["http"], DTYPE_MAP, chunksize, max_rows), start=1):
        print(f"  HTTP chunk {i}...")
        chunk = normalize_shared_columns(chunk, sort=False)

        # Defining off-hours web activity
        hour = chunk["timestamp"].dt.hour
        chunk["off_hours"] = (hour < work_hours[0]) | (hour > work_hours[1])

        # Vectorized URL normalization and domain extraction
        url = chunk["url"].fillna("").astype(str).str.strip().str.lower()
        chunk["url"] = url
        chunk["url_length"] = url.str.len()
        chunk["domain"] = (
            url.str.replace(r"^https?://", "", regex=True)
               .str.split("/", n=1).str[0]
        )

        # Boolean flags
        activity = chunk["activity"]
        chunk["is_www_visit"] = (activity == "WWW Visit")
        chunk["is_www_download"] = (activity == "WWW Download")
        chunk["is_www_upload"] = (activity == "WWW Upload")

        domain = chunk["domain"]
        chunk["is_job_site"] = domain.isin(JOB_DOMAINS)
        chunk["is_cloud_storage"] = domain.isin(CLOUD_STORAGE_DOMAINS)
        chunk["is_suspicious_domain"] = domain.isin(SUSPICIOUS_DOMAINS)
        chunk["is_long_url"] = (chunk["url_length"] >= LONG_URL_THRESHOLD)

        # Grouping chunk and aggregating features
        agg_chunk = chunk.groupby(MERGE_COLS, observed=True, sort=False, dropna=False).agg(
            http_total_requests=("url", "count"),
            http_visit_count=("is_www_visit", "sum"),
            http_download_count=("is_www_download", "sum"),
            http_upload_count=("is_www_upload", "sum"),
            http_jobsite_visits=("is_job_site", "sum"),
            http_cloud_storage_visits=("is_cloud_storage", "sum"),
            http_suspicious_site_visits=("is_suspicious_domain", "sum"),
            off_hours_http_requests=("off_hours", "sum"),
            http_long_url_count=("is_long_url", "sum"),
        ).reset_index()

        # Accumulating deduplicated tuples (user, pc, day, domain) for unique domain counting
        identity_frames.append(chunk[MERGE_COLS + ["domain"]].drop_duplicates())
        partial_aggs.append(agg_chunk)
        del chunk, agg_chunk

    # Computes aggregation and unique count operations across all chunks
    print(f"  Combining {len(partial_aggs)} HTTP chunks...")
    combined = combine_partial_aggregations(partial_aggs, MERGE_COLS)
    print(f"  Building unique count domains...")
    unique_domains = build_unique_count(identity_frames, MERGE_COLS, "domain", "unique_domains_visited")
    features = combined.merge(unique_domains, on=MERGE_COLS, how="left")

    # Returns an additional frame consisting of (user, day, domain) granularity
    if return_identity_frame:
        # Collapsing identity frames
        http_identity = (
            pd.concat(identity_frames, ignore_index=True)[["user", "day", "domain"]]
            .drop_duplicates()
        )
        return features, http_identity

    return features


def merge_behavioral_features(feature_tables: list[pd.DataFrame], merge_cols: list=["user", "pc", "day"]) -> pd.DataFrame:
    """
    Merges multiple aggregated behavioral feature tables into a single dataset.
    
    Args:
        feature_tables: A list of aggregated features dataframes to merge, where each table must contain the merge columns
        merge_cols: Key columns to merge on
        
    Returns:
        pd.DataFrame: A unified table with missing activity filled with zeros
    """
    # Filtering out empty tables
    valid_tables = [df for df in feature_tables if ((df is not None) and (not df.empty))]
    
    if not valid_tables:
        raise ValueError("No valid feature tables provided for merging")

    # N-way concatenation on shared index
    indexed = [t.set_index(merge_cols) for t in valid_tables]
    merged_df = pd.concat(indexed, axis=1).reset_index()

    # Identifying feature columns, excluding identifiers
    feature_cols = [col for col in merged_df.columns if col not in merge_cols]

    # Filling missing feature values with zero
    merged_df[feature_cols] = merged_df[feature_cols].fillna(0).astype("int32")
    
    # Sorting dataframe for consistency
    merged_df.sort_values(by=merge_cols, inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    
    # Ensuring no duplicate rows are 
    if merged_df.duplicated(subset=merge_cols).any():
        raise ValueError("Duplicate rows detected after merging feature tables.")
    
    return merged_df


def add_pc_features(df: pd.DataFrame, min_history: int=10) -> pd.DataFrame:
    """
    Adds PC-derived behavioral history to an aggregated behavioral matrix.
    
    Args:
        df: The behavioral matrix
        min_history: Minimum prior observations required before flagging new PC usage as abnormal activity
        
    Returns:
        pd.DataFrame: A behavioral matrix with added user-to-PC behavioral history
    """
    df = df.copy()
    df.sort_values(by=["user", "day", "pc"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Tracking historical PC usage counts
    df["pc_prior_use_count"] = df.groupby(["user", "pc"], observed=True, sort=False).cumcount()
    df["user_total_prior_days"] = df.groupby("user", observed=True, sort=False).cumcount()
    df["pc_seen_before"]  = (df["pc_prior_use_count"] > 0).astype(int)

    df["pc_prior_use_ratio"] = np.where(
        df["user_total_prior_days"] > 0,
        df["pc_prior_use_count"] / df["user_total_prior_days"],
        0.0
    )

    # Identifying a user's primary PC (i.e., most-used PC)
    primary_pc_map = (
        df.groupby(["user", "pc"], observed=True, sort=False)
        .size()
        .reset_index(name="count")
        .sort_values(["user", "count", "pc"], ascending=[True, False, True])
        .drop_duplicates(subset=["user"])
        .set_index("user")["pc"]
        .to_dict()
    )

    df["pc_is_primary"] = (df["pc"] == df["user"].map(primary_pc_map).astype(df["pc"].dtype)).astype(int)

    # Tracking the number of distinct PCs previously used
    first_seen = ~df.duplicated(subset=["user", "pc"], keep="first")
    df["distinct_pcs_used_prior"] = (
        first_seen.groupby(df["user"], observed=True).cumsum() - first_seen.astype(int)
    )

    # Tracking the number of unique PC's used on a given day
    same_day_counts = (
        df.groupby(["user", "day"], observed=True, sort=False)["pc"]
        .transform("nunique")
    )
    
    df["n_pcs_used_today"] = same_day_counts
    
    # Identifies new PC usage after an established history
    df["new_pc_after_stable_history"] = (
        (df["pc_prior_use_count"] == 0) &
        (df["user_total_prior_days"] >= min_history)
    ).astype(int)
    
    df.drop(columns=["user_total_prior_days"], inplace=True)
    
    return df


def build_layer_a(
    cert_path: str,
    work_hours: tuple = (9, 17),
    return_nunique_frames: bool = False,
    sample_n: int | None = DEFAULT_SAMPLE_LOG_ROWS
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Builds the complete layer A drill-down-ready dataset at the (user, pc, day) level.

    Args:
        cert_path: The base path containing the CERT dataset
        work_hours: The range describing regular work hours based on a 24-hour format
        return_nunique_frames: When True, also returns a dict of identity frames needed
            by collapse_layer() to compute true per-(user, day) nunique values at Layer B.
            Keys are the output column names; values are (source_df, value_col) tuples.

    Returns:
        pd.DataFrame: Layer A dataset at the (user, pc, day) level.
        If return_nunique_frames is True, returns a (layer_a_df, nunique_frames) tuple.
    """
    # Loading the raw log files from the CERT dataset
    print("Loading raw CERT logs (sample mode)...")
    raw_logs = load_raw_logs(cert_path, sample_n=sample_n)

    # Normalizing and validating files
    normalized_logs = {}

    print("Normalizing CERT logs...")
    for name, df in raw_logs.items():
        if isinstance(df, dict) and df.get("chunked"):
            print(f"  {name}.csv: deferred (will normalize per-chunk)")
            normalized_logs[name] = df["path"]
        else:
            print(f"  Normalizing {name}.csv...")
            normalized_logs[name] = normalize_shared_columns(df)

    if return_nunique_frames:
        print("Extracting logon features...")
        logon_ft = extract_logon_features(normalized_logs["logon"], work_hours)
        print("Extracting file features...")
        file_features, file_id = extract_file_features(normalized_logs["file"], work_hours, return_identity_frame=True)
        print("Extracting device (USB) features...")
        device_ft = extract_device_features(normalized_logs["device"], work_hours)
        print("Extracting email features (chunked)...")
        email_features, email_id = extract_email_features_chunked(
            normalized_logs["email"],
            work_hours,
            max_rows=raw_logs["email"]["sample_n"],
            return_identity_frame=True,
        )
        print("Extracting HTTP features (chunked)...")
        http_features, http_id = extract_http_features_chunked(
            normalized_logs["http"],
            work_hours,
            max_rows=raw_logs["http"]["sample_n"],
            return_identity_frame=True,
        )

        feature_tables = [logon_ft, file_features, device_ft, email_features, http_features]
    else:
        print("Extracting logon features...")
        logon_ft = extract_logon_features(normalized_logs["logon"], work_hours)
        print("Extracting file features...")
        file_ft = extract_file_features(normalized_logs["file"], work_hours)
        print("Extracting device (USB) features...")
        device_ft = extract_device_features(normalized_logs["device"], work_hours)
        print("Extracting email features (chunked)...")
        email_ft = extract_email_features_chunked(
            normalized_logs["email"],
            work_hours,
            max_rows=raw_logs["email"]["sample_n"],
        )
        print("Extracting HTTP features (chunked)...")
        http_ft = extract_http_features_chunked(
            normalized_logs["http"],
            work_hours,
            max_rows=raw_logs["http"]["sample_n"],
        )
        feature_tables = [logon_ft, file_ft, device_ft, email_ft, http_ft]

    # Merging the feature tables
    print("Merging behavioral feature tables...")
    behavioral_matrix = merge_behavioral_features(feature_tables)

    # Adding pc behavioral features
    print("Adding PC behavioral features...")
    layer_a_matrix = add_pc_features(behavioral_matrix)
    print(f"Layer A complete — {len(layer_a_matrix):,} rows, {len(layer_a_matrix.columns)} features.")

    if return_nunique_frames:
        nunique_frames = {
            "unique_files_accessed": (file_id,  "filename"),
            "unique_recipients": (email_id, "to"),
            "unique_domains_visited": (http_id,  "domain"),
        }
        return layer_a_matrix, nunique_frames

    return layer_a_matrix


def save_dataset(dataset: pd.DataFrame, filename: str, output_dir: str=DEFAULT_OUTPUT_DIR) -> str:
    """
    Saves the UEBA-enhanced dataset to the specified path as a CSV file.
    
    Args:
        dataset: The UEBA-enhanced dataset
        file_name: The desired name of the CSV dataset
        output_dir: Directory where processed outputs are saved
        
    Returns:
        str: Full path to the saved dataset
    """
    # Ensures directory exists
    save_path = os.path.join(os.getcwd(), "processed_datasets", output_dir)
    os.makedirs(save_path, exist_ok=True)
    
    # Creates full file path
    file_path = os.path.join(save_path, filename)
    
    # Saving the dataset
    dataset.to_csv(file_path)
    print(f"Dataset successfully saved to: {file_path}")

    return file_path


def chronological_split(
    csv_path: str | None=None,
    df: pd.DataFrame | None=None,
    split_ratio: float=0.9
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and creates a chronological split for a UEBA-enhanced dataset.

    Either a path to a CSV file or a DataFrame can be provided. The split ratio determines the percentage that
    will be used for model training. The remaining percentage will be used for model validation.

    Args:
        csv_path: Path where the processed UEBA dataset is stored
        df: UEBA-enhanced dataset
        split_ratio: The ratio to dedicate to model training

    Returns:
        tuple: A training and testing DataFrame
    """
    if csv_path is None and df is None:
        raise ValueError("Please provide either a CSV path or a DataFrame to create a split.")

    if df is None:
        df = pd.read_csv(csv_path, index_col=0)

    # Normalize "user" and "day" columns
    df["user"] = df["user"].str.strip().str.lower()
    df["day"] = pd.to_datetime(df["day"]).dt.normalize()

    # Ensure sorted globally by time
    df = df.sort_values("day").reset_index(drop=True)

    unique_days = np.sort(df["day"].unique())
    cutoff_index = int(len(unique_days) * split_ratio)
    cutoff_day = unique_days[cutoff_index]

    train_df = df[df["day"] <= cutoff_day]
    test_df  = df[df["day"] > cutoff_day]

    return train_df, test_df


# Layer B

# Constants
LAYER_B_ID_COLS = ["user", "day"]

LAYER_B_SUM_COLS = [
    "logon_count",
    "logoff_count",
    "off_hours_logon",
    "file_open_count",
    "file_write_count",
    "file_copy_count",
    "file_delete_count",
    "off_hours_files_accessed",
    "usb_insert_count",
    "usb_remove_count",
    "off_hours_usb_usage",
    "emails_sent",
    "external_emails_sent",
    "attachments_sent",
    "off_hours_emails",
    "http_total_requests",
    "http_visit_count",
    "http_download_count",
    "http_upload_count",
    "http_jobsite_visits",
    "http_cloud_storage_visits",
    "http_suspicious_site_visits",
    "off_hours_http_requests",
    "http_long_url_count",
]

LAYER_B_MAX_COLS = [
    "pc_seen_before",
    "new_pc_after_stable_history"
]

LAYER_B_MEAN_COLS = [
    "pc_prior_use_ratio",
    "pc_is_primary"
]

LAYER_B_CONTEXT_MAX_COLS = [
    "distinct_pcs_used_prior"
]

LAYER_B_UEBA_BASE_FEATURES = [
    "logon_count",
    "logoff_count",
    "off_hours_logon",
    "file_open_count",
    "file_write_count",
    "file_copy_count",
    "file_delete_count",
    "unique_files_accessed",
    "off_hours_files_accessed",
    "usb_insert_count",
    "usb_remove_count",
    "off_hours_usb_usage",
    "emails_sent",
    "external_emails_sent",
    "attachments_sent",
    "off_hours_emails",
    "unique_recipients",
    "http_total_requests",
    "http_visit_count",
    "http_download_count",
    "http_upload_count",
    "http_jobsite_visits",
    "http_cloud_storage_visits",
    "http_suspicious_site_visits",
    "off_hours_http_requests",
    "http_long_url_count",
    "unique_domains_visited",
    "pcs_used_count",
    "non_primary_pc_used_flag",
    "non_primary_pc_http_requests_flag",
    "non_primary_pc_usb_flag",
    "non_primary_pc_file_copy_flag",
]

# Functions
def collapse_layer(
    layer_a_dataset: pd.DataFrame,
    required_cols: list = ["user", "pc", "day"],
    nunique_frames: dict[str, tuple[pd.DataFrame, str]] | None = None
) -> pd.DataFrame:
    """
    Collapses the layer A (user, pc, day) behavioral matrix into a layer B (user, day) matrix.

    Args:
        layer_a_dataset: Layer A DataFrame at the (user, day, pc) level
        required_cols: Required key columns in the inputted dataset
        nunique_frames: Optional mapping of output column name → (source_df, value_col).
            For each entry, computes the true per-(user, day) nunique of value_col from
            the raw event DataFrame rather than summing per-PC nunique values, which would
            overcount items appearing on multiple PCs.

    Returns:
        pd.DataFrame: Layer B dataframe at the (user, day) level
    """
    df = layer_a_dataset.copy()
    df.sort_values(by=required_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Ensuring no required columns are missing
    missing = set(required_cols).difference(df.columns)
    if missing:
        raise ValueError(f"Layer A dataset missing required columns: {sorted(missing)}")
    
    # Mapping feature column to their corresponding operation
    agg_map = {}
    
    for col in LAYER_B_SUM_COLS:
        if col in df.columns:
            agg_map[col] = "sum"
            
    for col in LAYER_B_MAX_COLS:
        if col in df.columns:
            agg_map[col] = "max"
            
    for col in LAYER_B_MEAN_COLS:
        if col in df.columns:
            agg_map[col] = "mean"
            
    for col in LAYER_B_CONTEXT_MAX_COLS:
        if col in df.columns:
            agg_map[col] = "max"
            
    print("  Aggregating per-PC behavioral features...")
    layer_b_df = df.groupby(by=["user", "day"], as_index=False, observed=True, sort=False).agg(agg_map)

    # Defining non-primary PC column
    if "pc_is_primary" in df.columns:
        non_primary_mask = (df["pc_is_primary"] == 0).astype(int)
    else:
        non_primary_mask = pd.Series(0, index=df.index)

    # Extracting counts from different logs
    http_source = df["http_total_requests"] if "http_total_requests" in df.columns else 0
    usb_source = df["usb_insert_count"] if "usb_insert_count" in df.columns else 0
    file_copy_source = df["file_copy_count"] if "file_copy_count" in df.columns else 0

    # Deriving cross-channel flags
    print("  Computing cross-channel flags...")
    derived = (
        df.assign(
            non_primary_pc_used=non_primary_mask,
            non_primary_http=non_primary_mask * http_source,
            non_primary_usb=non_primary_mask * usb_source,
            non_primary_file_copy=non_primary_mask * file_copy_source,
        )
        .groupby(["user", "day"], observed=True, sort=False)
        .agg(
            pcs_used_count=("pc", "nunique"),
            non_primary_pc_used_flag=("non_primary_pc_used", "sum"),
            non_primary_pc_http_requests_flag=("non_primary_http", "sum"),
            non_primary_pc_usb_flag=("non_primary_usb", "sum"),
            non_primary_pc_file_copy_flag=("non_primary_file_copy", "sum"),
        )
        .reset_index()
    )

    layer_b_df = layer_b_df.merge(derived, on=["user", "day"], how="left")

    # Recompute nunique columns from raw event frames
    if nunique_frames:
        print("  Recomputing true nunique counts from raw event frames...")
        for col_name, (source_df, value_col) in nunique_frames.items():
            true_nunique = (
                source_df.groupby(["user", "day"], observed=True, sort=False)[value_col]
                .nunique()
                .reset_index(name=col_name)
            )
            layer_b_df = layer_b_df.merge(true_nunique, on=["user", "day"], how="left")

    # Renaming primary PC column
    if "pc_is_primary" in layer_b_df.columns:
        layer_b_df.rename(columns={"pc_is_primary": "primary_pc_activity_ratio"}, inplace=True)

    # Filling missing values in feature columns
    feature_cols = [col for col in layer_b_df.columns if col not in LAYER_B_ID_COLS]
    layer_b_df[feature_cols] = layer_b_df[feature_cols].fillna(0)
    
    # Ensuring there's not duplicate rows in the final dataset
    if layer_b_df.duplicated(subset=LAYER_B_ID_COLS).any():
        raise ValueError("Duplicate rows detected in layer B after collapse.")
    
    return layer_b_df


def get_layer_b_features(df: pd.DataFrame) -> list[str]:
    """
    Returns the layer B columns that should receive z-scores and rolling deltas.
    
    Args:
        df: The collapsed layer B dataset
        
    Returns
        str: A list of feature column names
    """
    col_names = [col for col in LAYER_B_UEBA_BASE_FEATURES if col in df.columns]
    return col_names


def apply_ueba_enhancements(df: pd.DataFrame, feature_cols: list, rolling_window: int=5) -> pd.DataFrame:
    """
    Applying UEBA-specific enhancements to a behavioral matrix such as:
    - Per-user causal z-score deviations based only on prior history
    - Causal rolling mean deltas that exclude the current row
    - Cross-channel risk flags
    
    Args:
        df: A layer B dataset at the (user, day) granularity
        feature_cols: Feature columns to apply z-scores and rolling deltas to
        rolling_window: Window size in days for rolling statistics
        
    Returns:
        pd.DataFrame: An enhanced UEBA-ready feature dataset
    """
    df = df.copy()
    df.sort_values(by=["user", "day"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Shifting all feature columns within each user group in one pass
    print("  Computing per-user causal z-score deviations...")
    shifted = df.groupby("user", observed=True, sort=False)[feature_cols].shift(1)
    user_shifted = shifted.groupby(df["user"], observed=True, sort=False)

    # Expanding mean/std across all columns at once
    prior_mean = user_shifted.expanding(min_periods=1).mean().reset_index(level=0, drop=True)
    prior_std = (
        user_shifted.expanding(min_periods=2).std()
        .reset_index(level=0, drop=True)
        .replace(0, np.nan)
    )

    # Computing bounded z-scores
    z_scores = ((df[feature_cols] - prior_mean) / prior_std).fillna(0.0)
    z_scores = z_scores.clip(-10, 10)
    z_scores.columns = [f"{col}_zscore" for col in feature_cols]

    # Computing temporal rolling deltas
    print("  Computing causal rolling mean deltas...")
    prior_rolling = (
        user_shifted.rolling(window=rolling_window, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )
    rolling_deltas = (df[feature_cols] - prior_rolling).fillna(0.0)
    rolling_deltas.columns = [f"{col}_rolling_delta" for col in feature_cols]

    df = pd.concat([df, z_scores, rolling_deltas], axis=1)
        
    # Extracting off-hour columns
    print("  Adding cross-channel risk flags...")
    off_hours_cols = [col for col in feature_cols if col.startswith("off_hours")]
    
    if off_hours_cols:
        df["off_hours_activity_flag"] = (df[off_hours_cols].sum(axis=1) > 0).astype(int)
    else:
        df["off_hours_activity_flag"] = 0
        
    # Defining cross-channel risk flags
    df["usb_file_activity_flag"] = ((df.get("usb_insert_count", 0) > 0) & (df.get("file_write_count", 0) > 0)).astype(int)
        
    df["external_comm_activity_flag"] = (df.get("external_emails_sent", 0) > 0).astype(int)
    
    df["jobsite_usb_activity_flag"] = ((df.get("http_jobsite_visits", 0) > 0) & (df.get("usb_insert_count", 0) > 0)).astype(int)
    
    df["suspicious_upload_flag"] = ((df.get("http_jobsite_visits", 0) > 0) & (df.get("http_upload_count", 0) > 0)).astype(int)
    
    df["cloud_upload_flag"] = ((df.get("http_upload_count", 0) > 0) & (df.get("http_cloud_storage_visits", 0) > 0)).astype(int)
    
    df["non_primary_pc_risk_flag"] = (
        (df.get("non_primary_pc_http_requests_flag", 0) > 0) |
        (df.get("non_primary_pc_usb_flag", 0) > 0) |
        (df.get("non_primary_pc_file_copy_flag", 0) > 0)
    ).astype(int)

    return df


def build_layer_b(
    layer_a_df: pd.DataFrame,
    rolling_window: int = 5,
    nunique_frames: dict[str, tuple[pd.DataFrame, str]] | None = None
) -> pd.DataFrame:
    """
    Builds the final layer B user-day UEBA modeling matrix.

    Args:
        layer_a_df: The behavioral matrix produced in layer A
        rolling_window: The window size in days to compute the rolling delta
        nunique_frames: Passed through to collapse_layer(). Maps output column name →
            (source_df, value_col) for features that require true per-(user, day) nunique
            computed from raw event data rather than summing per-PC nunique values.

    Returns:
        pd.DataFrame: A model-ready UEBA dataset at the (user, day) level
    """
    print("Collapsing Layer A to (user, day) granularity...")
    layer_b_df = collapse_layer(layer_a_df, nunique_frames=nunique_frames)
    feature_cols = get_layer_b_features(layer_b_df)
    print("Applying UEBA enhancements (z-scores, rolling deltas, risk flags)...")
    layer_b_df = apply_ueba_enhancements(layer_b_df, feature_cols=feature_cols, rolling_window=rolling_window)
    print(f"Layer B complete — {len(layer_b_df):,} rows, {len(layer_b_df.columns)} features.")
    return layer_b_df


# Drill-Down Functionality

# Functions
def get_drilldown(layer_a_df: pd.DataFrame, user: str, day: str, sorting_cols: list[str] | None=None) -> pd.DataFrame:
    """
    Creates drill-down support which serves as a lookup layer built from layer A's dataset.
    
    Args:
        layer_a_df: The behavioral matrix produced in layer A
        user: User to investigate
        day: Day to investigate (YYYY-MM-DD format)
        sorting_cols: Columns used to sort returned values
        
    Returns:
        pd.DataFrame: 
    """
    # Lookup of the specified user and day
    drilldown_df = layer_a_df[(layer_a_df["user"] == user) & (layer_a_df["day"] == day)].copy()
    
    # Sorts rows based on the specified columns
    if sorting_cols:
        drilldown_df.sort_values(
            by=sorting_cols,
            ascending=False,
            inplace=True,
            ignore_index=True)
    
    return drilldown_df