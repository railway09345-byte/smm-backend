"""Private R2 storage; never falls back to a local/public asset on upload failure."""
import hashlib
from functools import lru_cache
from app.core.config import settings


@lru_cache(maxsize=1)
def r2_client():
    import boto3
    from botocore.config import Config
    if not all((settings.R2_ACCOUNT_ID, settings.R2_ACCESS_KEY_ID, settings.R2_SECRET_ACCESS_KEY, settings.R2_BUCKET)):
        raise RuntimeError("Configure R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY and R2_BUCKET")
    return boto3.client("s3", endpoint_url=f"https://{settings.R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=settings.R2_ACCESS_KEY_ID, aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
        region_name="auto", config=Config(signature_version="s3v4", connect_timeout=10, read_timeout=60, retries={"max_attempts": 2}))


def store_asset(data, organization_id, brand_id, run_id, asset_id, kind):
    if min(organization_id, brand_id, run_id, asset_id) <= 0 or kind not in ("image", "video"):
        raise ValueError("Invalid tenant asset path")
    extension, mime = ("png", "image/png") if kind == "image" else ("mp4", "video/mp4")
    key = f"tenants/{organization_id}/brands/{brand_id}/runs/{run_id}/{asset_id}.{extension}"
    digest = hashlib.sha256(data).hexdigest()
    client = r2_client()
    client.put_object(Bucket=settings.R2_BUCKET, Key=key, Body=data, ContentType=mime, Metadata={"sha256": digest})
    head = client.head_object(Bucket=settings.R2_BUCKET, Key=key)
    if head["ContentLength"] != len(data) or head.get("Metadata", {}).get("sha256") != digest:
        raise RuntimeError("R2 upload verification failed")
    return f"r2://{settings.R2_BUCKET}/{key}", {"storage": "r2", "object_key": key, "sha256": digest, "size_bytes": len(data)}


def delivery_url(reference, organization_id):
    if not reference or not reference.startswith("r2://"):
        return reference
    prefix = f"r2://{settings.R2_BUCKET}/tenants/{organization_id}/"
    if not reference.startswith(prefix) or ".." in reference:
        raise ValueError("Asset is outside the current tenant")
    key = reference.split("/", 3)[3]
    return r2_client().generate_presigned_url("get_object", Params={"Bucket": settings.R2_BUCKET, "Key": key}, ExpiresIn=3600)
