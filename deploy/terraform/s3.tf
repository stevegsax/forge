resource "aws_s3_bucket" "ocr" {
  bucket = var.ocr_bucket_name
  tags   = local.base_tags
}

resource "aws_s3_bucket_public_access_block" "ocr" {
  bucket                  = aws_s3_bucket.ocr.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "ocr" {
  bucket = aws_s3_bucket.ocr.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "ocr" {
  bucket = aws_s3_bucket.ocr.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
