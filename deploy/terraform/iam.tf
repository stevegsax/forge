data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "instance" {
  name               = "forge-instance-role"
  assume_role_policy = data.aws_iam_policy_document.assume.json
  tags               = local.base_tags
}

# SSM Session Manager (management without SSH / open ports).
resource "aws_iam_role_policy_attachment" "ssm_core" {
  role       = aws_iam_role.instance.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

data "aws_iam_policy_document" "instance" {
  # OCR blobs: scoped to this bucket only.
  statement {
    sid       = "OcrBucketObjects"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = ["${aws_s3_bucket.ocr.arn}/*"]
  }
  statement {
    sid       = "OcrBucketList"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.ocr.arn]
  }
  # Read only the /forge/* secrets (API keys, DB URL, TLS material).
  statement {
    sid       = "ReadForgeSecrets"
    actions   = ["ssm:GetParameter", "ssm:GetParameters", "ssm:GetParametersByPath"]
    resources = ["arn:aws:ssm:${var.region}:*:parameter${var.ssm_prefix}/*"]
  }
  # Decrypt SecureString params (they use the aws/ssm managed KMS key).
  statement {
    sid       = "DecryptSsm"
    actions   = ["kms:Decrypt"]
    resources = ["*"]
    condition {
      test     = "StringEquals"
      variable = "kms:ViaService"
      values   = ["ssm.${var.region}.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "instance" {
  name   = "forge-instance-policy"
  role   = aws_iam_role.instance.id
  policy = data.aws_iam_policy_document.instance.json
}

resource "aws_iam_instance_profile" "instance" {
  name = "forge-instance-profile"
  role = aws_iam_role.instance.name
}
