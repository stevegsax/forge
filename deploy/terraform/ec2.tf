resource "aws_instance" "forge" {
  ami                    = data.aws_ssm_parameter.al2023.value
  instance_type          = var.instance_type
  subnet_id              = data.aws_subnets.default.ids[0]
  vpc_security_group_ids = [aws_security_group.instance.id]
  iam_instance_profile   = aws_iam_instance_profile.instance.name

  # No SSH key pair: management is via SSM Session Manager only.

  metadata_options {
    http_tokens   = "required" # enforce IMDSv2
    http_endpoint = "enabled"
  }

  root_block_device {
    volume_type = "gp3"
    volume_size = var.root_volume_size_gb
    encrypted   = true
  }

  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    ssm_prefix = var.ssm_prefix
    repo_org   = var.repo_org
    forge_ref  = var.forge_ref
    saxllm_ref = var.saxllm_ref
    pbook_ref  = var.pbook_ref
    ocr_bucket = var.ocr_bucket_name
    with_pbook = var.with_pbook ? "true" : "false"
  })

  tags = merge(local.base_tags, { Name = "forge" })
}

resource "aws_eip" "forge" {
  instance = aws_instance.forge.id
  domain   = "vpc"
  tags     = merge(local.base_tags, { Name = "forge" })
}
