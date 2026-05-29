# Self-contained networking: use the account's default VPC + a public subnet.
# Swap these data sources for explicit vpc_id/subnet_id if you run a custom VPC.

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# Latest Amazon Linux 2023 AMI (x86_64) via the public SSM parameter.
data "aws_ssm_parameter" "al2023" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}

locals {
  base_tags = merge({ Project = "forge", ManagedBy = "terraform" }, var.tags)
}
