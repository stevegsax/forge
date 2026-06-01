# Only the mTLS gateway port is open. No SSH (22), no Temporal gRPC (7233), no
# UI (8080) — those are loopback-only and reached via SSM port-forwarding.
resource "aws_security_group" "instance" {
  name        = "forge-instance-sg"
  description = "Forge: inbound mTLS gateway only; all egress"
  vpc_id      = data.aws_vpc.default.id
  tags        = local.base_tags

  ingress {
    description = "Temporal mTLS gateway (gRPC over TLS, client-cert required)"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = var.allowed_client_cidrs
  }

  egress {
    description = "All outbound (Supabase, S3, Anthropic, Mistral, GitHub, OpenAI)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
