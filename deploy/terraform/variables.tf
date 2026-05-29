variable "region" {
  type        = string
  description = "AWS region. Co-locate with your Supabase project to cut latency."
  default     = "us-east-1"
}

variable "instance_type" {
  type        = string
  description = "EC2 instance type. t3.large (2 vCPU/8GB) is comfortable; t3.medium works at low volume. Avoid spot (long workflows)."
  default     = "t3.large"
}

variable "allowed_client_cidrs" {
  type        = list(string)
  description = <<-EOT
    CIDRs allowed to reach the mTLS gateway on :443. mTLS is the primary auth
    gate, so ["0.0.0.0/0"] is acceptable, but narrowing to known office/VPN
    egress IPs is recommended defense-in-depth. No default — choose consciously.
  EOT
}

variable "ocr_bucket_name" {
  type        = string
  description = "Globally-unique S3 bucket name for OCR blobs (e.g. forge-ocr-blobs-<acct>)."
}

variable "ssm_prefix" {
  type        = string
  description = "SSM Parameter Store prefix holding secrets + TLS material."
  default     = "/forge"
}

variable "root_volume_size_gb" {
  type        = number
  description = "Root gp3 size. Holds the venv, cloned repos, worktrees, logs — all disposable."
  default     = 60
}

variable "with_pbook" {
  type        = bool
  description = "Also deploy the pbook worker + run migrations (transcript ingestion)."
  default     = false
}

variable "repo_org" {
  type        = string
  description = "GitHub org/owner for the forge, sax-llm, pbook repos."
  default     = "stevegsax"
}

variable "forge_ref" {
  type        = string
  description = "Git ref (branch/tag/SHA) of forge to deploy."
  default     = "main"
}

variable "saxllm_ref" {
  type    = string
  default = "main"
}

variable "pbook_ref" {
  type    = string
  default = "main"
}

variable "tags" {
  type        = map(string)
  description = "Extra tags applied to all resources."
  default     = {}
}
