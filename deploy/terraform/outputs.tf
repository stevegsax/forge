output "public_ip" {
  description = "Elastic IP of the instance."
  value       = aws_eip.forge.public_ip
}

output "gateway_endpoint" {
  description = "Set this as FORGE_TEMPORAL_ADDRESS / PBOOK_TEMPORAL_ADDRESS on clients."
  value       = "${aws_eip.forge.public_ip}:443"
}

output "instance_id" {
  description = "Use with: aws ssm start-session --target <id>"
  value       = aws_instance.forge.id
}

output "ocr_bucket" {
  value = aws_s3_bucket.ocr.bucket
}

output "next_steps" {
  value = <<-EOT
    1. Issue the server cert for THIS endpoint and upload TLS material to SSM:
         cd ../certs && ./gen-server-cert.sh ${aws_eip.forge.public_ip}
         (then put TLS_SERVER_CERT / TLS_SERVER_KEY / TLS_CLIENT_CA in ${var.ssm_prefix})
       If you used a DNS name, point it at ${aws_eip.forge.public_ip} and use that name instead.
    2. Put secrets in SSM: ${var.ssm_prefix}/{SUPABASE_FORGE_DB_URL,SUPABASE_HOST,SUPABASE_USER,
       SUPABASE_TEMPORAL_PWD,ANTHROPIC_API_KEY,SAX_GITHUB_TOKEN[,MISTRAL_API_KEY,OPENAI_API_KEY]}
    3. Reboot (or re-run bootstrap) so the gateway picks up the certs.
    4. Issue a client cert per user: cd ../certs && ./gen-client-cert.sh <name>
    5. Clients: see deploy/client/ONBOARDING.md
  EOT
}
