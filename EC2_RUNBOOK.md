# AWS EC2 vLLM runbook

The AWS OpenClaw host is an EC2 `g6.xlarge` in `us-east-1`, backed by an NVIDIA
L4 with 24 GB VRAM. It serves `Qwen/Qwen3.5-9B` as the shared model name
`openclaw` through vLLM's OpenAI-compatible API.

## Current host

| Setting | Value |
|---------|-------|
| AWS profile | `ml-prep-deploy` (IAM Identity Center / SSO) |
| Region | `us-east-1` |
| Instance ID | `i-0d7385b600fd36704` |
| Instance type | `g6.xlarge` |
| GPU | NVIDIA L4, 24 GB VRAM |
| Root storage | 250 GB encrypted gp3 |
| OS user | `ubuntu` |
| vLLM listener | `127.0.0.1:11434` only |
| Public API | `https://98-80-123-250.sslip.io/v1/*` (bearer auth) |
| System services | `openclaw-vllm-ec2.service`, `openclaw-caddy-ec2.service` |

The instance has no Elastic IP, so its public address can change after a
stop/start cycle. The EC2 security group admits SSH only from the current admin
public IPv4 `/32`. It exposes 80/443 for Caddy, but never exposes port 11434.

## Log in and open the API tunnel

Refresh the AWS SSO session and look up the current public IP:

```bash
aws sso login --profile ml-prep-deploy

EC2_IP=$(aws ec2 describe-instances \
  --profile ml-prep-deploy \
  --region us-east-1 \
  --instance-ids i-0d7385b600fd36704 \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)
```

SSH normally:

```bash
ssh -i ~/.ssh/id_ed25519 "ubuntu@$EC2_IP"
```

Or keep a tunnel open so local clients can use the API without any public vLLM
port. Native vLLM bearer authentication still applies through the tunnel:

```bash
ssh -i ~/.ssh/id_ed25519 -N \
  -L 11434:127.0.0.1:11434 "ubuntu@$EC2_IP"
```

From the machine holding that tunnel:

```bash
API_KEY=$(ssh -i ~/.ssh/id_ed25519 "ubuntu@$EC2_IP" \
  "sudo sed -n 's/^VLLM_API_KEY=//p' /etc/openclaw-vllm-ec2.env")

curl -s http://127.0.0.1:11434/v1/models \
  -H "Authorization: Bearer $API_KEY"
# OpenAI base URL: http://127.0.0.1:11434/v1
# Model name:      openclaw
```

The public endpoint uses the same OpenAI request shape and requires the key
stored in the instance's root-only environment file:

```bash
API_KEY=$(ssh -i ~/.ssh/id_ed25519 "ubuntu@$EC2_IP" \
  "sudo sed -n 's/^VLLM_API_KEY=//p' /etc/openclaw-vllm-ec2.env")

curl https://98-80-123-250.sslip.io/v1/chat/completions \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "openclaw",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 150
  }'
```

The generated `sslip.io` name follows the current public IP. If the instance is
stopped and started without an Elastic IP, rerun the installer to update the
hostname and certificate after the IP changes. Set `OPENCLAW_PUBLIC_HOST` when
running the installer to use a real DNS name instead.

## Install and operate the service

The service is a system unit, not a login-scoped user unit. It starts after
Docker at boot and recreates the pinned container if vLLM exits.

```bash
./install-vllm-ec2.sh

sudo systemctl status openclaw-vllm-ec2.service
sudo systemctl restart openclaw-vllm-ec2.service
sudo systemctl stop openclaw-vllm-ec2.service
sudo journalctl -u openclaw-vllm-ec2.service -f
sudo systemctl status openclaw-caddy-ec2.service
sudo journalctl -u openclaw-caddy-ec2.service -f
```

Use `./install-vllm-ec2.sh --no-start` to install or update the unit without
interrupting a live model or benchmark. Start/restart it explicitly afterward.

## Pinned serving configuration

The source of truth is
[`vllm/openclaw-vllm-ec2.service`](./vllm/openclaw-vllm-ec2.service):

- image: `vllm/vllm-openai:v0.18.1`
- model: `Qwen/Qwen3.5-9B`, served as `openclaw`
- language model only; thinking disabled in the server chat template
- BF16 model weights and FP8 KV cache
- 16,384-token maximum model length
- 95% GPU-memory utilization and at most four concurrent sequences
- eager execution
- Hugging Face cache persisted at `/home/ubuntu/.cache/huggingface`
- host API bound to `127.0.0.1:11434`
- native vLLM bearer-key validation, with the key stored mode `0600` in
  `/etc/openclaw-vllm-ec2.env`
- Caddy automatic HTTPS on ports 80/443, proxying only `/v1/*`

The model cache is roughly 19 GB. The 250 GB root volume leaves ample room for
this model, container layers, benchmark results, and additional quantized model
experiments without repeating the original small-disk mistake.

## Stop/start the EC2 instance

Stopping the EC2 instance stops compute billing while retaining the EBS volume.
Starting it again causes Docker and then `openclaw-vllm-ec2.service` to start;
the cached model is reused, though loading it into VRAM still takes time.

```bash
aws ec2 stop-instances --profile ml-prep-deploy --region us-east-1 \
  --instance-ids i-0d7385b600fd36704

aws ec2 start-instances --profile ml-prep-deploy --region us-east-1 \
  --instance-ids i-0d7385b600fd36704
```

After a stop/start, repeat the public-IP lookup above and update the SSH security
group's `/32` rule if the administrator's public IP has changed.

Benchmark results and comparisons with `beast` and `jetson-orin` live in
[`BENCHMARKS.md`](./BENCHMARKS.md).
