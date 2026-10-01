set -e
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
echo "=== verify ==="
python -c "
import torch
print('torch', torch.__version__, '| cuda', torch.version.cuda, '| avail', torch.cuda.is_available())
p = torch.cuda.get_device_properties(0)
print(f'{p.name} | VRAM {p.total_memory/1e9:.1f}GB | SM {p.multi_processor_count} | cap {p.major}.{p.minor}')
x = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
print('bf16 matmul ok:', (x @ x).float().abs().mean().item() > 0)
"
