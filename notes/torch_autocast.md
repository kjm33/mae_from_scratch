  Why torch.autocast instead of casting the batch manually:

  The old approach cast inputs and weights to bf16 upfront. torch.autocast is smarter — it runs ops that are safe in bf16 (matmuls, convs) in bf16, but keeps numerically sensitive ops (softmax, layer norm,
  loss) in fp32 automatically. This gives the same speed benefit with better numerical stability.