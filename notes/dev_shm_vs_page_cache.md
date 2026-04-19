# /dev/shm vs OS Page Cache for Dataset Loading

## Conclusion: Not worth it

The dataset (`yiddish_lines.npy`) is 3.7 GB on a 62 GB RAM machine. After the first access,
the OS page cache holds the entire file in RAM. Every `data[idx]` call in the DALI source
callback hits RAM, not disk — identical to /dev/shm performance.

## When /dev/shm would actually help

| Scenario | Applies here? |
|---|---|
| Dataset doesn't fit in RAM | No — 3.7 GB in 62 GB |
| Memory pressure evicts pages | No — 53 GB available |
| First epoch cold-start is slow | Irrelevant — page cache survives between runs |
| NUMA: process and cache on different nodes | Only on multi-socket servers |

## Additional note

Even if data loading were slower, the training runs at 98.5% GPU kernel density — the GPU
is never waiting for data. Moving the file to /dev/shm would have zero effect on step time.
