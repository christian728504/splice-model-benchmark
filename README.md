# Benchmark for Splice Models

- SpliceAI
- SegmentNT
- Pangolin
- SpliceTransformer

## Setup environment
```bash
singularity build env/splice-benchmark.sif docker://clarity001/bioinformatics:splice-benchmark
singularity shell --nv --bind /zata,/data env/splice-benchmark.sif
```

```bash
uv sync
```

```bash
uv add axial-positional-embedding sinkhorn-transformer transformers spliceai gtfparse git+https://github.com/instadeepai/nucleotide-transformer@main git+https://github.com/tkzeng/Pangolin.git@main
```