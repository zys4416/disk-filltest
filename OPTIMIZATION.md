# LCG Performance Optimization

## Background

The original `lcg_random()` is a scalar, serial-dependency LCG. Each call depends on the
previous result, preventing CPU out-of-order execution from overlapping iterations. On fast
NVMe SSDs, this becomes the bottleneck — the CPU cannot generate random data fast enough
to saturate disk write bandwidth.

Two optimizations were applied, both producing byte-identical output to the original
`lcg_random()` loop, preserving full compatibility with existing files.

## Optimization A: 4-Lane Scalar Parallel LCG

**Principle**: LCG allows "fast-forward" — given state `x_n`, we can compute `x_{n+k}`
directly using precomputed constants `a^k` and `c_k`. This breaks the serial dependency
chain by running 4 independent LCG streams in parallel:

- Lane 0 computes: x[0], x[4], x[8], ...
- Lane 1 computes: x[1], x[5], x[9], ...
- Lane 2 computes: x[2], x[6], x[10], ...
- Lane 3 computes: x[3], x[7], x[11], ...

Each lane uses the recurrence `x_{n+4} = a^4 * x_n + c_4` with precomputed constants:

```
a   = 0x27BB2EE687B0B0FD   (original multiplier)
c   = 0xB504F32D           (original addend)
a^4 = 0x8DD29D8705EB5451   (4-step multiplier)
c_4 = 0x7BCFDF788DC37E7C   (4-step addend)
```

The 4 lanes use separate local variables (`lane0`..`lane3`), ensuring the compiler places
them in registers. The CPU's out-of-order engine overlaps the multiply operations from
independent lanes.

**Functions**: `lcg_fill_block()` (write path), `lcg_check_block()` (read/verify path).

**Expected speedup**: 3-4x for the RNG computation (breaks the multiply-latency bottleneck).

## Optimization B: AVX2 SIMD Vectorization

**Principle**: Pack all 4 lanes into a single `__m256i` register and process them with
SIMD instructions. AVX2 lacks native 64x64->64 multiply, so it is emulated using
32x32->64 operations:

```
a * x mod 2^64 = a_lo*x_lo + (a_lo*x_hi + a_hi*x_lo) << 32
```

This translates to ~8 AVX2 instructions per iteration (3x `_mm256_mul_epu32`,
1x `_mm256_srli_epi64`, 1x `_mm256_slli_epi64`, 3x `_mm256_add_epi64`), processing
all 4 lanes simultaneously.

**Runtime detection**: `lcg_init()` uses `__builtin_cpu_supports("avx2")` (GCC/Clang/MinGW)
to detect AVX2 at runtime. Function pointers (`lcg_fill`, `lcg_check`) are set to either
the AVX2 or scalar implementation. A built-in self-test at startup verifies AVX2 output
matches scalar output; on mismatch, it falls back to scalar automatically.

**Compiler support**: AVX2 functions use `__attribute__((target("avx2")))`, so no global
`-mavx2` flag is needed. The rest of the file compiles with the default ISA. Non-x86
platforms (`HAVE_X86 == 0`) always use the scalar path.

**Functions**: `lcg_fill_block_avx2()` (write path), `lcg_check_block_avx2()` (read/verify
path). The verify path uses `_mm256_cmpeq_epi64` + `_mm256_movemask_epi8` for SIMD
comparison.

**Expected speedup**: 1.5-2x on top of Optimization A (wider execution, reduced instruction
count per element).

## Compatibility

- Output is byte-identical to the original `lcg_random()` for any seed
- All existing files can be verified by the optimized binary, and vice versa
- `-S` (file size), `-s` (seed), `-f` (file limit) and all other parameters are unaffected
- Non-AVX2 CPUs automatically use the scalar 4-lane path
- Non-x86 architectures (ARM, etc.) use the scalar path via compile-time guards
- No Makefile changes required
