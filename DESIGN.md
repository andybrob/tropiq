# TropiQ: Tropical Algebra Engine
## Technical Design Document v0.2

---

## 1. Formal Technical Specification

### 1.1 The Core Idea

The **min-plus tropical semiring** $(\mathbb{R} \cup \{+\infty\}, \oplus, \otimes)$ redefines two operations:

$$a \oplus b = \min(a, b) \qquad a \otimes b = a + b$$

"Addition" finds the minimum; "multiplication" computes sums. This is exactly the structure of shortest-path relaxation and dynamic programming recurrences. Reformulating these algorithms as tropical matrix operations enables vectorization and parallelism using the same infrastructure as dense linear algebra — BLAS-style kernels, cache blocking, SIMD, OpenMP.

**What this library is not**: a replacement for NetworkX or igraph for typical graph problems. Dijkstra with a priority queue is faster than any matrix method for sparse graphs. TropiQ targets three specific cases where the tropical reformulation provides genuine, measurable advantage over existing Python tooling.

### 1.2 The Three Use Cases

---

#### Use Case 1: Viterbi Decoding for Large-State HMMs

**Background**: A Hidden Markov Model (HMM) represents a sequence of hidden states $S_1, S_2, \ldots, S_T$ that you cannot observe directly, and a sequence of observations $O_1, O_2, \ldots, O_T$ that you can. Two matrices define it:

- **Transition matrix** $A$: $A_{ij} = P(\text{go to state } j \mid \text{currently in state } i)$
- **Emission matrix** $B$: $B_{is} = P(\text{observe } O_s \mid \text{in state } i)$

The Viterbi algorithm finds the most probable hidden state sequence given observations. In log space, its recurrence is:

$$\delta_t(s) = \max_{s'} \left[ \delta_{t-1}(s') + \log A_{s's} \right] + \log B_{s, O_t}$$

The inner operation $\max_{s'}[\cdot + \cdot]$ is a **max-plus tropical matrix-vector product**. The entire algorithm is this product repeated $T$ times.

**The problem**: for large state spaces (S = 1,000–10,000 states), the inner max runs over all $S^2$ state pairs per time step. Existing Python implementations (hmmlearn, pomegranate) use either pure Python loops or scipy sparse routines not optimized for this pattern. There is no vectorized tropical kernel available.

**The target**: a C kernel for the max-plus matrix-vector product that cache-blocks the $S \times S$ transition matrix and SIMD-vectorizes the inner loop. For $S = 5,000$ and $T = 500$ time steps, target 10–50× speedup over hmmlearn's current implementation.

**Concrete example**: fitting an HMM to daily GMV time series to infer hidden demand regimes (e.g., "promotional", "organic growth", "external shock"). With a rich enough state space to capture regional variation, $S$ easily reaches thousands.

---

#### Use Case 2: All-Pairs Shortest Paths on Dense Graphs

**When this matters**: if your graph is dense — a fully-connected cost matrix between $N$ locations in a logistics network — you need all-pairs shortest paths and Floyd-Warshall ($O(N^3)$) is unavoidable. The tropical reformulation doesn't change the asymptotic complexity, but it enables:

1. **Vectorized inner loop**: the Floyd-Warshall relaxation `D[i,j] = min(D[i,j], D[i,k] + D[k,j])` is identical to a row of a tropical matrix product. Cache-blocked implementation gives 5–10× speedup over naive NumPy.
2. **GPU offload**: the tropical matmul kernel has the same memory layout as standard matmul — it can be offloaded to a GPU with minimal changes. No existing Python library supports GPU-accelerated tropical matmul.

**Out of scope**: sparse graphs (use NetworkX + Dijkstra), single-source shortest path (same).

---

#### Use Case 3: Parametric Shortest Path Analysis

**The question standard algorithms can't answer**: given a logistics network where edge costs depend on a parameter $\lambda$ (e.g., fuel price, toll rate, time-of-day multiplier), *at what value of $\lambda$ does the optimal route change?*

Standard algorithms give you the shortest path for a fixed $\lambda$. Parametric analysis gives you the **complete breakpoint structure**: the set of $\lambda$ values where the optimal route switches, and which route is optimal on each interval.

**The tropical connection**: when edge costs are linear in $\lambda$ (i.e., $w_{ij}(\lambda) = a_{ij} + b_{ij}\lambda$), the shortest path as a function of $\lambda$ is a **tropical polynomial** — a piecewise-linear convex function. Its breakpoints are the roots of a tropical polynomial, computable algebraically without re-running Dijkstra for every $\lambda$ value.

**What this enables**: supply chain sensitivity analysis. "If fuel costs rise 20%, which routes become suboptimal? At what price point does shifting from air to sea freight become worthwhile?" These are parametric shortest path questions.

**There is essentially no Python tooling for this.** Academic implementations exist in MATLAB and specialized CAS (computer algebra systems). TropiQ would fill a real gap.

---

### 1.3 Scope Summary

| Use Case | Acceleration Mechanism | Gap Filled |
|----------|----------------------|------------|
| Viterbi (large HMMs) | Vectorized max-plus matvec | hmmlearn is slow at large S |
| Dense all-pairs SP | Cache-blocked tropical matmul, GPU | No GPU tropical matmul exists |
| Parametric analysis | Tropical polynomial roots | No Python tooling exists |

**Explicitly out of scope**: CRFs (complex, narrow use case outside NLP), sparse graph shortest paths (Dijkstra is better), HMM training / Baum-Welch (separate problem from decoding).

### 1.4 Correctness Invariants

- All operations must be exactly equivalent to scalar loop counterparts (no approximation)
- IEEE 754 `+inf` is the tropical zero for min-plus; `-inf` for max-plus
- Numerical equivalence tests against networkx (shortest path) and hmmlearn (Viterbi) are mandatory before any release

---

## 2. Architecture

### 2.1 Repository Layout

```
tropiq/
├── src/
│   └── tropiq/
│       ├── __init__.py             # Public API: re-exports user-facing symbols
│       ├── array.py                # TropicalArray: core data structure
│       ├── linalg.py               # matmul, matpow, matvec
│       ├── hmm.py                  # HMM class, viterbi(), fit() stub
│       ├── graph.py                # allpairs_shortest_path()
│       ├── parametric.py           # tropical_polynomial, breakpoints()
│       └── _backends/
│           ├── __init__.py         # Dispatch: C ext → NumPy fallback
│           └── numpy_backend.py    # Pure NumPy fallback (always available)
├── csrc/
│   ├── tropiq_core.h               # Kernel declarations
│   ├── minplus_dense.c             # Dense min-plus matmul (shortest path)
│   ├── maxplus_matvec.c            # Max-plus matrix-vector (Viterbi hot path)
│   ├── maxplus_dense.c             # Dense max-plus matmul (GPU prep / matpow)
│   └── python_bindings.c           # CPython extension module
├── tests/
│   ├── test_array.py               # TropicalArray algebra identities
│   ├── test_linalg.py              # matmul, matpow correctness
│   ├── test_hmm.py                 # Viterbi vs. hmmlearn on small examples
│   ├── test_graph.py               # allpairs vs. networkx
│   └── test_parametric.py          # Breakpoints on hand-verified examples
├── benchmarks/
│   ├── bench_viterbi.py            # TropiQ vs. hmmlearn, varying S and T
│   ├── bench_allpairs.py           # TropiQ vs. scipy/networkx, varying N
│   └── bench_parametric.py         # Breakpoint computation vs. naive sweep
├── examples/
│   ├── gmv_regime_detection.py     # HMM on synthetic GMV time series
│   └── logistics_sensitivity.py    # Parametric SP on a toy supply chain
├── docker/
│   ├── Dockerfile                  # Compiles C extension, installs library
│   └── entrypoint.sh               # Sets up Python path, runs smoke test
├── pyproject.toml
└── setup.py                        # Builds C extension via setuptools
```

### 2.2 C Extension Layer (`csrc/`)

All kernels operate on raw `double*` buffers obtained via the Python buffer protocol (NumPy arrays). No memory allocation inside kernels.

**`tropiq_core.h`**

```c
/* Min-plus dense matmul: C = A ⊗_min B, shapes (M,K) x (K,N) → (M,N) */
void minplus_matmul_f64(
    const double *A, const double *B, double *C,
    int M, int K, int N
);

/* Max-plus matrix-vector product: y = A ⊗_max x, shapes (M,K) x (K,) → (M,) */
/* Hot path for Viterbi inner loop */
void maxplus_matvec_f64(
    const double *A, const double *x, double *y,
    int M, int K
);

/* Max-plus dense matmul: C = A ⊗_max B */
void maxplus_matmul_f64(
    const double *A, const double *B, double *C,
    int M, int K, int N
);
```

**`minplus_dense.c`** — Floyd-Warshall / all-pairs kernel:
- Tile the (M, K, N) triple loop to fit the K-strip of B in L2 cache
- Short-circuit: `if A[i,k] == INF, continue` (skip update for missing edges)
- Inner loop `C[i,j] = min(C[i,j], A[i,k] + B[k,j])` is auto-vectorizable with `-O2 -march=native`
- Optional `#pragma omp parallel for` on outer loop

**`maxplus_matvec.c`** — Viterbi inner loop:
- Computes `y[i] = max_k(A[i,k] + x[k])` for each row `i`
- Memory access pattern: row-major traversal of A, sequential read of x
- The bottleneck is the reduction over K — unrolled manually by factor 4 to enable pipelining
- This is the single most performance-critical function in the library

**`python_bindings.c`** — CPython extension:
- Validates buffer protocol (contiguous, dtype float64, correct ndim)
- Calls kernels directly
- Exposes `tropiq._core` with `minplus_matmul`, `maxplus_matvec`, `maxplus_matmul`
- Falls back gracefully if import fails (C extension not compiled)

### 2.3 Python Layer

**`array.py` — `TropicalArray`**

```python
class TropicalArray:
    def __init__(self, data: np.ndarray, semiring: str = 'minplus'):
        # semiring: 'minplus' (+inf as zero) or 'maxplus' (-inf as zero)
        ...
    def __add__(self, other):   # tropical addition: min() or max()
    def __matmul__(self, other): # tropical matmul, dispatches to backend
    def __pow__(self, k):        # repeated squaring via __matmul__
```

Wraps a `float64` NumPy array. `+inf` / `-inf` sentinels are handled transparently.

**`linalg.py`**

- `matmul(A, B, semiring)` — dispatch to C ext or NumPy fallback
- `matpow(W, k)` — repeated squaring, $O(n^3 \log k)$
- `matvec(A, x, semiring)` — for Viterbi step; routes to fast `maxplus_matvec` kernel

**`hmm.py`**

```python
class HMM:
    def __init__(self, n_states: int):
        self.log_trans: np.ndarray  # (S, S) log transition matrix
        self.log_emit: np.ndarray   # (S, V) log emission matrix
        self.log_init: np.ndarray   # (S,) log initial distribution

    def viterbi(self, observations: np.ndarray) -> np.ndarray:
        # Returns most probable state sequence, shape (T,)
        # Inner loop: linalg.matvec(log_trans, delta_t, semiring='maxplus')
        ...

    def fit(self, observations: np.ndarray):
        # Baum-Welch (EM) - v1 stub, full implementation future work
        raise NotImplementedError
```

**`graph.py`**

```python
def allpairs_shortest_path(W: np.ndarray) -> np.ndarray:
    # W: (N, N) cost matrix, np.inf for no edge
    # Returns: (N, N) distance matrix
    # Implementation: tropical matpow via minplus_matmul
```

**`parametric.py`**

```python
class TropicalPolynomial:
    # Represents f(λ) = min_i(a_i + b_i * λ) — piecewise linear convex function
    def __init__(self, coeffs: np.ndarray):
        # coeffs: (K, 2) array of (a_i, b_i) pairs
        ...
    def evaluate(self, lam: float) -> float:
    def breakpoints(self) -> np.ndarray:
        # Returns sorted λ values where the minimizing term changes
        # These are the parametric shortest path breakpoints

def parametric_shortest_path(
    A_costs: np.ndarray,   # (N, N) constant cost terms
    B_costs: np.ndarray,   # (N, N) λ-coefficient terms
) -> list[TropicalPolynomial]:
    # Returns one TropicalPolynomial per (source, dest) pair
    # Each polynomial's breakpoints are where the optimal route changes
```

### 2.4 Data Flow

**Viterbi:**
```
HMM.viterbi(observations)
    │
    ├── initialize delta_0 = log_init + log_emit[:, O_0]
    │
    └── for t in 1..T:
            linalg.matvec(log_trans, delta_{t-1}, semiring='maxplus')
            │
            ▼
            _core.maxplus_matvec_f64  ← C kernel (hot path)
            │
            ▼
            delta_t += log_emit[:, O_t]
    │
    └── traceback → state sequence (T,)
```

**All-pairs shortest path:**
```
allpairs_shortest_path(W)
    │
    ▼
TropicalArray(W, semiring='minplus') ** N
    │
    ▼
linalg.matpow → repeated minplus_matmul
    │
    ▼
_core.minplus_matmul_f64  ← C kernel
    │
    ▼
distance matrix (N, N)
```

**Parametric analysis:**
```
parametric_shortest_path(A_costs, B_costs)
    │
    ▼  (for each source node)
tropical matrix power over parameterized edge weights
    │
    ▼
TropicalPolynomial per (source, dest)
    │
    ▼
.breakpoints() → sorted λ values where optimal route changes
```

### 2.5 Containerization (`docker/`)

The C extension must be compiled for the target platform. Docker removes the "works on my machine" problem.

**`Dockerfile`**:
```dockerfile
FROM python:3.11-slim
RUN apt-get install -y gcc
COPY . /tropiq
WORKDIR /tropiq
RUN pip install -e .          # compiles C extension via setup.py
RUN python -c "import tropiq" # smoke test
```

Users install via:
```bash
docker build -t tropiq .
docker run -v $(pwd):/work tropiq python /work/my_analysis.py
```

The library is importable inside the container exactly as a normal pip install. No Jupyter, no server — plain library, plain Python scripts mounted as volumes.

**Future**: multi-arch builds (`--platform linux/amd64,linux/arm64`) with `-march=native` per target, so Apple Silicon and x86 server both get optimized binaries.

### 2.6 Dependencies

| Layer | Dependency | Required? |
|-------|-----------|-----------|
| Python | NumPy ≥ 1.22 | Yes |
| C build | gcc, Python.h, C99 | Yes for C extension (Docker handles this) |
| Test | pytest, networkx, hmmlearn | Dev only |
| Examples | pandas | Optional |

### 2.7 Performance Targets

**Viterbi** ($T = 500$ time steps):

| State space S | hmmlearn | TropiQ (C kernel) |
|--------------|----------|-------------------|
| 100 | 5ms | 1ms |
| 1,000 | 500ms | 20ms |
| 5,000 | ~60s | ~500ms |

**All-pairs shortest path** (dense graph, single-threaded):

| N nodes | scipy (dense) | TropiQ |
|---------|--------------|--------|
| 500 | 800ms | 80ms |
| 1,000 | 6s | 500ms |

These are targets, not measurements. `benchmarks/` will validate.

---

## 3. Build Order

Suggested implementation sequence, each step testable in isolation:

1. `csrc/maxplus_matvec.c` + `python_bindings.c` — smallest kernel, highest payoff (Viterbi)
2. `array.py` + `linalg.matvec` — Python wrapper, verify against scalar loop
3. `hmm.py` — Viterbi using the kernel; test against hmmlearn on small examples
4. `csrc/minplus_dense.c` — all-pairs kernel
5. `graph.allpairs_shortest_path` — test against networkx
6. `parametric.py` — no new C code; pure Python over existing linalg primitives
7. `docker/Dockerfile` — containerize the whole thing
8. `benchmarks/` — measure against targets above

---

## 4. Future Work

- GPU path: CuPy custom kernel for tropical matmul (same kernel structure, different memory model)
- HMM training: Baum-Welch (forward-backward) as tropical + standard matmul combination
- Multi-arch Docker: optimized builds for x86 and ARM via `--platform` flag
