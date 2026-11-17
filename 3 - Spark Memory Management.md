# Spark Memory Management — Notes + Hands-On (Databricks Free Edition)

**How to use this file**

1. Read each section's notes.
2. Open a **Python serverless notebook** in Databricks: Workspace → Create → Notebook → Python → Serverless.
3. Copy each code block below into its own cell, run it, then open the **Spark UI** (top-right of your notebook) alongside the notes.
4. Sections with runnable code: §4, §5, §6, §9. The rest is conceptual (see §13 for why).

**Free Edition caveat**

Serverless mode does not let you set `spark.executor.memory`, `spark.memory.fraction`, `spark.memory.storageFraction`, or off-heap settings. You can **read** these values and **observe** caching behavior in the Spark UI, but you cannot **change** them. The layout and math below describe classic clusters — they are still essential for interviews and for paid-cluster administration.

---

## 1. Why memory management matters

- Diagnose OOM errors and GC issues.
- Decide when to cache (caching uses storage memory, which competes with execution memory).
- Choose partition sizes.
- Know when off-heap helps.

---

## 2. Executor memory layout — 3 big blocks

An executor's memory splits into three blocks:

```
┌──────────────────────────────────────────────────────┐
│                 Spark Executor Container             │
│                                                       │
│  ┌────────────────────────────────────────────────┐  │
│  │             ON-HEAP MEMORY (JVM)               │  │  ← largest block
│  │  ┌──────────────┬──────────────┐               │  │
│  │  │  Execution   │   Storage    │ ← Unified     │  │
│  │  └──────────────┴──────────────┘               │  │
│  │  ┌──────────────────┬──────────────────┐       │  │
│  │  │   User Memory    │ Reserved (300MB) │       │  │
│  │  └──────────────────┴──────────────────┘       │  │
│  └────────────────────────────────────────────────┘  │
│                                                       │
│  ┌────────────────────────────────────────────────┐  │
│  │       OFF-HEAP MEMORY (OS-managed, optional)   │  │
│  └────────────────────────────────────────────────┘  │
│                                                       │
│  ┌────────────────────────────────────────────────┐  │
│  │       OVERHEAD MEMORY (system/internal)        │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

- **On-heap** — managed by the JVM; where most Spark work happens; the only region controlled by `spark.executor.memory`.
- **Off-heap** — optional; disabled by default; managed by the OS.
- **Overhead** — internal system memory; never used for Spark data.

Spark is written in Scala (which runs on the JVM). PySpark is a Python wrapper around the Java API — the actual execution happens on the JVM, so the memory model is the JVM memory model.

---

## 3. On-heap memory — 4 sub-regions

| Region | Purpose | Controlled by |
|---|---|---|
| **Execution** | Joins, shuffles, sorts, aggregations | `spark.memory.fraction` × 0.5 (dynamic) |
| **Storage** | Cache, broadcast variables | `spark.memory.fraction` × 0.5 (dynamic) |
| **User** | Your objects and UDF closures | remainder |
| **Reserved** | Internal Spark objects | fixed 300 MB |

Execution + Storage = **Unified Memory** — shares a movable boundary.

| Parameter | Default | Meaning |
|---|---|---|
| `spark.memory.fraction` | 0.6 | Fraction of on-heap for Unified Memory |
| `spark.memory.storageFraction` | 0.5 | Fraction of Unified initially for storage |

---

## 4. Worked calculation — 10 GB executor with defaults

**Step 1 — Unified Memory**
```
Unified = 0.6 × 10 GB = 6 GB
```

**Step 2 — Storage**
```
Storage = 0.5 × 6 GB = 3 GB
```

**Step 3 — Execution**
```
Execution = 6 GB − 3 GB = 3 GB
```

**Step 4 — User + Reserved**
```
Remaining  = 10 GB − 6 GB = 4 GB
Reserved   = 300 MB = 0.3 GB
User       = 4 GB − 0.3 GB = 3.7 GB
```

**Result:** Execution 3 GB, Storage 3 GB, User 3.7 GB, Reserved 0.3 GB.

**Code — reproduce the calculation:**

```python
defaults = {
    "spark.memory.fraction":        spark.conf.get("spark.memory.fraction",        "0.6"),
    "spark.memory.storageFraction": spark.conf.get("spark.memory.storageFraction", "0.5"),
    "spark.memory.offHeap.enabled": spark.conf.get("spark.memory.offHeap.enabled", "false"),
    "spark.memory.offHeap.size":    spark.conf.get("spark.memory.offHeap.size",    "0"),
}
for k, v in defaults.items():
    print(f"{k:40s} = {v}")

EXECUTOR_GB  = 10
FRACTION     = float(defaults["spark.memory.fraction"])
STORAGE_FRAC = float(defaults["spark.memory.storageFraction"])

unified   = EXECUTOR_GB * FRACTION
storage   = unified * STORAGE_FRAC
execution = unified - storage
remaining = EXECUTOR_GB - unified
reserved  = 0.3
user      = remaining - reserved

print(f"\nWith spark.executor.memory = {EXECUTOR_GB} GB:")
print(f"  Unified total : {unified:.2f} GB")
print(f"    Execution   : {execution:.2f} GB")
print(f"    Storage     : {storage:.2f} GB")
print(f"  Remaining     : {remaining:.2f} GB")
print(f"    User        : {user:.2f} GB")
print(f"    Reserved    : {reserved:.2f} GB")
```

---

## 5. Overhead memory and what the cluster manager is asked for

**Formula:**
```
overhead  = max(384 MB, 0.10 × spark.executor.memory)
requested = spark.executor.memory + overhead + spark.memory.offHeap.size
```

With `executor.memory = 10 GB`, off-heap disabled:
```
requested = 10 GB + 1 GB + 0 = 11 GB per executor
```

`spark.executor.memory` does **not** include overhead or off-heap. If you forget the overhead when sizing containers, YARN/K8s can kill them before Spark itself runs out of memory.

**Code:**

```python
EXECUTOR_GB = 10
overhead_gb = max(0.384, 0.10 * EXECUTOR_GB)
offheap_gb  = 0

requested = EXECUTOR_GB + overhead_gb + offheap_gb

print(f"spark.executor.memory      = {EXECUTOR_GB:.2f} GB")
print(f"overhead                   = {overhead_gb:.2f} GB")
print(f"off-heap (disabled)        = {offheap_gb:.2f} GB")
print(f"TOTAL REQUESTED            = {requested:.2f} GB")
```

---

## 6. Off-heap memory

**Why it exists:** When on-heap fills up, the JVM runs **Garbage Collection** which pauses your program. Off-heap memory lives outside the JVM heap — GC does not touch it.

**Trade-offs:**

| Pros | Cons |
|---|---|
| No GC pauses | You allocate and free memory manually |
| Faster than disk spill | Slower than on-heap |
| Useful for huge caches | Easy to leak memory |
| Good for large join state | Adds complexity |

**Decision rule:** when on-heap is full, choose off-heap over disk spill — disk is orders of magnitude slower.

**Enable it (paid cluster only):**
```
spark.memory.offHeap.enabled true
spark.memory.offHeap.size    1g
```
Good starting point: 10–20% of `executor.memory`.

**Code — read current values (serverless shows defaults):**

```python
for k in ("spark.memory.offHeap.enabled", "spark.memory.offHeap.size"):
    print(f"{k:35s} = {spark.conf.get(k, '<not set>')}")
```

---

## 7. Unified Memory — the movable slider

**Execution + Storage = Unified Memory.** They share one pool with a boundary that moves at runtime.

```
  ┌────────────────────────────────────────────────────┐
  │  Execution          │   Storage                    │
  │  ← can grow right   │   can grow left →            │
  └─────────────────────┴──────────────────────────────┘
                          ↑ movable boundary
```

**Three slider cases:**

**Case A — Execution needs more, storage is idle.** Execution expands into unused storage. No eviction.

**Case B — Execution needs more, storage has cached blocks.** Storage evicts LRU blocks. Execution wins.

**Case C — Storage needs more, execution is using its space.** Storage cannot evict execution. It spills to disk or drops the new cache.

**Rule (memorize):** Execution can evict storage. Storage can never evict execution.

**LRU eviction in plain English:** storage memory holds cached DataFrames. When space is needed, drop the least-recently-used block. If still not enough, drop the next. That evicted block, if needed again, must be recomputed from source.

---

## 8. Before vs after Spark 1.6

**Before 1.6 — fixed split:**
```
┌──────────────────────┬──────────────────────┐
│  Execution 3 GB      │  Storage   3 GB      │  ← boundary is frozen
└──────────────────────┴──────────────────────┘
```
If execution was full and storage was empty, execution could not use the empty storage. Memory wasted.

**From 1.6 onward — dynamic split:**
```
┌──────────────────────┬──────────────────────┐
│  Execution 4.2 GB ←  │  → Storage  1.8 GB   │  ← boundary moves at runtime
└──────────────────────┴──────────────────────┘
```
The boundary moves at runtime. This was one of the biggest performance wins in Spark 1.6 — no config change, just smarter allocation.

---

## 9. LRU eviction — hands-on with caching

**What to watch for:** when you `.cache()` a DataFrame, Spark lazily stores partitions in storage memory. Open **Spark UI → Storage** tab to see exactly how much storage was used, whether blocks spilled to disk, and whether any partitions were evicted.

**Code — cache and observe:**

```python
from pyspark.sql import functions as F

orders = spark.read.table("samples.tpch.orders")

orders_cached = orders.select("orderkey", "custkey", "totalprice", "orderstatus").cache()
orders_cached.count()            # materializes the cache

orders_cached.explain(True)      # look for InMemoryTableScan / InMemoryRelation

orders_cached.groupBy("orderstatus").count().show()
```

**In Spark UI → Storage tab:**
- The cached DataFrame appears with **Size in Memory**, **Size on Disk**, and **Fraction Cached**.
- **In the SQL tab** the second query's plan should show `InMemoryTableScan` (not `FileScan`).

**Code — two caches competing:**

```python
cached_a = orders.select("orderkey", "totalprice", "orderstatus").cache()
cached_a.count()

customer = spark.read.table("samples.tpch.customer")
cached_b = customer.cache()
cached_b.count()

# Both caches live — Spark UI Storage tab shows both.
# If combined size > storage memory, one shows evicted partitions.

cached_a.unpersist()
cached_b.unpersist()
orders_cached.unpersist()
```

**Always `unpersist()` after experimenting** — a forgotten cache eats storage memory for the rest of the session.

---

## 10. Why execution has priority over storage

- **Execution runs the mandatory work** — joins, shuffles, sorts, aggregations. Without memory these spill to disk and become catastrophically slow.
- **Cached blocks can be recomputed** from source. Shuffle state cannot — mid-job it is irreplaceable.
- **Caching is an optimization, shuffle memory is mandatory.**

**Cache only when:**
- The DataFrame is reused multiple times downstream.
- It fits in storage memory.
- It's the result of an expensive computation.

**Do not cache:**
- DataFrames used once.
- Small, fast-to-read tables.
- Anything the plan doesn't reuse.

---

## 11. Garbage collection

GC fires when on-heap memory fills. The JVM pauses your program (stop-the-world), frees unreferenced objects, then resumes. Pauses range from milliseconds to seconds, and during a pause no work happens on that worker.

**When off-heap helps:**
- Frequent long GC pauses (see Spark UI → Executors → **Task Time (GC Time)** column).
- Large in-memory caches or joins.
- Executor memory of 10 GB or more.

**Trade-off:** off-heap avoids GC but you manage allocation and deallocation yourself.

---

## 12. Cheat sheet

**On-heap layout (defaults, 10 GB executor):**

| Region | Size | Purpose |
|---|---|---|
| Execution | 3 GB | joins, shuffles, sorts, aggregations |
| Storage | 3 GB | cache, broadcast variables |
| User | 3.7 GB | your objects, UDF closures |
| Reserved | 0.3 GB | internal Spark objects |

**Formulas:**

```
Unified     = executor_memory × spark.memory.fraction          (0.6)
Storage     = Unified × spark.memory.storageFraction           (0.5)
Execution   = Unified − Storage
User        = (executor_memory − Unified) − 0.3 GB
Overhead    = max(384 MB, 0.10 × executor_memory)
Requested   = executor_memory + overhead + offHeap.size
```

**Slider rules:**

| Scenario | Result |
|---|---|
| Execution needs more, storage idle | Execution expands into storage — no eviction |
| Execution needs more, storage has blocks | Storage evicts LRU; execution wins |
| Storage needs more, execution uses its space | Storage spills or drops cache; never evicts execution |
| Storage needs more, execution has spare | Storage grows into unused execution space |

**Cache do's and don'ts:**

| Cache | Don't cache |
|---|---|
| Reused 2+ times downstream | Used once |
| Expensive computation result | Small, fast-read table |
| Fits in storage memory | Larger than storage memory |
| Confirmed reused by the plan | No reuse in the plan |

**Config params (readable on serverless, settable on classic):**

| Config | Default |
|---|---|
| `spark.executor.memory` | cluster-dependent (on-heap only) |
| `spark.memory.fraction` | 0.6 |
| `spark.memory.storageFraction` | 0.5 |
| `spark.memory.offHeap.enabled` | false |
| `spark.memory.offHeap.size` | 0 |
| Overhead | max(384MB, 10% × executor.memory) |

---

## 13. Free Edition adaptations

| What the transcript does | Free Edition (serverless) |
|---|---|
| Set `spark.executor.memory` | Not configurable |
| Set `spark.memory.fraction` / `storageFraction` | Not configurable |
| Set `spark.memory.offHeap.enabled` / `.size` | Not configurable |
| Read any config | Works via `spark.conf.get()` |
| Cache a DataFrame, see it in Spark UI Storage tab | Works |
| Observe eviction with two competing caches | Works (use small tables) |
| Calculate layout numbers manually | Works — pure math (§4 code) |

You can **verify the concept** on serverless — cache, watch storage-memory use, unpersist. You cannot **change** the memory layout yourself because the platform manages it. The layout and calculation part is conceptual knowledge for interviews, exams, and paid-cluster administration.

---

## End of notes — key takeaways

1. Three blocks: on-heap (JVM), off-heap (optional, OS), overhead (system).
2. On-heap splits into execution, storage, user, reserved.
3. Execution + Storage = Unified Memory with a dynamic boundary.
4. Execution has priority — storage evicts LRU to make room.
5. `Overhead = max(384 MB, 10% × executor.memory)`. Cluster manager gets `executor.memory + overhead + offHeap.size`.
6. Off-heap escapes GC pauses — you manage alloc and dealloc. Start at 10–20% of executor memory.
7. Only cache what you reuse. Caching competes with execution memory.
8. Always `unpersist()` after experimenting.