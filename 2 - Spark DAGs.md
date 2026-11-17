# Spark DAG — Notes + Hands-On (Databricks Free Edition)

**How to use this notebook**

1. Click **Run All** — every cell below runs on serverless without changes.
2. After each code cell runs, open the **Spark UI** link (top-right of the notebook).
3. Go to the **Jobs** tab → expand a job → look at **DAG visualization**, **Stages**, and **Tasks**.
4. Compare what you see with the notes in the markdown cells above each code block.

**What this notebook covers**

| # | Topic |
|---|---|
| 1 | What is a DAG? Jobs vs stages vs tasks |
| 2 | DAG for reading a file (the "metadata job" surprise) |
| 3 | DAG for narrow transformations (filter / add cols / select) |
| 4 | DAG for sort-merge join (3 jobs for 1 action) |
| 5 | DAG for broadcast join |
| 6 | DAG for group by — count |
| 7 | DAG for group by — sum |
| 8 | DAG for group by — count distinct (4 HashAggregates, 2 shuffles) |
| 9 | AQE — Adaptive Query Execution in action |
| 10 | Stages = shuffle boundaries |
| 11 | Cheat sheet |

**Free Edition rules used**

| ❌ Don't | ✅ Do |
|---|---|
| `df.rdd.getNumPartitions()` | Read `df.explain(True)` or Spark UI task count |
| `spark.conf.set(..., "autoBroadcastJoinThreshold", ...)` | `df.hint("merge")` / `df.hint("broadcast")` |
| DBFS paths | `samples.*` or `workspace.default.*` |
| RDD transforms | DataFrame API |
| Scala / R | Python only |

## 1. What is a DAG? Jobs, Stages, Tasks

A **DAG** (Directed Acyclic Graph) is the graph of steps Spark executes to compute your DataFrame. When you trigger an **action** (`show`, `count`, `write`, `collect`, `display`), Spark takes the DAG, breaks it into **stages**, and runs each stage as a set of **tasks** — one task per partition.

**Why care?**
- The DAG is *how* the query plan actually executes on the cluster.
- It tells you **where data moves** (shuffles) and where it doesn't (narrow steps).
- **Shuffles = stage boundaries.** If you know where your shuffles are, you know your bottlenecks.
- It's what the **Spark UI** renders for you.

**Transformation vs Action**
- **Transformation** (`filter`, `select`, `withColumn`, `groupBy`, `join`) — **lazy**, just builds the DAG.
- **Action** (`show`, `count`, `collect`, `write`, `display`) — **triggers** the DAG to actually run.

**3-level hierarchy**

| Level | What it is | When a new one starts |
|---|---|---|
| **Job** | One per action | Every `show`, `count`, `write`, `display` |
| **Stage** | Group of tasks between shuffles | **A shuffle ends the current stage and starts a new one** |
| **Task** | Work on one partition on one executor | One task per partition per stage |

**Golden rule you will reuse constantly:**
```
Number of tasks in a stage = Number of partitions being processed
Number of stages = Number of shuffles + 1
```

## 2. Setup — load sample data

We use two Unity Catalog tables that exist in every Free Edition workspace:
- **`samples.tpch.orders`** — behaves like the "transaction" dataset in the video.
- **`samples.tpch.customer`** — behaves like the "customer" dataset.

Both share **`custkey`** — our join key.

⚠️ **Free Edition note:** The transcript uses `df.rdd.getNumPartitions()`. On serverless the RDD API is unreliable — use `df.explain(True)` or count tasks in the Spark UI instead.

```python
from pyspark.sql import functions as F

orders   = spark.read.table("samples.tpch.orders")
customer = spark.read.table("samples.tpch.customer")

print("orders row count  :", orders.count())
print("customer row count:", customer.count())
```

**After running the cell above, open the Spark UI:**
- **2 jobs** appear (one per `.count()`).
- Each job has **1 stage**, no shuffle.
- Task count in each stage = number of partitions of that table.

Use the **Stages** tab task count to infer partition count on Free Edition.

## 3. DAG for reading a file — the "metadata job" surprise

You call `orders.show(5)`. You expect **one job**. Spark creates **two**:

1. **"Read parquet metadata" job** — reads the Parquet footer to learn schema, file sizes, row counts, partitioning.
2. **`show()` job** — does the actual scan and returns the rows.

**Why?** Spark needs schema + sizes before it can plan the work. Metadata lives at the **footer** of every Parquet file — the reader only reads a tiny bit.

**"Batches" ≠ "partitions"** — the Parquet reader pulls rows in **batches of ~4096 rows**. `.show(5)` stops after the first batch or two, so a 10 GB table with `.show(5)` only reads a small chunk. That's a huge efficiency gain.

**Plan nodes you'll see:**
- `BatchScan parquet` (columnar — Parquet reads in batches)
- `ColumnarToRow` (convert columnar to row for downstream)
- `CollectLimit(6)` (5 rows requested + 1 extra for safety)

```python
orders.show(5)
orders.explain(True)
```

**In the Spark UI you'll see:**
- **2 jobs** for this cell (metadata + show).
- DAG for the `show` job: `BatchScan parquet` → `ColumnarToRow` → `CollectLimit(6)`.
- DAG for the metadata job: scans file footers only — no data read.

## 4. DAG for narrow transformations

**Narrow = no shuffle = 1 stage = 1 job** (per action).

We do:
1. Filter `orderstatus = "O"`
2. Add `price_per_item = totalprice / 10`
3. Round it
4. Select columns
5. **Trigger full computation** via `.write.format("noop").save()`

**Why `format("noop")` instead of `.show()`?** `.show(5)` stops after ~5 rows. `noop` (Spark's built-in "no-op" data source) forces Spark to **compute all rows** but writes nothing to disk — exactly what the transcript does.

⚠️ **Free Edition safe:** `format("noop")` needs no filesystem path and works on serverless.

```python
narrow_df = (
    orders
    .filter(F.col("orderstatus") == "O")
    .withColumn("price_per_item", F.col("totalprice") / 10)
    .withColumn("price_per_item", F.round("price_per_item", 2))
    .select("orderkey", "custkey", "totalprice", "price_per_item")
)

narrow_df.write.format("noop").mode("overwrite").save()
narrow_df.explain(True)
```

**What you'll see in the Spark UI:**
- **1 job, 1 stage** — narrow op, no shuffle.
- DAG nodes:
  1. `FileScan parquet` (projection pushdown reads only used columns)
  2. `ColumnarToRow`
  3. `Filter (orderstatus = O)`
  4. `Project` — **all `withColumn` / `round` / `select` collapse into one Project** (that's the "black box" the transcript mentions)
  5. `Write` (noop)

**Key insight:** Spark fuses all `withColumn` calls into a single `Project` node — like SQL `SELECT a, round(b/10) AS c`.

## 5. DAG for sort-merge join (wide transformation)

**A join is wide → shuffles.** The transcript sees **3 jobs for 1 action** because:
1. **Job 1** — read + shuffle the **left** side (`orders`)
2. **Job 2** — read + shuffle the **right** side (`customer`)
3. **Job 3** — actual **sort-merge join** + write

Each side's shuffle runs in its own job because shuffles are lazily scheduled.

**Hash partitioning recap:** shuffle uses `hash(custkey) mod 200` — rows with the same `custkey` land on the same partition. Then both sides are **sorted by `custkey`** and the merge walks both pointers.

⚠️ **Free Edition alternative to `spark.conf.set`:** use `.hint("merge")` on **both** DataFrames to force sort-merge.

```python
joined_df = (
    orders.hint("merge")
          .join(customer.hint("merge"), on="custkey", how="inner")
)

joined_df.write.format("noop").mode("overwrite").save()
joined_df.explain(True)
```

**What you'll see in the Spark UI:**
- **3 jobs** appear.
- Each job has **1 stage**.
- DAG in the join job:
  - Two `Exchange hashpartitioning(custkey, 200)` nodes (one per side)
  - `AQEShuffleRead coalesced(N)` — AQE reduces 200 partitions to ~24 at runtime
  - Two `Sort` nodes
  - `SortMergeJoin`
  - `Project` → `Write`

**Skew handling:** if one partition had vastly more data, AQE splits it — you'd see `AQEShuffleRead` with multiple split ranges.

**`AdaptiveSparkPlan isFinalPlan=false`** in the `explain()` output means: "the plan shown is pre-execution; AQE will pick the real plan using runtime stats." The Spark UI shows the *final* plan with `isFinalPlan=true`.

## 6. DAG for broadcast join

**What it is:** if one side is **small enough**, Spark **broadcasts** it — sends a full copy to every executor. Each executor joins its chunk of the big table **locally** — **no shuffle of the big side**.

**Result:** much faster than sort-merge for small-vs-large joins.

⚠️ **Free Edition alternative:** instead of `spark.conf.set("spark.sql.autoBroadcastJoinThreshold", ...)`, use **`.hint("broadcast")` on the small side**.

In our case, `customer` is much smaller than `orders`, so it's an ideal broadcast candidate.

```python
broadcast_joined_df = (
    orders.join(customer.hint("broadcast"), on="custkey", how="inner")
)

broadcast_joined_df.write.format("noop").mode("overwrite").save()
broadcast_joined_df.explain(True)
```

**What you'll see in the Spark UI:**
- **2 jobs** appear (one for the small side's scan + broadcast, one for the large side join).
- DAG shows `BroadcastExchange` (small side pushed everywhere) → `BroadcastHashJoin`.
- Only **0–1 shuffle** total (small side is replicated, not shuffled).

**In `explain()`:** look for `BroadcastHashJoin` and `BroadcastExchange` nodes.

## 7. DAG for group by — count (wide)

Spark does **partial aggregation** locally, then shuffles, then **final aggregation**.

**Without partial:** all raw rows shuffle → huge network cost.
**With partial (Spark's actual behavior):** each partition counts *locally* → tiny partial counts shuffle → final count. **Divide and conquer.**

**Plan nodes:**
1. `HashAggregate keys=[orderstatus], functions=[partial_count(1)]` — **local**
2. `Exchange hashpartitioning(orderstatus, 200)` — shuffle on group key
3. `AQEShuffleRead coalesced(N)` — AQE collapses empties
4. `HashAggregate keys=[orderstatus], functions=[count(1)]` — **final**

```python
count_df = orders.groupBy("orderstatus").count()

count_df.write.format("noop").mode("overwrite").save()
count_df.explain(True)
display(count_df)
```

**What you'll see in the Spark UI:**
- **2 jobs, 2 stages** (1 shuffle = 2 stages).
- The word **"partial"** in the first `HashAggregate` is your visual cue that divide-and-conquer worked.

If you ever see the first `HashAggregate` **without** "partial", Spark missed the optimization (happens with non-associative ops).

## 8. DAG for group by — sum (wide)

**Identical shape to count**, just `sum` instead of `count`. Look for `partial_sum` → `Exchange` → `sum`.

```python
sum_df = (
    orders.groupBy("orderstatus")
          .agg(F.sum("totalprice").alias("total_price"))
)

sum_df.write.format("noop").mode("overwrite").save()
sum_df.explain(True)
display(sum_df)
```

## 9. DAG for group by — count distinct (the tricky one)

**The problem:** `countDistinct` can't just add partial counts — a customer who visited the **same city twice** still counts as **1**.

**The 4-step dance:**
1. **Local distinct** on `(custkey, orderstatus)` — collapse duplicates in each partition
2. **Shuffle #1** on `(custkey, orderstatus)` — bring pairs together
3. **Global distinct** on `(custkey, orderstatus)` — collapse across cluster
4. **Partial count** → **Shuffle #2** on `custkey` → **final count**

**Result:** **4 `HashAggregate` nodes** and **2 `Exchange` nodes** → **3 stages** (stages = shuffles + 1).

**Tell-tale sign:** an **empty `functions=[]`** inside a `HashAggregate` means Spark is doing a **distinct**, not a count.

```python
distinct_df = (
    orders.groupBy("custkey")
          .agg(F.countDistinct("orderstatus").alias("distinct_status_count"))
)

distinct_df.write.format("noop").mode("overwrite").save()
distinct_df.explain(True)
display(distinct_df.limit(10))
```

**What you'll see in the Spark UI:**
- **3 stages** in the plan (2 shuffles = 3 stages).
- DAG nodes in order:
  1. `HashAggregate keys=[custkey, orderstatus], functions=[]` (**distinct**)
  2. `Exchange hashpartitioning(custkey, orderstatus, 200)` — **Shuffle #1**
  3. `HashAggregate keys=[custkey, orderstatus], functions=[]` (distinct again)
  4. `HashAggregate keys=[custkey], functions=[partial_count(...)]`
  5. `Exchange hashpartitioning(custkey, 200)` — **Shuffle #2**
  6. `HashAggregate keys=[custkey], functions=[count(...)]`
  7. `Write`

**This is why `countDistinct` is expensive** — use it sparingly.

## 10. AQE — Adaptive Query Execution in action

**Introduced in Spark 3.0.** Uses **runtime statistics** to *re-plan* after each shuffle.

**Three AQE optimizations visible in the DAGs above:**

| Optimization | What it does | Where you see it |
|---|---|---|
| **Coalesce shuffle partitions** | 200 → only as many as needed | `AQEShuffleRead coalesced(N)` |
| **Skew join handling** | Split a huge partition into chunks | `AQEShuffleRead` with multiple ranges |
| **Dynamic join switch** | Convert sort-merge → broadcast if small | `BroadcastHashJoin` without a hint |

**`AdaptiveSparkPlan isFinalPlan=false`** in `explain()` means "the plan you see is pre-execution; AQE will choose the actual plan at runtime." After the query runs, Spark UI shows the *final* plan.

```python
# Compare static (pre-execution) plan vs final plan
print("=" * 60)
print("STATIC PLAN (AQE hasn't run yet)")
print("=" * 60)
joined_df.explain(True)

# Now trigger the action
print("\n" + "=" * 60)
print("Triggering execution — check Spark UI for the FINAL plan")
print("=" * 60)
joined_df.write.format("noop").mode("overwrite").save()
print("Done. Open Spark UI → SQL tab → click the query → see the FINAL plan with AQE applied.")
```

## 11. Stages = shuffle boundaries (the key rule)

Memorize:
```
1 shuffle  → 2 stages
2 shuffles → 3 stages
N shuffles → N + 1 stages
```

**Why?** A stage is the maximal set of tasks that can be pipelined **without** a shuffle. A shuffle must finish *completely* (all tasks on all executors) before the next stage starts — that's the boundary.

| Operation | Shuffles | Stages |
|---|---|---|
| Read + `.show()` | 0 | 1 (+1 metadata job) |
| Narrow filter / select + write | 0 | 1 |
| Sort-merge join | 2 (one per side) | **3 jobs**, each 1 stage |
| Broadcast join | 0–1 | **2 jobs**, each 1 stage |
| Group by count / sum | 1 | 2 |
| Group by count distinct | 2 | **3 stages** |

**Rule of thumb:** open Spark UI → Stages tab → count stages → each shuffle in the plan corresponds to one boundary.

## 12. Cheat sheet — plan patterns

| Task | Action | Node to look for | Shuffle? | Stages |
|---|---|---|---|---|
| Filter + select | `df.filter(...).select(...).write.format("noop").save()` | `FileScan` → `ColumnarToRow` → `Filter` → `Project` | ❌ | 1 |
| Repartition | `df.repartition(N)` | `Exchange RoundRobinPartitioning(N)` | ✅ | 2 |
| Coalesce | `df.coalesce(N)` | `Coalesce(N)` (no `Exchange` if safe) | ⚠️ | 1 or 2 |
| Sort-merge join | `a.hint("merge").join(b.hint("merge"), ...)` | 2× `Exchange hashpartitioning` → 2× `Sort` → `SortMergeJoin` | ✅ both | 3 (3 jobs) |
| Broadcast join | `a.join(b.hint("broadcast"), ...)` | `BroadcastExchange` → `BroadcastHashJoin` | ✅ one | 2 (2 jobs) |
| Group by count | `df.groupBy(...).count()` | `HashAggregate(partial_count)` → `Exchange` → `HashAggregate(count)` | ✅ | 2 |
| Group by sum | `df.groupBy(...).agg(F.sum(...))` | `HashAggregate(partial_sum)` → `Exchange` → `HashAggregate(sum)` | ✅ | 2 |
| Group by count distinct | `df.groupBy(...).agg(F.countDistinct(...))` | **4×** `HashAggregate`, **2×** `Exchange` | ✅ twice | 3 |

## 13. Free Edition adaptations — summary

| Transcript used | Free Edition alternative |
|---|---|
| `df.rdd.getNumPartitions()` | `df.explain(True)` — partition count shows in `Exchange hashpartitioning(..., N)` or `BatchScan`; or count tasks in Spark UI Stages tab |
| `spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")` | `df.hint("merge")` on both sides |
| `spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10485760")` | `small_df.hint("broadcast")` on small side |
| `df.write.format("noop").save()` | ✅ Works as-is (built-in NoopDataSource) |
| `df.rdd` in general | Blocked on serverless — use `explain()` + Spark UI |
| Custom `pip install` | Pre-installed libs only |
| Scala / R | Python only |

## ✅ You're done — what you should now recognize instantly

1. **Reading a file** → 2 jobs (metadata + action), 1 stage each.
2. **Narrow op** → 1 job, 1 stage, no `Exchange`.
3. **Sort-merge join** → 3 jobs, `Exchange` on both sides, `SortMergeJoin`.
4. **Broadcast join** → 2 jobs, `BroadcastExchange` + `BroadcastHashJoin`.
5. **Group by count/sum** → 2 stages, `partial_*` in first `HashAggregate`.
6. **Group by countDistinct** → **3 stages**, **4 HashAggregates**, **2 shuffles**, empty `functions=[]` = distinct step.
7. **AQE** always shrinks shuffle partitions at runtime — look for `AQEShuffleRead coalesced(N)`.

**Next habit to build:** After every `write` or `show`, open Spark UI → Stages → sort by **Shuffle Read** — the biggest stage is your bottleneck.

**End of notebook.**

If you want the previous tutorial's `.md` file (for Spark **query plans**), you already have it — it's the one you saved earlier as `Spark_Query_Plans_Free_Edition.md`. This one covers **DAGs / Spark UI / jobs+stages+tasks**, which is the natural next step after query plans.