# Spark Query Plans — Easy Notes + Hands-On

**How to use this notebook**

1. Open this file on **Databricks Free Edition** (serverless compute is on by default).
2. Click **Run All** — every code cell below works without changes.
3. Each code cell calls `.explain(True)` so you can *see* the plan Spark builds.
4. Read the notes above each code cell first, then look at the output.

**What this file covers**
- What a query plan is and why it matters
- How Spark builds a plan (5 steps)
- Catalyst optimizer (projection pushdown, filter pushdown)
- Narrow transformations (filter / select / withColumn)
- Repartition (shuffle → round-robin)
- Coalesce (minimize shuffle)
- Joins (sort-merge join, hash partitioning)
- Group by: count, sum, count distinct
- Why Spark sometimes adds a *redundant* filter
- When predicate pushdown fails (map columns, CAST expressions)

**Free Edition rules I followed throughout**
- ❌ No `df.rdd` (blocked on serverless)
- ✅ Only Python + PySpark DataFrame API
- ✅ Unity Catalog tables (`samples.tpch.*`)
- ✅ Hints instead of `spark.conf` for join strategy
- ✅ Small datasets (Free Edition has limited compute)

---

## 1. What is a query plan and why should you care?

When you write:

```python
df.filter("city = 'Boston'").select("name")
```

…Spark does **not** run it directly. First it builds a **plan** — a step-by-step recipe for how to actually read the data, filter it, transform it, and return the result.

**Why care?**
- It tells you if Spark is doing **extra work** you didn't ask for.
- It tells you if **data is being shuffled** (expensive, slow).
- It's the *first thing* you should look at before optimizing any Spark job.

Think of it like a recipe: same ingredients, but one recipe says "chop everything then cook" and the other says "cook then chop". Same dish, very different time and effort. The query plan is Spark's recipe.

---

## 2. How Spark builds a plan — 5 steps

Spark takes your code through **five stages** before anything runs on the cluster:

| Step | Name | What happens |
|---|---|---|
| 1 | **Unresolved Logical Plan** | Spark reads your code and checks the syntax. It does **not** yet know if tables/columns exist. |
| 2 | **Analyzed Logical Plan** | Spark checks the **catalog** (metadata): "does `orders` exist? does it have a `custkey` column? is it integer?" If yes, the plan becomes *analyzed*. |
| 3 | **Optimized Logical Plan** | The **Catalyst Optimizer** rewrites the plan to be faster (e.g., projection pushdown, filter pushdown). |
| 4 | **Physical Plans** | Spark generates several *physical* plans (different join methods, different scan types). A cost model picks the cheapest one. |
| 5 | **Execution** | The chosen physical plan is what actually runs on the cluster. |

To see all of them in one shot, use: **`df.explain(True)`**

**Two Catalyst optimizations you need to know:**
- **Projection pushdown** — you wrote `select *`, but Spark notices you only use 3 columns, so it reads only those 3. Less I/O.
- **Filter pushdown** — you filtered *after* reading, but Spark notices the filter can be applied *while* reading, so fewer rows enter the pipeline.

---

## 3. Setup — load sample data

We use two tables that already live in the `samples` catalog on Free Edition:
- **`samples.tpch.orders`** — behaves like the *transaction* dataset in the video.
- **`samples.tpch.customer`** — behaves like the *customer* dataset.

Both share a common column: **`custkey`** (the customer id). We'll join them later.

```python
from pyspark.sql import functions as F

orders   = spark.read.table("samples.tpch.orders")
customer = spark.read.table("samples.tpch.customer")

print("orders columns :", orders.columns)
print("customer columns:", customer.columns)
print("orders row count:", orders.count())
print("customer row count:", customer.count())
```

---

## 4. Narrow transformations (filter, select, withColumn)

**What is a narrow transformation?** An operation where each output partition depends on **only one** input partition. **No shuffle.** Fast.

**Examples:** `filter`, `select`, `withColumn`, `drop`, `cast`.

**What to look for in the plan:**
- `FileScan` with a `PushedFilters` list — Spark pushed your filter to the source.
- A separate `Filter` node — Spark also re-applies the filter after the scan (fail-safe; explained in section 12).
- A `Project` node — your `select`/`withColumn` expressions.

```python
narrow_df = (
    orders
    .filter(F.col("orderstatus") == "O")                        # filter
    .withColumn("price_per_item", F.col("totalprice") / 10)     # new column
    .withColumn("price_per_item", F.round("price_per_item", 2)) # update column
    .select("orderkey", "custkey", "totalprice", "price_per_item")  # select
)

narrow_df.explain(True)
```

### What you will see (and why)

- **`FileScan parquet`** — Spark reads only the columns you used (projection pushdown).
- **`PushedFilters: [IsNotNull(orderstatus), EqualTo(orderstatus,O)]`** — Spark pushed your filter into the Parquet reader.
- **`Filter (orderstatus = O)`** — Spark *also* keeps a filter step after the scan. Looks redundant, but it's intentional (section 12).
- **`Project`** — the `price_per_item` calculation sits here.

👉 **No `Exchange` node = no shuffle = fast.**

---

## 5. Repartition (wide transformation → shuffle)

**What it does:** changes the number of partitions. Data is redistributed **across the cluster**, which requires a **shuffle**.

**Partitioning scheme:** **Round-robin** — Spark sends row 0 → partition 0, row 1 → partition 1, … row N → partition (N mod total). Like dealing cards one-by-one to players.

**What to look for:** an `Exchange RoundRobinPartitioning(N)` node.

⚠️ **Free Edition note:** do **not** try `df.rdd.getNumPartitions()` — RDDs are blocked on serverless. Read the plan instead.

```python
repartitioned_df = orders.repartition(24)
repartitioned_df.explain(True)
```

### What you will see

- `Exchange RoundRobinPartitioning(24)` → **shuffle happened**.
- `AdaptiveSparkPlan isFinalPlan=false` → runtime stats can still tweak the plan (AQE).

**Rule of thumb:** if you see `Exchange` in the plan, data moved. If you don't, it didn't.

---

## 6. Coalesce (reduce partitions, avoid shuffle)

**What it does:** reduces the number of partitions — but **tries not to shuffle**.

**How?** It merges partitions that are already on the **same executor** into one bigger partition. Data on executor 1 stays on executor 1; data on executor 2 stays on executor 2. Only when it *has* to move data across executors does a shuffle happen.

**Why prefer coalesce over repartition for shrinking?** Repartition always shuffles. Coalesce usually doesn't.

**What to look for:** a `Coalesce(N)` node. If there's **no `Exchange` next to it**, no shuffle happened.

```python
coalesced_df = orders.coalesce(1)
coalesced_df.explain(True)
```

### What you will see

- `Coalesce 1` — and usually **no `Exchange`**, meaning Spark merged partitions within executors.
- If you tried `coalesce(2)` from a very large partition count, you'd start seeing `Exchange` — an aggressive reduction forces a shuffle.

---

## 7. Joins — sort-merge join and hash partitioning

**What happens during a join?** Both sides get shuffled so that **rows with the same join key land in the same partition**. That way Spark can compare matching keys side-by-side.

**Hash partitioning formula:**

```
target_partition = hash(join_key) mod number_of_shuffle_partitions
```

Example with 4 shuffle partitions:

| Key | hash(key) | mod 4 | Goes to partition |
|---|---|---|---|
| 1 | 1 | 1 | 1 |
| 2 | 2 | 2 | 2 |
| 3 | 3 | 3 | 3 |
| 4 | 4 | 0 | 0 |

Same key on **both** sides → **same partition** → join can happen locally.

**Sort-merge join:** after the shuffle, each partition is **sorted** by key. Then merging two sorted lists is cheap — just walk both pointers.

⚠️ **Free Edition note:** use **hints** to force sort-merge. `spark.conf.set(...)` is unreliable on serverless.

```python
joined_df = (
    orders.hint("merge")
          .join(customer.hint("merge"), on="custkey", how="inner")
)
joined_df.explain(True)
```

### What you will see

Reading the plan **bottom to top** (that's the Spark convention):

1. `FileScan` on `customer` — with `IsNotNull(custkey)` filter (Spark's own optimization).
2. `Exchange hashpartitioning(custkey, 200)` — shuffle the customer side.
3. `Sort` — sort by `custkey` within each partition.
4. Repeat steps 1–3 for `orders`.
5. `SortMergeJoin` — walk both sorted sides.
6. `Project` — keep only the columns you asked for.

If you see `BroadcastHashJoin` instead, one side was small enough that Spark just sent it to every executor. That's also valid — often *faster* for small tables. The `hint("merge")` just makes the sort-merge example visible for learning.

---

## 8. Group by — count

**Naive approach (bad):** shuffle all raw rows, then count.

**What Spark actually does (good): partial aggregation first.**

1. **Partial count** on each partition — count within the partition only.
2. **Shuffle** — bring same keys together.
3. **Final count** — add up the partial counts.

This is called **divide and conquer**: you only move small partial results across the network, not every row.

**What to look for:** two `HashAggregate` nodes (`partial_count` and `count`) separated by an `Exchange`.

```python
count_df = orders.groupBy("orderstatus").count()
count_df.explain(True)
display(count_df)
```

### What you will see

- `HashAggregate keys=[orderstatus], functions=[partial_count(1)]` — local count per partition.
- `Exchange hashpartitioning(orderstatus, 200)` — shuffle on group key.
- `HashAggregate keys=[orderstatus], functions=[count(1)]` — final total.

The word **partial** is your visual cue that Spark did the divide-and-conquer optimization.

---

## 9. Group by — sum

Same idea as count, just with `sum` instead of `count`. Look for `partial_sum` in the first `HashAggregate` and `sum` in the final one.

```python
sum_df = orders.groupBy("orderstatus").agg(F.sum("totalprice").alias("total_price"))
sum_df.explain(True)
display(sum_df)
```

### What you will see

- First `HashAggregate` → `partial_sum(totalprice)` (local).
- `Exchange` on `orderstatus`.
- Second `HashAggregate` → `sum(totalprice)` (final).

---

## 10. Group by — count distinct (the tricky one)

`countDistinct` needs to make sure each distinct value is counted **once globally** — not once per partition. That's why Spark does **extra work** here.

**The 4-step dance:**

1. **Local distinct** — collapse duplicates *within* each partition.
2. **Shuffle #1** on `(group_key, distinct_col)` — bring all copies of a value to the same partition.
3. **Global distinct** — collapse duplicates again across the cluster.
4. **Partial count + shuffle #2 + final count** — count distinct values per group.

**Result:** **four `HashAggregate` nodes** and **two `Exchange` nodes**. That's why `countDistinct` is expensive — use it sparingly.

```python
distinct_df = orders.groupBy("custkey").agg(
    F.countDistinct("orderstatus").alias("distinct_statuses")
)
distinct_df.explain(True)
display(distinct_df.limit(10))
```

### What you will see

Count the `HashAggregate` nodes — there will be **four**. Count the `Exchange` nodes — there will be **two**.

The keys of the first `HashAggregate` will be `[custkey, orderstatus]` (not just `custkey`) — that's Spark doing the "local distinct" step before the first shuffle.

---

## 11. Interesting observation — the "redundant" filter

If you look back at the narrow-transformation plan (section 4), you'll notice:

- The **FileScan** already pushed `orderstatus = 'O'` to the source.
- Spark **still** adds a separate `Filter` node afterwards that does the *same thing*.

**Why?**
- **Pushdown happens in the logical plan** (Catalyst).
- **The post-scan filter is added in the physical plan** — because *not every filter can be pushed*. Filters on `map` columns, on `CAST` expressions, on non-supported data sources, etc. simply can't go to the source. So Spark always adds a **fail-safe filter** to guarantee correctness.

**Is it wasteful?** Not really. The pushed filter already cut down the rows at the source. The post-scan filter only runs on a **much smaller dataset**. It's a tiny cost for a big correctness guarantee.

```python
redundant_df = orders.filter(F.col("orderstatus") == "O")
redundant_df.explain(True)
```

### What to look for

In the output, find:
- `PushedFilters: [..., EqualTo(orderstatus,O)]` (inside `FileScan`)
- **And** a `Filter (isnotnull(orderstatus) AND (orderstatus = O))` node below the scan.

Both are present. That's not a bug — it's the design.

---

## 12. When does predicate pushdown **not** work?

Two classic failures:

### 12.1 Filter on a `map` column key

If you have a `map<string, string>` column (like a `properties` column) and you filter on `properties["color"] == "red"`, the Parquet reader **cannot** evaluate that — map lookups are done by Spark, not by the storage layer. So no pushdown.

### 12.2 Filter on a `CAST` expression

If a column is stored as `string` but you cast it to `int` and filter on the cast (`cast(age as int) > 50`), the source has no way to apply that predicate — it doesn't know how to cast. So only `IsNotNull` gets pushed; the cast-and-compare happens after the scan.

```python
cast_filter_df = orders.filter(F.col("orderkey").cast("string").startswith("1"))
cast_filter_df.explain(True)
```

### What to look for

- `PushedFilters: [IsNotNull(orderkey)]` — only the null-check was pushed.
- The actual cast + `startswith` logic appears **after** the scan in a `Filter` node.

**Lesson:** predicate pushdown is **data-source dependent**. Some databases (e.g., Snowflake, BigQuery) push down much more than Parquet does. Always check the plan.

---

## 13. Cheat sheet — plan patterns at a glance

| Operation | Key nodes in plan | Shuffle? |
|---|---|---|
| **filter / select / withColumn** | `FileScan` (with `PushedFilters`), `Filter`, `Project` | ❌ |
| **repartition(n)** | `Exchange RoundRobinPartitioning(n)` | ✅ |
| **coalesce(n)** | `Coalesce(n)` (usually no `Exchange`) | ⚠️ only if aggressive |
| **sort-merge join** | 2× `Exchange hashpartitioning`, 2× `Sort`, `SortMergeJoin` | ✅ both sides |
| **broadcast join** | `BroadcastHashJoin` (no `Exchange` on small side) | ✅ one side |
| **groupBy().count()** | `HashAggregate(partial_count)` → `Exchange` → `HashAggregate(count)` | ✅ |
| **groupBy().sum()** | `HashAggregate(partial_sum)` → `Exchange` → `HashAggregate(sum)` | ✅ |
| **groupBy().agg(countDistinct)** | **4×** `HashAggregate`, **2×** `Exchange` | ✅ twice |
| **filter on CAST** | `PushedFilters` incomplete; cast-filter appears post-scan | ❌ |

---

## 14. Free Edition gotchas — keep these in mind

| Don't do this | Why | Do this instead |
|---|---|---|
| `df.rdd.getNumPartitions()` | RDD API blocked on serverless | Read `Exchange N` from `df.explain(True)` |
| `df.rdd.map(...)` | RDD API blocked | `df.withColumn(...)` or `mapInPandas` |
| `spark.conf.set("spark.sql.autoBroadcastJoinThreshold","-1")` | Many confs are ignored on serverless | `df.hint("merge")` / `df.hint("broadcast")` |
| Write to `dbfs:/...` | Not used in Free Edition | `df.write.saveAsTable("workspace.default.my_table")` |
| `pip install <rare-package>` at runtime | Not whitelisted | Use pre-installed libs (pyspark, pandas, scikit-learn) |
| Scala or R notebooks | Not supported | Python or SQL only |
| Huge datasets | 2X-Small SQL warehouse / limited daily quota | Use `samples.*` tables or filtered subsets |

---

## How to import this file into Databricks

1. Save this file locally as `Spark_Query_Plans_Free_Edition.md`.
2. In Databricks: **Workspace** → open any folder → **⋮** menu → **Import**.
3. Choose **File** → pick the `.md` file → Import.
4. Databricks automatically detects the Markdown and converts it into a **notebook** with alternating Markdown and code cells.
5. Click **Run All** at the top.

**Done!** You now have the full notes + runnable code in **one `.md` file**, matching what the transcript taught, and tuned for Free Edition.

---

### ✅ End of file — you've covered every topic from the transcript