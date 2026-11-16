# Databricks notebook source
# MAGIC %md
# MAGIC # Spark Query Plans — Easy Notes + Hands-On
# MAGIC
# MAGIC **How to use this notebook**
# MAGIC
# MAGIC 1. Open this notebook on **Databricks Free Edition** (serverless compute is on by default).
# MAGIC 2. Click **Run All** — every cell below works without changes.
# MAGIC 3. Each code cell calls `.explain(True)` so you can *see* the plan Spark builds.
# MAGIC 4. Read the markdown above each code cell first, then look at the output.
# MAGIC
# MAGIC **What this notebook covers**
# MAGIC - What a query plan is and why it matters
# MAGIC - How Spark builds a plan (5 steps)
# MAGIC - Catalyst optimizer (projection pushdown, filter pushdown)
# MAGIC - Narrow transformations (filter / select / withColumn)
# MAGIC - Repartition (shuffle → round-robin)
# MAGIC - Coalesce (minimize shuffle)
# MAGIC - Joins (sort-merge join, hash partitioning)
# MAGIC - Group by: count, sum, count distinct
# MAGIC - Why Spark sometimes adds a *redundant* filter
# MAGIC - When predicate pushdown fails (map columns, CAST expressions)
# MAGIC
# MAGIC **Free Edition rules I followed throughout**
# MAGIC - ❌ No `df.rdd` (blocked on serverless)
# MAGIC - ✅ Only Python + PySpark DataFrame API
# MAGIC - ✅ Unity Catalog tables (`samples.tpch.*`)
# MAGIC - ✅ Hints instead of `spark.conf` for join strategy
# MAGIC - ✅ Small datasets (Free Edition has limited compute)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. What is a query plan and why should you care?
# MAGIC
# MAGIC When you write:
# MAGIC
# MAGIC ```python
# MAGIC df.filter("city = 'Boston'").select("name")
# MAGIC ```
# MAGIC
# MAGIC …Spark does **not** run it directly. First it builds a **plan** — a step-by-step recipe for how to actually read the data, filter it, transform it, and return the result.
# MAGIC
# MAGIC **Why care?**
# MAGIC - It tells you if Spark is doing **extra work** you didn't ask for.
# MAGIC - It tells you if **data is being shuffled** (expensive, slow).
# MAGIC - It's the *first thing* you should look at before optimizing any Spark job.
# MAGIC
# MAGIC Think of it like a recipe: same ingredients, but one recipe says "chop everything then cook" and the other says "cook then chop". Same dish, very different time and effort. The query plan is Spark's recipe.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. How Spark builds a plan — 5 steps
# MAGIC
# MAGIC Spark takes your code through **five stages** before anything runs on the cluster:
# MAGIC
# MAGIC | Step | Name | What happens |
# MAGIC |---|---|---|
# MAGIC | 1 | **Unresolved Logical Plan** | Spark reads your code and checks the syntax. It does **not** yet know if tables/columns exist. |
# MAGIC | 2 | **Analyzed Logical Plan** | Spark checks the **catalog** (metadata): "does `orders` exist? does it have a `custkey` column? is it integer?" If yes, the plan becomes *analyzed*. |
# MAGIC | 3 | **Optimized Logical Plan** | The **Catalyst Optimizer** rewrites the plan to be faster (e.g., projection pushdown, filter pushdown). |
# MAGIC | 4 | **Physical Plans** | Spark generates several *physical* plans (different join methods, different scan types). A cost model picks the cheapest one. |
# MAGIC | 5 | **Execution** | The chosen physical plan is what actually runs on the cluster. |
# MAGIC
# MAGIC To see all of them in one shot, use: **`df.explain(True)`**
# MAGIC
# MAGIC **Two Catalyst optimizations you need to know:**
# MAGIC - **Projection pushdown** — you wrote `select *`, but Spark notices you only use 3 columns, so it reads only those 3. Less I/O.
# MAGIC - **Filter pushdown** — you filtered *after* reading, but Spark notices the filter can be applied *while* reading, so fewer rows enter the pipeline.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Setup — load sample data
# MAGIC
# MAGIC We use two tables that already live in the `samples` catalog on Free Edition:
# MAGIC - **`samples.tpch.orders`** — behaves like the *transaction* dataset in the video.
# MAGIC - **`samples.tpch.customer`** — behaves like the *customer* dataset.
# MAGIC
# MAGIC Both share a common column: **`custkey`** (the customer id). We'll join them later.

# COMMAND ----------

from pyspark.sql import functions as F

orders   = spark.read.table("samples.tpch.orders")
customer = spark.read.table("samples.tpch.customer")

print("orders columns :", orders.columns)
print("customer columns:", customer.columns)
print("orders row count:", orders.count())
print("customer row count:", customer.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Narrow transformations (filter, select, withColumn)
# MAGIC
# MAGIC **What is a narrow transformation?** An operation where each output partition depends on **only one** input partition. **No shuffle.** Fast.
# MAGIC
# MAGIC **Examples:** `filter`, `select`, `withColumn`, `drop`, `cast`.
# MAGIC
# MAGIC **What to look for in the plan:**
# MAGIC - `FileScan` with a `PushedFilters` list — Spark pushed your filter to the source.
# MAGIC - A separate `Filter` node — Spark also re-applies the filter after the scan (fail-safe; explained in section 12).
# MAGIC - A `Project` node — your `select`/`withColumn` expressions.

# COMMAND ----------

narrow_df = (
    orders
    .filter(F.col("orderstatus") == "O")                        # filter
    .withColumn("price_per_item", F.col("totalprice") / 10)     # new column
    .withColumn("price_per_item", F.round("price_per_item", 2)) # update column
    .select("orderkey", "custkey", "totalprice", "price_per_item")  # select
)

narrow_df.explain(True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### What you will see (and why)
# MAGIC
# MAGIC - **`FileScan parquet`** — Spark reads only the columns you used (projection pushdown).
# MAGIC - **`PushedFilters: [IsNotNull(orderstatus), EqualTo(orderstatus,O)]`** — Spark pushed your filter into the Parquet reader.
# MAGIC - **`Filter (orderstatus = O)`** — Spark *also* keeps a filter step after the scan. Looks redundant, but it's intentional (section 12).
# MAGIC - **`Project`** — the `price_per_item` calculation sits here.
# MAGIC
# MAGIC 👉 **No `Exchange` node = no shuffle = fast.**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Repartition (wide transformation → shuffle)
# MAGIC
# MAGIC **What it does:** changes the number of partitions. Data is redistributed **across the cluster**, which requires a **shuffle**.
# MAGIC
# MAGIC **Partitioning scheme:** **Round-robin** — Spark sends row 0 → partition 0, row 1 → partition 1, … row N → partition (N mod total). Like dealing cards one-by-one to players.
# MAGIC
# MAGIC **What to look for:** an `Exchange RoundRobinPartitioning(N)` node.
# MAGIC
# MAGIC ⚠️ **Free Edition note:** do **not** try `df.rdd.getNumPartitions()` — RDDs are blocked on serverless. Read the plan instead.

# COMMAND ----------

repartitioned_df = orders.repartition(24)
repartitioned_df.explain(True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### What you will see
# MAGIC
# MAGIC - `Exchange RoundRobinPartitioning(24)` → **shuffle happened**.
# MAGIC - `AdaptiveSparkPlan isFinalPlan=false` → runtime stats can still tweak the plan (AQE).
# MAGIC
# MAGIC **Rule of thumb:** if you see `Exchange` in the plan, data moved. If you don't, it didn't.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Coalesce (reduce partitions, avoid shuffle)
# MAGIC
# MAGIC **What it does:** reduces the number of partitions — but **tries not to shuffle**.
# MAGIC
# MAGIC **How?** It merges partitions that are already on the **same executor** into one bigger partition. Data on executor 1 stays on executor 1; data on executor 2 stays on executor 2. Only when it *has* to move data across executors does a shuffle happen.
# MAGIC
# MAGIC **Why prefer coalesce over repartition for shrinking?** Repartition always shuffles. Coalesce usually doesn't.
# MAGIC
# MAGIC **What to look for:** a `Coalesce(N)` node. If there's **no `Exchange` next to it**, no shuffle happened.

# COMMAND ----------

coalesced_df = orders.coalesce(1)
coalesced_df.explain(True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### What you will see
# MAGIC
# MAGIC - `Coalesce 1` — and usually **no `Exchange`**, meaning Spark merged partitions within executors.
# MAGIC - If you tried `coalesce(2)` from a very large partition count, you'd start seeing `Exchange` — an aggressive reduction forces a shuffle.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Joins — sort-merge join and hash partitioning
# MAGIC
# MAGIC **What happens during a join?** Both sides get shuffled so that **rows with the same join key land in the same partition**. That way Spark can compare matching keys side-by-side.
# MAGIC
# MAGIC **Hash partitioning formula:**
# MAGIC
# MAGIC ```
# MAGIC target_partition = hash(join_key) mod number_of_shuffle_partitions
# MAGIC ```
# MAGIC
# MAGIC Example with 4 shuffle partitions:
# MAGIC
# MAGIC | Key | hash(key) | mod 4 | Goes to partition |
# MAGIC |---|---|---|---|
# MAGIC | 1 | 1 | 1 | 1 |
# MAGIC | 2 | 2 | 2 | 2 |
# MAGIC | 3 | 3 | 3 | 3 |
# MAGIC | 4 | 4 | 0 | 0 |
# MAGIC
# MAGIC Same key on **both** sides → **same partition** → join can happen locally.
# MAGIC
# MAGIC **Sort-merge join:** after the shuffle, each partition is **sorted** by key. Then merging two sorted lists is cheap — just walk both pointers.
# MAGIC
# MAGIC ⚠️ **Free Edition note:** use **hints** to force sort-merge. `spark.conf.set(...)` is unreliable on serverless.

# COMMAND ----------

joined_df = (
    orders.hint("merge")
          .join(customer.hint("merge"), on="custkey", how="inner")
)
joined_df.explain(True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### What you will see
# MAGIC
# MAGIC Reading the plan **bottom to top** (that's the Spark convention):
# MAGIC
# MAGIC 1. `FileScan` on `customer` — with `IsNotNull(custkey)` filter (Spark's own optimization).
# MAGIC 2. `Exchange hashpartitioning(custkey, 200)` — shuffle the customer side.
# MAGIC 3. `Sort` — sort by `custkey` within each partition.
# MAGIC 4. Repeat steps 1–3 for `orders`.
# MAGIC 5. `SortMergeJoin` — walk both sorted sides.
# MAGIC 6. `Project` — keep only the columns you asked for.
# MAGIC
# MAGIC If you see `BroadcastHashJoin` instead, one side was small enough that Spark just sent it to every executor. That's also valid — often *faster* for small tables. The `hint("merge")` just makes the sort-merge example visible for learning.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Group by — count
# MAGIC
# MAGIC **Naive approach (bad):** shuffle all raw rows, then count.
# MAGIC
# MAGIC **What Spark actually does (good): partial aggregation first.**
# MAGIC
# MAGIC 1. **Partial count** on each partition — count within the partition only.
# MAGIC 2. **Shuffle** — bring same keys together.
# MAGIC 3. **Final count** — add up the partial counts.
# MAGIC
# MAGIC This is called **divide and conquer**: you only move small partial results across the network, not every row.
# MAGIC
# MAGIC **What to look for:** two `HashAggregate` nodes (`partial_count` and `count`) separated by an `Exchange`.

# COMMAND ----------

count_df = orders.groupBy("orderstatus").count()
count_df.explain(True)
display(count_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### What you will see
# MAGIC
# MAGIC - `HashAggregate keys=[orderstatus], functions=[partial_count(1)]` — local count per partition.
# MAGIC - `Exchange hashpartitioning(orderstatus, 200)` — shuffle on group key.
# MAGIC - `HashAggregate keys=[orderstatus], functions=[count(1)]` — final total.
# MAGIC
# MAGIC The word **partial** is your visual cue that Spark did the divide-and-conquer optimization.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Group by — sum
# MAGIC
# MAGIC Same idea as count, just with `sum` instead of `count`. Look for `partial_sum` in the first `HashAggregate` and `sum` in the final one.

# COMMAND ----------

sum_df = orders.groupBy("orderstatus").agg(F.sum("totalprice").alias("total_price"))
sum_df.explain(True)
display(sum_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### What you will see
# MAGIC
# MAGIC - First `HashAggregate` → `partial_sum(totalprice)` (local).
# MAGIC - `Exchange` on `orderstatus`.
# MAGIC - Second `HashAggregate` → `sum(totalprice)` (final).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Group by — count distinct (the tricky one)
# MAGIC
# MAGIC `countDistinct` needs to make sure each distinct value is counted **once globally** — not once per partition. That's why Spark does **extra work** here.
# MAGIC
# MAGIC **The 4-step dance:**
# MAGIC
# MAGIC 1. **Local distinct** — collapse duplicates *within* each partition.
# MAGIC 2. **Shuffle #1** on `(group_key, distinct_col)` — bring all copies of a value to the same partition.
# MAGIC 3. **Global distinct** — collapse duplicates again across the cluster.
# MAGIC 4. **Partial count + shuffle #2 + final count** — count distinct values per group.
# MAGIC
# MAGIC **Result:** **four `HashAggregate` nodes** and **two `Exchange` nodes**. That's why `countDistinct` is expensive — use it sparingly.

# COMMAND ----------

distinct_df = orders.groupBy("custkey").agg(
    F.countDistinct("orderstatus").alias("distinct_statuses")
)
distinct_df.explain(True)
display(distinct_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### What you will see
# MAGIC
# MAGIC Count the `HashAggregate` nodes — there will be **four**. Count the `Exchange` nodes — there will be **two**.
# MAGIC
# MAGIC The keys of the first `HashAggregate` will be `[custkey, orderstatus]` (not just `custkey`) — that's Spark doing the "local distinct" step before the first shuffle.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Interesting observation — the "redundant" filter
# MAGIC
# MAGIC If you look back at the narrow-transformation plan (section 4), you'll notice:
# MAGIC
# MAGIC - The **FileScan** already pushed `orderstatus = 'O'` to the source.
# MAGIC - Spark **still** adds a separate `Filter` node afterwards that does the *same thing*.
# MAGIC
# MAGIC **Why?**
# MAGIC - **Pushdown happens in the logical plan** (Catalyst).
# MAGIC - **The post-scan filter is added in the physical plan** — because *not every filter can be pushed*. Filters on `map` columns, on `CAST` expressions, on non-supported data sources, etc. simply can't go to the source. So Spark always adds a **fail-safe filter** to guarantee correctness.
# MAGIC
# MAGIC **Is it wasteful?** Not really. The pushed filter already cut down the rows at the source. The post-scan filter only runs on a **much smaller dataset**. It's a tiny cost for a big correctness guarantee.

# COMMAND ----------

redundant_df = orders.filter(F.col("orderstatus") == "O")
redundant_df.explain(True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### What to look for
# MAGIC
# MAGIC In the output, find:
# MAGIC - `PushedFilters: [..., EqualTo(orderstatus,O)]` (inside `FileScan`)
# MAGIC - **And** a `Filter (isnotnull(orderstatus) AND (orderstatus = O))` node below the scan.
# MAGIC
# MAGIC Both are present. That's not a bug — it's the design.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. When does predicate pushdown **not** work?
# MAGIC
# MAGIC Two classic failures:
# MAGIC
# MAGIC ### 12.1 Filter on a `map` column key
# MAGIC If you have a `map<string, string>` column (like a `properties` column) and you filter on `properties["color"] == "red"`, the Parquet reader **cannot** evaluate that — map lookups are done by Spark, not by the storage layer. So no pushdown.
# MAGIC
# MAGIC ### 12.2 Filter on a `CAST` expression
# MAGIC If a column is stored as `string` but you cast it to `int` and filter on the cast (`cast(age as int) > 50`), the source has no way to apply that predicate — it doesn't know how to cast. So only `IsNotNull` gets pushed; the cast-and-compare happens after the scan.

# COMMAND ----------

cast_filter_df = orders.filter(F.col("orderkey").cast("string").startswith("1"))
cast_filter_df.explain(True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### What to look for
# MAGIC
# MAGIC - `PushedFilters: [IsNotNull(orderkey)]` — only the null-check was pushed.
# MAGIC - The actual cast + `startswith` logic appears **after** the scan in a `Filter` node.
# MAGIC
# MAGIC **Lesson:** predicate pushdown is **data-source dependent**. Some databases (e.g., Snowflake, BigQuery) push down much more than Parquet does. Always check the plan.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Cheat sheet — plan patterns at a glance
# MAGIC
# MAGIC | Operation | Key nodes in plan | Shuffle? |
# MAGIC |---|---|---|
# MAGIC | **filter / select / withColumn** | `FileScan` (with `PushedFilters`), `Filter`, `Project` | ❌ |
# MAGIC | **repartition(n)** | `Exchange RoundRobinPartitioning(n)` | ✅ |
# MAGIC | **coalesce(n)** | `Coalesce(n)` (usually no `Exchange`) | ⚠️ only if aggressive |
# MAGIC | **sort-merge join** | 2× `Exchange hashpartitioning`, 2× `Sort`, `SortMergeJoin` | ✅ both sides |
# MAGIC | **broadcast join** | `BroadcastHashJoin` (no `Exchange` on small side) | ✅ one side |
# MAGIC | **groupBy().count()** | `HashAggregate(partial_count)` → `Exchange` → `HashAggregate(count)` | ✅ |
# MAGIC | **groupBy().sum()** | `HashAggregate(partial_sum)` → `Exchange` → `HashAggregate(sum)` | ✅ |
# MAGIC | **groupBy().agg(countDistinct)** | **4×** `HashAggregate`, **2×** `Exchange` | ✅ twice |
# MAGIC | **filter on CAST** | `PushedFilters` incomplete; cast-filter appears post-scan | ❌ |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Free Edition gotchas — keep these in mind
# MAGIC
# MAGIC | Don't do this | Why | Do this instead |
# MAGIC |---|---|---|
# MAGIC | `df.rdd.getNumPartitions()` | RDD API blocked on serverless | Read `Exchange N` from `df.explain(True)` |
# MAGIC | `df.rdd.map(...)` | RDD API blocked | `df.withColumn(...)` or `mapInPandas` |
# MAGIC | `spark.conf.set("spark.sql.autoBroadcastJoinThreshold","-1")` | Many confs are ignored on serverless | `df.hint("merge")` / `df.hint("broadcast")` |
# MAGIC | Write to `dbfs:/...` | Not used in Free Edition | `df.write.saveAsTable("workspace.default.my_table")` |
# MAGIC | `pip install <rare-package>` at runtime | Not whitelisted | Use pre-installed libs (pyspark, pandas, scikit-learn) |
# MAGIC | Scala or R notebooks | Not supported | Python or SQL only |
# MAGIC | Huge datasets | 2X-Small SQL warehouse / limited daily quota | Use `samples.*` tables or filtered subsets |
# MAGIC
# MAGIC **How to import this file into Databricks**
# MAGIC
# MAGIC 1. Save this file locally as `Spark_Query_Plans_Free_Edition.py`.
# MAGIC 2. In Databricks: **Workspace** → open any folder → **⋮** menu → **Import**.
# MAGIC 3. Choose **File** → pick the `.py` file → Import.
# MAGIC 4. It opens as a notebook. Click **Run All** at the top.
# MAGIC
# MAGIC **Done!** You now have the full notes + runnable code in **one file**, matching what the transcript taught, and tuned for Free Edition.

# COMMAND ----------

# MAGIC %md
# MAGIC ### ✅ End of notebook — you've covered every topic from the transcript