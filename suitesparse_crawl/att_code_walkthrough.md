# A·Aᵀ 上三角 SpGEMM 代码讲解

> 对应代码:`src/spgemm_kernel_formulations.cu` 的 att 段(line 469 起)。
> 本文档逐层讲解:数学原理 → 数据结构 → 4 种公式的 kernel → compact 机制 → host 流程。

---

## 1. 数学:在算什么

$$C = A \cdot A^\top, \quad C[i,j] = \sum_k A[i,k] \cdot A[j,k]$$

直观理解:**C[i,j] 是 A 的第 i 行和第 j 行的点积**(按列号 k 对齐相乘求和)。

关键性质:
- **对称**:$C^\top = (A A^\top)^\top = A A^\top = C$,即 $C[i,j] = C[j,i]$。
- 因此**只需算上三角 $i \le j$**,下三角由对称性得到。这省掉约一半的中间项 + 输出。

**中间项**:对于 $i \le j$,凡存在 $k$ 使 $A[i,k] \ne 0$ 且 $A[j,k] \ne 0$,就有一条贡献 $A[i,k] \cdot A[j,k]$ 到 $C[i,j]$。
所有中间项的集合:
$$\text{intermediates} = \{(i,j,k) : i \le j,\; A[i,k] \ne 0,\; A[j,k] \ne 0\}$$

---

## 2. ESC 框架(Expand–Sort–Compress)

和自乘 A² 用同一套 ESC 方法论,适配上三角:

```
count(数中间项) → scan(求偏移) → expand(写中间项) → [compact(压紧)] → sort+reduce(合并) → CSR
```

**为什么有的公式需要 compact?**
- **外积**按 $k$ 展开时,$col_k$(列 $k$ 的非零行)中取 $a \le b$ 对,天然 $i \le j$,count 精确 = $c_k(c_k{+}1)/2$,无间隙。
- **Gustavson / 列向**按 $i$ 或 $j$ 展开时,用 filter(`j≥i` 或 `i≤j`)丢掉下三角,实际写出的条目数 < count 的上界 → 行间有间隙 → 需要 compact 压紧。

---

## 3. 公共工具(inline helper)

### 3.1 `pack_key(row, col)` — 把 (row, col) 打包成 64-bit key
```cpp
static __device__ unsigned long long pack_key(int row, int col) {
    return ((unsigned long long)row << 32) | (unsigned int)col;
}
```
- row 放高 32 位,col 放低 32 位。
- 好处:一次 `sort_by_key` 就按 (row, col) 联合排序;一次 `reduce_by_key` 按 key 去重求和。
- 比维护两个独立数组(key_col + key_row)简单。

### 3.2 `dpi()` / `dcpi()` — thrust device_ptr 简写
```cpp
static inline thrust::device_ptr<int> dpi(int *p) { ... }
static inline thrust::device_ptr<const int> dcpi(const int *p) { ... }
```
- thrust 的 `inclusive_scan` / `sort_by_key` 需要把裸指针包成 `device_ptr` 才能用。
- `dcpi` 是 const 版(compact 时扫描只读的 `act[]` 数组用)。

### 3.3 `rows_grid(n)` — 一行一块的 grid 大小
```cpp
static int rows_grid(int n) { return (n + 255) / 256; }
```
用于 count kernel(count 每行一个线程,不是一块)。

---

## 4. `AttCtx` 结构体 — 缓冲区生命周期管理

这是重构后最核心的可读性改进。一次 att 计算用到 **9 块显存**,原来散在 host 里裸指针 + 手动 malloc/free,容易漏。现在全收进一个 struct:

```cpp
struct AttCtx {
    // A 的两种存储格式
    void  *dA;              // 单块 CSR(row_ptr | col_idx | val),从 host H2D 来的
    int   *csr_rp, *csr_ci; // → dA 内部的三个指针(不单独分配,指向 dA 的偏移)
    float *csr_val;
    int   *csc_cp, *csc_ri; // A 的 CSC(build_csc 从 CSR 转出来的,单独分配)
    float *csc_val;

    // 中间项 COO 的工作区
    int   *row_ub;          // 每行中间项数(上界),count kernel 填
    int   *off;             // row_ub 的前缀和 → 每行在 COO 中的起始偏移
    int   *act;             // 每行实际写出的条目数(expand 后填,compact 用)
    unsigned long long *key; // COO 的 key=(row<<32|col)
    float *val;              // COO 的 value

    int A_rows;
};
```

### 四个方法对应 ESC 流程的四个阶段:

| 方法 | 何时调 | 做什么 |
|---|---|---|
| `init(A_buffer, rows, nnz)` | 最开始 | H2D 上传 A + `build_csc` 转 CSC + 分配 `row_ub` |
| `prepare_coo()` | count kernel 之后 | scan(`row_ub`→`off`) + 读 `total_ub` + 分配 `key/val/act`。返回 `total_ub` |
| `compact()` | expand 之后(gust/colw/inner) | scan(`act`→`coff`) + `att_compact_kernel` 搬运 + 回拷。返回 `total_actual` |
| `free_all()` | 最后 | 一次性释放全部 9 块 |

> **`prepare_coo()` 详解**:
> 1. `inclusive_scan(row_ub → off+1)`:得每行偏移。`off[A_rows]` = 总上界 `total_ub`。
> 2. 回拷 `total_ub` 到 host(为了知道要 malloc 多大的 `key/val`)。
> 3. 分配 `key[total_ub]`(u64) + `val[total_ub]`(f32) + `act[A_rows]`。
>
> 注意:`inclusive_scan` 写到 `off+1`,配合 `off[0]=0`(memset),得 `off=[0, ub₀, ub₀+ub₁, …]`。这和旧的 `exclusive_scan` 写到 `off+1` 的 bug 不同——那个会产生 off-by-one。

> **`compact()` 详解**:
> 1. `inclusive_scan(act → coff+1)`:得每行**紧凑后**的偏移。`coff[A_rows]` = `total_actual`。
> 2. 分配临时 `nkey/nval[total_actual]`。
> 3. `att_compact_kernel`:每行一块,把 `key[off[ax]…+act[ax])` 搬到 `nkey[coff[ax]…)`。
> 4. 回拷 `nkey→key`(in-place 安全:coff ≤ off 总成立)。
> 5. 返回 `total_actual`(≤ total_ub)。

---

## 5. 四种公式的 kernel

### 5.1 外积 att(axis = k) — 最干净

**count**:`ub[k] = c_k(c_k+1)/2`,其中 `c_k = |col_k|`(列 k 的非零行数)。

```
att_outer_count: 一线程一行,算 c_k*(c_k+1)/2(上三角对数,含对角)
```

**expand**:**一块一列 k**,遍历 $col_k$ 中所有 $a \le b$ 对:

```cpp
// att_outer_expand 核心循环
for (int a = cs + threadIdx.x; a < ce; a += blockDim.x) {
    int i = csc_ri[a]; float va = csc_v[a];    // A[i,k]
    for (int b = a; b < ce; b++) {              // b ≥ a
        int j = csc_ri[b];                      // A[j,k]
        int lo = min(i,j), hi = max(i,j);       // 强制 lo ≤ hi(上三角)
        key[slot] = pack_key(lo, hi);
        val[slot] = va * csc_v[b];               // A[i,k] * A[j,k]
    }
}
```

> **为什么用 min/max?**
> CSC 列内的行号 **不一定有序**(build_csc 用 atomicAdd 散播,顺序不确定)。所以 $a \le b$ 不保证 $csc\_ri[a] \le csc\_ri[b]$。用 min/max 强制 $lo \le hi$,确保 key 总是上三角。这是一个曾导致 bug 的细节(CSC 无序 → outer 给出 19 而非正确的 15)。

**优势**:count 精确(精确知道写多少 → 无间隙 → **不需要 compact**)。

### 5.2 Gustavson att(axis = i, filter j ≥ i)

**count**:`ub[i] = Σ_{k ∈ row_i} |col_k|`(上界,不 filter)。

```
att_gust_count: 一线程一行 i,遍历 row_i(CSR)的每个 k,累加 |col_k|
```

**expand**:**一块一行 i**,filter `j ≥ i`:

```cpp
// att_gust_expand 核心循环
for (int p = csr_rp[i] + threadIdx.x; p < csr_rp[i+1]; p += blockDim.x) {
    int k = csr_ci[p]; float aik = csr_v[p];    // A[i,k],走 CSR 行 i
    for (int q = csc_cp[k]; q < csc_cp[k+1]; q++) {
        int j = csc_ri[q];                      // A[j,k],走 CSC 列 k
        if (j >= i) {                           // filter: 只保留上三角
            key[slot] = pack_key(i, j);
            val[slot] = aik * csc_v[q];
        }
    }
}
```

- `act[i] = pos - off[i]`:记本行实际写出数(filter 后 ≤ ub[i])。
- 因为 filter,**行间有间隙** → expand 后必须 `ctx.compact()` 压紧。

### 5.3 列向 att(axis = j, filter i ≤ j)

与 Gustavson 镜像,但 axis 换成 j:

**count**:`ub[j] = Σ_{k ∈ row_j} |col_k|`。

> **关键**:$C[:,j]$ 固定 $j$ 时,$k$ 必须满足 $A[j,k] \ne 0$,即 $k \in row_j(A)$(**CSR 行 j**)。不是 $col_j$!这是之前出过 bug 的地方(误用了 CSC col_j)。

**expand**:**一块一列 j**,filter `i ≤ j`:

```cpp
for (int p = csr_rp[j] + threadIdx.x; p < csr_rp[j+1]; p += blockDim.x) {
    int k = csr_ci[p]; float ajk = csr_v[p];    // A[j,k],走 CSR 行 j
    for (int q = csc_cp[k]; q < csc_cp[k+1]; q++) {
        int i = csc_ri[q];                      // A[i,k],走 CSC 列 k
        if (i <= j) {                           // filter: 只保留上三角
            key[slot] = pack_key(i, j);
            val[slot] = csc_v[q] * ajk;
        }
    }
}
```

### 5.4 内积 att(ESC 符号 + 逐元素 merge)

内积不直接展开写值,而是**两阶段**:

**符号阶段**:复用 Gustavson 的 count+expand+compact,得到上三角结构 `(c_col, c_rp)`(c_val 丢弃,后面重算)。

**数值阶段**(`att_inner_numeric`):对结构中每个 $(i, j)$,独立计算 $C[i,j] = \text{row}_i \cdot \text{row}_j$:

```cpp
__global__ void att_inner_numeric(... const int *C_rp, const int *C_ci, float *C_val) {
    int i = blockIdx.x;
    int rs = csr_rp[i], re = csr_rp[i+1];      // row_i 的范围
    for (int t = C_rp[i] + threadIdx.x; t < C_rp[i+1]; t += blockDim.x) {
        int j = C_ci[t];                        // 结构里的 j
        C_val[t] = rowrow_dot(csr_ci, csr_val, rs, re, csr_rp[j], csr_rp[j+1]);
    }
}
```

`rowrow_dot` 是两条 CSR 行的**归并点积**(经典 merge:两个指针,col 相同则乘加,不同则前进小的)。

> **为什么内积需要两阶段?** 逐元素内积 $C[i,j] = \text{row}_i \cdot \text{row}_j$ 需要知道哪些 $(i,j)$ 存在才能算。而"找结构"本身就是 SpGEMM。所以复用 Gustavson 的符号展开拿到结构,再做数值归并。代价是 row_i 被该行每个 j 重读(低效,但"逐元素点积"原汁原味)。

---

## 6. Compact 机制详解

`att_compact_kernel`(device)+ `AttCtx::compact()`(host)配合:

```
expand 后的布局(有间隙):
key:  [----row0----] [----row1----]        [----row2----]
       off[0]..+act[0]  off[1]..+act[1]     off[2]..+act[2]
       ↑ 紧凑          ↑ off[1]=ub0(有空隙)  ↑ off[2]=ub0+ub1

compact 后的布局(连续):
key:  [----row0----][----row1----][----row2----]
       coff[0]       coff[1]       coff[2]
       ↑ = act0      ↑ = act0+act1  ↑ = act0+act1+act2
```

- `coff = inclusive_scan(act)`:每行紧凑后的偏移。
- `total_actual = coff[A_rows-1] + act[A_rows-1]` = sum(act)。
- kernel 把每行的 `act[ax]` 条从 `off[ax]` 搬到 `coff[ax]`。
- 搬完后 key/val 前 `total_actual` 个是紧凑连续的 → 可以直接 `esc_merge(key, val, total_actual)`。

> **为什么不直接精确 count(避免 compact)?** Gustavson/列向的 filter(`j≥i` / `i≤j`)使得每行的精确中间项数需要"检查每个 j 是否 ≥ i",在 count 阶段就要做一遍 expand 的遍历。用上界 + compact 更简单(count 只加行长度,不做 filter 判断)。

---

## 7. `esc_merge`(在 att 段之前定义,att 复用)

```
esc_merge(key[total], val[total], A_rows) → (c_col, c_val, c_rp), 返回 Cnnz
```

三步:
1. **`sort_by_key(key, val)`**:按 key=(row<<32|col) 升序排 → 同 (row,col) 的贡献相邻。
2. **`reduce_by_key(key, val) → (red_key, red_val)`**:相邻同 key 求和 → 去重后的 (key, val)。
3. **`finalize_csr`**:拆 key → col,写 C_col_idx/C_val,atomicAdd 统计每行 nnz → scan 得 C_row_ptr。

> **Cnnz** = reduce 后的条目数 = 上三角 distinct (i,j) 数。

---

## 8. Host 流程(以 Gustavson 为例)

```cpp
void spgemm_att_gust(...) {
    g_tag = "attg"; dbg("[attg] start\n");

    AttCtx ctx;
    ctx.init(A_buffer, A_rows, A_nnz);    // ① H2D + CSC
    dbg("[attg] h2d\n");

    att_gust_count<<<rows_grid(A_rows), 256>>>(...);   // ② 每行中间项上界
    sync; dbg("[attg] count\n");

    ctx.prepare_coo();                    // ③ scan → off + alloc COO
    dbg("[attg] scan\n");

    att_gust_expand<<<A_rows, 256>>>(...); // ④ 展开(filter j≥i,紧凑写,记 act)
    sync; dbg("[attg] expand\n");

    int total = ctx.compact();            // ⑤ 压紧行间间隙
    dbg("[attg] compact\n");

    int Cnnz = esc_merge(ctx.key, ctx.val, total, ...);  // ⑥ sort+reduce → CSR
    *C_buffer_out = pack_and_download(...);               // ⑦ D2D 拼块 + D2H 回传
    ctx.free_all();                                       // ⑧ 释放
}
```

**阶段桩**:`[attg] start / h2d / count / scan / expand / compact`(host 发)+ `[attg] csc / sort / reduce / final / pack / d2h`(build_csc / esc_merge / pack_and_download 内部发,通过 `g_tag`)。profiler 按相邻两戳相减得各阶段耗时。

---

## 9. 四种公式的对比总结

| 公式 | axis | filter | count 精确? | 需 compact? | 数值阶段 | 读什么 |
|---|---|---|---|---|---|---|
| 外积 | k | min/max(天然) | ✓ $c_k(c_k{+}1)/2$ | ✗ | expand 即终值 | CSC col_k |
| Gustavson | i | j≥i | ✗(上界) | ✓ | expand 即终值 | CSR row_i + CSC col_k |
| 列向 | j | i≤j | ✗(上界) | ✓ | expand 即终值 | CSR row_j + CSC col_k |
| 内积 | i(符号) | j≥i(符号) | ✗(上界) | ✓(符号) | 逐元素 rowrow_dot | CSR row_i + CSR row_j |

**结果一致**(4 法产出的上三角 nnz 完全相同),**性能接近**(展开同一批中间项,差异只在访存模式)。外积最干净(精确 count,无 compact),内积最慢(多一遍数值归并)。

---

## 10. 设计要点 / 易错点

1. **CSC 列内不一定有序**(build_csc 用 atomicAdd 散播)→ 外积必须用 min/max 强制上三角,不能靠 $a \le b$。
2. **列向的 k 源是 CSR row_j**($A[j,k] \ne 0$),不是 CSC col_j。早期版本写反了。
3. **`inclusive_scan` 写到 `off+1`** 配合 `off[0]=0`,得正确前缀和 `[0, ub₀, ub₀+ub₁, …]`。旧的 `exclusive_scan` 写到 `off+1` 会有 off-by-one(行 0/1 重叠)。
4. **compact 回拷 in-place 安全**:`coff[ax] ≤ off[ax]`(compact 去间隙 → 偏移只减不增),且 `coff[ax]+act[ax] = coff[ax+1] ≤ off[ax+1]`,所以搬运不会覆盖未读的源。
5. **int 溢出**:count 用 `long long` 累加(`c*(c+1)/2` 或 `Σ|col_k|`),存入 `int` 时假设单行不溢出(对稀疏矩阵成立;极稠密列需注意)。
6. **`shared int pos`**:每个 block(= 每个 axis)一个共享计数器,线程间 `atomicAdd` 领号写 COO → block 内紧凑、跨 block 不冲突。
7. **对称性只存上三角**:输出 CSR 只有 $i \le j$ 的条目。与 cuSPARSE 全量对比时用 $2 \times \text{upper} - \text{diag} = \text{full}$ 校验。
