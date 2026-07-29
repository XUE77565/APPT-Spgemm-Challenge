# 工程优化 — 小白问答(Q&A)

> 这份文件把讲解工程优化(h2d/d2h、pinned 池、kernel fusion 等)时问到的几个基础问题整理在一起,每个名词都从头解释。
> 配合 `inno/engopt_slide_summary.md`(总览)和图 `fig/engopt_slide.png`、`fig/pinned_pool_detail.png`、`fig/launch_fusion_detail.png` 一起看。

---

## Q1:"upload" 和 "download" 是不是 CPU 向 GPU 上传数据、下载结果?

**对,完全正确。**

| 图里的词 | 含义 | 谁到谁 | 例子 |
|---|---|---|---|
| **upload / h2d**(host→device) | 上传**输入数据** | CPU → GPU | 把稀疏矩阵 A(row_ptr / col_idx / val)从 CPU 内存送进 GPU 显存,GPU 才能开始算 |
| **download / d2h**(device→host) | 下载**计算结果** | GPU → CPU | 算完的结果矩阵 C,从 GPU 显存取回 CPU 内存 |

```
CPU 内存                  PCIe(数据通道)              GPU 显存
  │                                                        │
  │── upload / h2d ──输入 A──────────────────────────────→│  (GPU 开始算 A·A)
  │                                                        │
  │←──────────────download / d2h ── 结果 C─────────────────│
```

- **host** = CPU 这一侧(主机);**device** = GPU 这一侧(设备)。**h**ost **to** **d**evice = CPU→GPU,反过来 d2h = GPU→CPU。
- 发生在**真正计算的前后**:先 upload 输入 → GPU 算 → download 结果。

---

## Q2:host 是 lock-free 还是 unlock?

**都不是 —— 这里的 "lock" 指的是"锁页(page-lock / pin)",跟"无锁(lock-free)"完全是两码事,只是英文都叫 lock。**

| 名词 | 中文 | 是什么 | 跟我们有关吗 |
|---|---|---|---|
| **page-lock / pin** | **锁页 / 钉住** | 把内存"固定住",不让操作系统移动或换出它 | ✅ **就是这个**,pinned memory 说的就是它 |
| **lock-free** | **无锁(并发)** | 并发编程里"不用互斥锁"的算法(如原子操作) | ❌ 完全无关,英文恰好也叫 lock |

**为什么要把 host 内存"锁住"?** 普通内存操作系统随时可能挪动/换到硬盘(换页),GPU 正在直接读它时一挪就乱。所以:
- **锁住(lock / pin / cudaMallocHost)** = 告诉操作系统"这块内存固定住,别动" → GPU 才能直接、满速地读它(DMA)。
- **解锁(unlock / unpin / cudaFreeHost)** = "现在可以动了"。

所以 host 内存是 **page-LOCKED(锁页 / pinned)**——这是好事(让 GPU 能直接满速搬数据)。**不是 lock-free,也不是 unlock。**

池化前后,host 内存**始终是锁着的**(这正是我们想要的);省掉的只是"反复重新锁/解锁"的那套开销。

---

## Q3:`pool.get() O(1) bump` 是什么意思?

拆三个词:

**① `pool.get()`** = 从池子里**拿一块缓冲来用**。

**② `O(1)`** = **常数时间**——不管池子多大、已经发出去多少块,拿一块花的时间都一样(极短)。对比 `O(n)`(越往后越慢)。

**③ "bump"** = **指针往前"撞"一下**(bump allocator,指针前移分配器)。池子有一个游标(cursor)指着"下一块空位":

```
池子(预先锁好的一大片):  [已用][已用][已用][←游标][空][空][空]...
                                          ↑
                                     get() 返回这里,游标往后挪一格
```

- `get()` = 把游标当前位置给你 → 游标前挪一格。没有"找空位"、没有"记账"、没有"问操作系统",就是挪个指针 → O(1)、几乎 0 开销。
- `pool.return()` 也一样 O(1)(游标 reset 或标记可复用,no-op)。

> **生活类比:** 旅馆前台一排已备好的房卡,标记处抽一张、标记后挪一位(bump),一秒搞定;对比 `cudaMallocHost` = 每次找经理现找空房、办入住、登记(锁页 syscall),慢。

| 操作 | 干了啥 | 花费 |
|---|---|---|
| `cudaMallocHost`(没池) | 找空闲页 + 锁页 + 改页表,**系统调用** | 慢(几十微秒~毫秒) |
| `pool.get()` bump | **挪一下指针**,池子已预先锁好 | 极快(O(1),纳秒级,无系统调用) |

---

## Q4:pool 是不是预先分配一大段 pinned,GPU block 算的时候从里取?

**前一半对,后半句要纠正。**

✅ **对的部分:** pool = 预先分配一大段内存,整片锁页(pin)一次,切成很多小缓冲,反复用 → 循环里 0 次锁页。

❌ **纠正:** 这个 pinned 池在 **CPU(主机/host)侧**,给 **h2d/d2h 搬运**用,由 **CPU** 取/还。**GPU block 真正算的时候用的是 GPU 自己的显存(device memory),不从这个池取。**

| | HOST pinned pool(我们优化的) | GPU device memory(显存) |
|---|---|---|
| 在哪 | CPU 内存(主机) | GPU 显存(设备) |
| 谁用 | **CPU** 搬运时取缓冲 | **GPU block** 真正计算时用 |
| 怎么来 | 预先锁页一大段、bump 切块 | `cudaMalloc` 在 GPU 上单独分配 |
| 干嘛 | 当 h2d/d2h 的中转缓冲 | 存矩阵、算结果 |

**完整时间线:**
```
CPU 侧                                    GPU 侧(显存)
─────────                                ──────────
1. pool.get() 拿一块 pinned 缓冲,放输入 A
2. ──── h2d 上传 ──────────────────────→  A 进显存
                                          3. GPU block 在显存里算 A·A
                                             (用 device memory,不是 pinned 池)
                                          4. 算完,结果 C 在显存
5. pool.get() 再拿一块 pinned 缓冲  ←──── d2h 下载 ────
   接收结果 C
6. pool.return() 还回去
```

**pinned 池只在第 1、5 步(搬运)出场;第 3 步 GPU 算的时候用的是显存,跟这个池无关。**

> 顺带:我们还试过 **device pool(GPU 显存里的池)**——那个才是给 GPU block 算的时候省 `cudaMalloc` 用的,但实测让小阵变慢 → **回退关掉**(图里 "device pool reverted → OFF")。所以现在只有 host pinned 池在用,服务于搬运。

---

## Q5:代码里是把整个 A 都传上去,还是一行一行传?

**整个 A 一次传上去,整个 C 一次传下来 —— 不是逐行。** 逐行传会慢得离谱(几万次小传输,开销爆炸)。

**传输粒度 = 按整个数组(CSR 的 3 个大数组),不是按行:**
```
上传(h2d):  整个 row_ptr  ──┐
            整个 col_idx  ──┼──→ GPU 显存(整个 A 在上面,GPU 才能算 A·A)
            整个 val      ──┘
下载(d2h):  整个 C 的 row_ptr / col_idx / val  ←── 结果整个取回
```

> 为什么必须整个 A 在 GPU 上?因为算 C 的每一行都可能用到 A 的任意一行/列(SpGEMM 不是流式算法,没法"来一行算一行")。

**池子在这里怎么用:** 每次要传一个数组,`pool.get(这段数组的大小)` 从池子里切出**一整段同样大小的 pinned 缓冲**(游标 bump 前移),用来搬。所以 `pool.get()` 拿的是整段(整个数组那么大),不是一小块、更不是一行。

**最大头 = 下载结果 C(为什么之前反复锁页很痛):**

| 传什么 | 大小(bcsstk30) | 之前(没池)每次都干嘛 |
|---|---|---|
| 上传 A | ~12 MB | cudaMallocHost 锁 12MB → 传 → 解锁 |
| **下载 C** | **~107 MB** | **cudaMallocHost 锁 107MB → 传 → 解锁**(巨贵) |

下载 C 这步,之前**每次调用都重新锁 107MB 的页**——这就是 58.7ms 的大头。池化后:这 107MB 的缓冲**开工时就锁好**,放池子里,每次下载直接拿来用,不再锁。

---

## Q6:async(异步)是什么意思?

**async = 异步 = "不等"**(发起就转头干别的);**sync = 同步 = "干等"**(发起后卡在那等完成)。

**生活类比(洗衣机):**
| | 你怎么做 | 效率 |
|---|---|---|
| **sync** | 丢进洗衣机,**站门口死等**洗完才走 | 被卡住,啥也干不了 |
| **async** | 丢进洗衣机、按启动,**立刻去干别的**,后台自己转 | 你和洗衣机**同时干活** |

**放到 GPU 搬运(`cudaMemcpyAsync`):**
- sync 搬运:CPU 发起后**干等**搬完才继续 → CPU 被卡住。
- async 搬运:CPU 发起后**立刻干别的**,数据在 PCIe **后台**搬 → CPU 和搬运**并行(overlap)**,可以把搬运藏进计算时间里。

```
sync:    CPU[发搬运]──等──等──等──[搬完,继续]          CPU 卡住
async:   CPU[发搬运]──[干别的活]──[干别的活]──[搬完?]   CPU 和搬运并行
                 └──── 数据在 PCIe 后台搬 ────┘
```

**关键:async 搬运必须用 pinned 内存。** pinned → 能 async + 满速 DMA;pageable → 只能 sync(驱动得中转,CPU 干等)。所以图里 "cudaMemcpyAsync · DMA · full PCIe" 是连一起的:**因为用了 pinned,所以能 async、能满速**。

> 一句话:async(异步)= CPU 发起搬运后不等、转头干别的,数据后台搬、能和计算重叠 → 更快;只有 pinned 内存能 async。这是 pinned 池化的第二个收益(第一个是不用每次锁页)。

---

## Q7:no-op 是什么意思?

**no-op = "no operation" = 空操作 = 啥也不干。** 一个调用在那儿,但执行起来没有任何效果、几乎不花时间。

**为什么 `pool.return()` 是 no-op:** 还缓冲回池子不用真释放、也不用记账——缓冲留在池子标记"可复用"即可。所以这个调用实际上啥也不做 → no-op。

| 操作 | 真的干了啥 | no-op? |
|---|---|---|
| `cudaFreeHost`(没池,要释放) | 解锁页 + 还内存给 OS,**系统调用** | ❌ 有活,慢 |
| `pool.return()`(我们的池) | **啥也不干**,留在池子等复用 | ✅ no-op,瞬间 |

**类比(旅馆房卡):** cudaFreeHost = 正式退房(注销、登记、清房,慢);pool.return() = 把房卡扔回那堆卡里就完事(不用手续,卡留下个客人直接拿)。

**这是池化快的另一半:** `pool.get()` 是 bump(O(1) 取),`pool.return()` 是 no-op(0 开销还)——循环里取和还都 ~0 开销:
```
没池:  cudaMallocHost(锁,慢) ── 用 ── cudaFreeHost(解锁,慢)    两头都付系统调用
有池:  pool.get()(bump,瞬间) ── 用 ── pool.return()(no-op,瞬间) 两头都 ~0 开销
```

> 一句话:no-op = 空操作(啥也不干);pool.return() 是 no-op,因为还缓冲不用真释放/记账。和 pool.get() 的 bump 一起,让循环里取/还都 ~0 开销——这是池化快的两半(bump 取 + no-op 还)。

---

## Q8:pin 了就能用 DMA 直接传输了是吗?

**对,完全正确。** pin(锁页)固定了物理地址,GPU 的 DMA 引擎就能直接读写 host 内存。

**DMA = Direct Memory Access = 直接内存访问。** GPU 的 **DMA 引擎**能自己通过 PCIe 直接读写 host 内存,不用 CPU 一个字节一个字节搬;CPU 下个单(源/目的地址 + 字节数)就可以走开,DMA 引擎自己搬完。

**为什么必须 pin 才能 DMA:** DMA 直接读写需要**物理地址固定、不能动**。
- **pinned(锁页)= 地址固定** → DMA 直接读 → 一步、满速 PCIe、能 async、不占 CPU ✅
- **pageable(普通)= 地址随时可能变**(OS 可能挪/换页)→ DMA 不敢直接读 → 驱动先复制到 pinned 中转区再 DMA(staging)→ 多一次复制、不能 async、慢 ❌

| | pinned(锁页) | pageable(普通) |
|---|---|---|
| DMA 直接读? | ✅ 能 | ❌ 得先中转 |
| 路径 | host ←DMA→ GPU(一步) | host → 中转 → DMA → GPU(两步) |
| 速度 | PCIe 满速 | 慢一截 |
| async? | ✅ 能 | ❌ 不能 |

**类比:** DMA = 直通车,直接到你家(固定地址)拉货;pinned = 地址固定 → 直通车直接来;pageable = 地址可能变 → 直通车不敢来,得先搬到中转站(固定地址)再发,多一道。

> 一句话:pin 固定物理地址 → DMA 引擎直接读写 host(满速、async、不占 CPU);普通内存地址会变,DMA 得先中转 → 慢。这就是图里 "DMA · full PCIe" 的来历。

---

## Q9:large matrix 里的 "no regression" 是什么意思?

**no regression = 没退化 / 没变差。** regression(退化/回退)= 改了代码后某方面比之前更差;no regression = 没让它变差。

**在这里:** fusion 只改了**小矩阵**的路径(A_rows≤1024 用单 block scan);**大矩阵**走另一条路(还用 thrust,gate 挡住)。所以:
- 小阵:变快(bp* 0.03–0.08→0.008ms)✅
- 大阵:走原路径 → **理应和以前一样快** → 实测 3.24→3.11ms(没变慢,甚至略快)= **no regression** ✅

| | 小阵(优化目标) | 大阵(没动它的路径) |
|---|---|---|
| 期望 | 变快 | 不变(不退步) |
| 实测 | 0.03–0.08→0.008ms | 3.24→3.11ms = **no regression** |

**为什么强调:** 优化一条路径最怕"捡芝麻丢西瓜"——为了小阵把大阵也搞慢。no regression = 确认没被优化的部分**没变差**。

**类比:** 给小房间换节能灯(省电),要确认大房间没因此变暗——大房间亮度不变 = no regression。

**软件里的来历:** regression 专指"改了代码后原来好的东西变差";"回归测试(regression test)"= 改完跑老测试确保没搞坏旧的。这里 no regression = 性能没回归(没掉)。

> 一句话:no regression = 没退化、没变差;这里指 fusion 只优化小阵,**大阵路径没被动、没变慢**(3.24→3.11ms),证明这个优化"只帮忙、不添乱"。

---

## Q10:为什么大阵不用单 block scan?

两个原因:

**① 硬件限制 —— 一个 block 最多 1024 个线程。** 单 block Hillis–Steele scan 是 1 个 block、每线程 1 个元素;GPU 一个 block 上限 1024 线程。小阵 A_rows≤1024 装得下 ✅;大阵 >1024 **装不下** ❌。大阵要拆多 block,一拆就回到 thrust 那套(跨 block 协调 = 多 kernel),launch 开销又请回来了,白优化。

**② 大阵 scan 占比极小,没收益。** 大阵 accumulate 占 77%,scan 可忽略;小阵计算≈0、launch 主导才值得压成单 kernel。所以:小阵优化有收益,大阵优化没收益。

```
A_rows ≤ 1024  →  单 block scan  (1 kernel,小阵省 launch)
A_rows > 1024  →  thrust scan    (能处理大数组,大阵这点开销无所谓)
```

**类比:** 单 block scan = 一个最多 10 桌的小餐厅;≤10 桌一个店搞定(快),50 桌装不下只能拆多店 + 协调(= thrust,又慢又复杂)。小客流用小店、大客流用连锁,各取所长;且大客流时接待时间不算啥(后厨炒菜=accumulate 才是大头)。

> 一句话:大阵不用单 block scan,因为 ① 一个 block 最多 1024 线程装不下;② 拆多 block 等于 thrust、开销请回来;③ 大阵 scan 占比极小没收益。按行数 gate:小阵单 block 省 launch,大阵 thrust,各取所长。

---

## Q11:bank conflict 是什么意思?

**bank conflict = 共享内存(SMEM)的"银行冲突" = 多个线程同时访问同一个 bank,导致排队变慢。**

**SMEM 分 32 个 bank**(像 32 个并行收银台),每个宽 4 字节、每周期每 bank 只服务 1 次访问。一个 warp 32 线程,理想是各访问不同 bank → 1 周期全搞定(满速)。

**地址映射:** `bank = (字编号) % 32`,循环轮转(0,1,…,31,0,1,…):

| 32 线程访问模式 | bank | 结果 |
|---|---|---|
| 连续(stride 1) | 各不同 | ✅ 无冲突,1 周期 |
| 2 线程同 bank | 撞一起 | ⚠️ 2-way 冲突,2 周期 |
| 跨步 32(word[0],word[32]…) | 全 bank 0 | ❌ 32-way 冲突,32 周期(最糟) |
| 读同一地址 | 同 bank 同址 | ✅ broadcast,不算冲突 |

**后果:** 撞一起的访问只能排队(serialize),N 个撞 = N 周期 → 慢 N 倍。

**类比:** 32 个收银台,32 顾客各去不同台 = 同时结账(快);几个挤到同一台 = 排队(慢)。关键看"号→收银台"映射:连续号 → 不同台;跨步 32 → 全撞一台。

**为什么重要:** SMEM 快的前提是无 bank conflict;冲突会让它退化成慢速串行。所以 hash 表、scan、count-sort 这些重度用 SMEM 的代码都得避免它(常见修法:padding 填充错开跨步、改 stride-1 访问)。

**跟项目关系:** hash SPA 的 `hash_slot(j)` 映射、count-sort/Hillis–Steele scan 的 SMEM 访问,都可能踩 bank conflict——"count-sort sync-free 地板""scan 用 SMEM"能真快,隐含前提就是避免 bank conflict。

> 一句话:bank conflict = SMEM 32 个 bank 里多个线程同时访问同一个 bank 导致排队变慢(连续访问无冲突,跨步 32 最糟);SMEM 要快就得避免它。

---

## 一句话总览(把 11 个问题串起来)

> 我们算 A·A 时,先把**整个 A**(3 个大数组)**upload 到 GPU 显存**,GPU block 在**显存里**(不是 pinned 池)算出**整个 C**,再 **download 回 CPU**。搬运用的 host 缓冲放在一个**预先锁好页的 pinned 池**里(锁页 = 把内存固定住让 GPU 能满速 DMA,跟"无锁 lock-free"无关);每次搬运用 `pool.get()` **bump 切一段**(O(1)、不惊动操作系统),用完 return,所以循环里 0 次锁页开销 → wall-clock 58.7→9.2ms。
