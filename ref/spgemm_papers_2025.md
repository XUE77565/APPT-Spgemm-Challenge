# SpGEMM 文献检索(第二轮,2025–2026 新发现)

> 日期:2026-07-14
> 范围:四大顶会(ISCA/MICRO/HPCA/ASPLOS)+ SC/PPoPP/ICS/IEEE,近两年(2024–2026)。
> 重点:能落到本项目 SpGEMM(尤其 sort/merge 瓶颈)的工作。
> 上一轮见 `worklog/spgemm_acceleration_paper_survey.md`;创新方向见 `ref/sort_innovation_directions.md`。
> 全文已爬取的放 `ref/papers/`。

---

## 核心判断

这一轮新发现印证:**当前 SpGEMM 加速的主流是"hash 累加器替代排序"**(正好打你的 sort 瓶颈),且有 2025 新作做了 **hash 多阶段**;**shared-mem 高利用率的累加器**也有新作带开源代码。SpGEMM(稀疏×稀疏)的直接相关;SpMM(稀疏×稠密,GNN 常用)只能借鉴访存/调度思想。

---

## A. 直接能用(SpGEMM 稀疏×稀疏)

### A1. Hash-based Multi-phase SpGEMM + AIA ⭐⭐⭐ 最贴瓶颈
- Li et al., [arXiv:2512.12036](https://arxiv.org/abs/2512.12036), 2025
- **hash 多阶段**替代排序 + 加速间接访存(AIA)+ processing-near-HBM。
- **对应方向 B/E**(hash 累加器 + symbolic-numeric 两阶段)。直接证明"hash 替代 sort"是当前热点。
- 全文:`ref/papers/hash_multiphase_spgemm.md`

### A2. HSMU-SpGEMM ⭐⭐ 有开源代码
- IEEE 2025;[GitHub: wuminqaq/HSMU-SpGEMM](https://github.com/wuminqaq/HSMU-SpGEMM)
- **高 shared-memory 利用率的累加器**,专为现代 GPU。有代码可读。
- **对应方向 D**(融合单 kernel,shared-mem 排+去重)。做 block-per-row 在 shared mem 里处理的现成参照。
- 摘要:`ref/papers/hsmu_spgemm.md`

### A3. Ocean: Fast Estimation-Based SpGEMM ⭐⭐
- [arXiv:2604.19004](https://arxiv.org/html/2604.19004v1), 2026
- **估计式**(estimation)避免完全展开中间项,减少要排的数据量。
- **对应方向 C**(缩小 sort 输入)和 E(不展开)。estimation 是你没探索过的新角度。
- 全文:`ref/papers/ocean_estimation_spgemm.md`

### A4. Register-Based / Register-Aware SpGEMM
- [Register-Aware, NPC'18](https://pacman.cs.tsinghua.edu.cn/npc2018/papers/register-aware.pdf);[reg-spgemm](https://scispace.com/pdf/register-based-implementation-of-the-sparse-general-matrix-60f3fai3tr.pdf)
- **register + shared-mem 累加器**,N-to-M 设计;Register-Aware 系统对比 sort/merge/hash 三种 accumulator。
- **对应方向 A/B/D**。"该选哪种 accumulator"的决策奠基。

### A5. MH-SpGEMM
- 2025;负载均衡 + 内存预分配。工程性参考(对应 arena/预分配思路)。

---

## B. 可借鉴(SpMM 稀疏×稠密,算法不同但思想可迁移)

> 注意:这些是 SpMM(稀疏×稠密,GNN 常用),和你 SpGEMM(稀疏×稀疏)算法不同,不能直接套,但**访存/调度思想**可借鉴。

- **Swift**(HPCA 2026,[链接](https://2026.hpca-conf.org/details/hpca-2026-main-conference/42/))— SpMM 加速矩阵加载(对照你的 h2d/d2h)。
- **GUST**(ASPLOS 2025)— graph edge-coloring 加速稀疏矩阵。**和你 gust 方法同名但不同**(图边着色减中间冲突)——edge-coloring 思路或可借鉴来减少中间项重复(方向 C)。
- **ACES**(2024)— 自适应执行流(adaptive execution flow),随稀疏模式动态调整(对应 Spada / 方向 B)。
- **KAMI**(2025,[PDF](https://www.ssslab.cn/assets/papers/2025-wang-KAMI.pdf))— 单 GPU 内的 communication-avoiding 矩阵运算。和你的 d2h/h2d 传输优化相关(CA 理论给框架)。

---

## 映射到创新方向(见 `ref/sort_innovation_directions.md`)

| 方向 | 最相关新文献 |
|---|---|
| **A: k-way merge 替代排序** | Register-Aware(merge accumulator)、SpArch —— GPU merge 仍是空白 |
| **B: 代价模型驱动的逐行自适应** | spECK、Register-Aware、ACES/Spada、**A1 hash 多阶段** |
| **C: expand 局部去重缩 sort** | **A3 Ocean(estimation)**、GUST(edge-coloring)、NSparse |
| **D: 融合单 kernel(shared mem)** | **A2 HSMU(有代码)**、Register-Based |
| **E: symbolic-numeric 两阶段** | **A1 hash 多阶段**、cuSPARSE、Ocean |

---

## 读哪几篇(优先级)

1. **A1 Hash-based Multi-phase SpGEMM**(arXiv:2512.12036)—— 最新、最贴 sort 瓶颈。**优先**。
2. **A2 HSMU-SpGEMM**(GitHub 有代码)—— 想做方向 D,直接看实现。
3. **A3 Ocean**(estimation)—— 方向 C 的新角度。

## 来源
- [arXiv:2512.12036 (hash 多阶段)](https://arxiv.org/abs/2512.12036)
- [GitHub: HSMU-SpGEMM](https://github.com/wuminqaq/HSMU-SpGEMM)
- [arXiv:2604.19004 (Ocean)](https://arxiv.org/html/2604.19004v1)
- [Register-Aware (NPC'18)](https://pacman.cs.tsinghua.edu.cn/npc2018/papers/register-aware.pdf)
- [Swift (HPCA 2026)](https://2026.hpca-conf.org/details/hpca-2026-main-conference/42/)
- [KAMI (2025)](https://www.ssslab.cn/assets/papers/2025-wang-KAMI.pdf)
- [ASPLOS 2025 program](https://www.asplos-conference.org/asplos2025/program.html)
