# SpGEMM 加速:体系结构顶会文献调研

> 调研日期:2026-07-13
> 目的:从体系结构四大顶会(ISCA / MICRO / HPCA / ASPLOS)及相邻会议(SC / PPoPP / PACT)近几年的工作中,找能用于本项目 SpGEMM GPU 加速(H100)的成果。
> 背景瓶颈(见 `suitesparse_crawl/profiling_analysis.md`):
>   **#1 d2h 传输**(下载 C + `cudaMallocHost` 锁页)占总耗时 ~60%;
>   **#2 ESC 合并的 `sort_by_key`**(手写法大矩阵 ~30–40%,bcsstk30 上 21.6ms)。

---

## 0. 一句话结论

- **论文能帮上忙的,集中在 #2(干掉 sort)**——见 §A 的 5 篇。
- **#1(d2h)没有顶会论文可借鉴**——是 CUDA API 工程问题,`suitesparse_crawl/transfer_optimization.md` 那套(pinned 池 / `KEEP_ON_DEVICE`)已到顶。
- **关键认识**:这些工作不是"**加速** sort_by_key",而是"**绕过全局排序**"——用 **hash 累加器**或**分段归并**替代 sort+reduce。sort 不是要优化,是要消除。

---

## A. 直接能用——换掉 ESC 的 sort+reduce(对应瓶颈 #2)

这一类把"sort_by_key → reduce_by_key"换成 hash 累加器或分段归并,正好解决手写法大矩阵输给 cuSPARSE 的根因(sort 随中间项数膨胀)。

| 工作 | 会议/年 | 核心点 | 对本项目的价值 |
|---|---|---|---|
| **Register-Aware SpGEMM**(清华 pacman) | [NPC 2018](https://pacman.cs.tsinghua.edu.cn/npc2018/papers/register-aware.pdf) | 在 GPU **寄存器**里对比 sort / merge / hash 三种 accumulator | **最该读**:我们现在的 ESC sort 就是其中一种,这篇告诉我们什么场景换 hash/merge、怎么放寄存器 |
| **spECK**(Steinberger, Graz) | [PDF](https://www.tugraz.at/fileadmin/user_upload/Institute/ICG/Downloads/team_steinberger/Publications/spECK.pdf) | 可适配的 **hash accumulator**,按行特征动态调 | 直接替换 sort,最对口 |
| **Segmented Merge** | [SSSLab 2021](https://www.ssslab.cn/assets/papers/2021-ji-segmerge.pdf) | "分段归并"原语替代全局 sort | 免 sort 的另一条路(比 hash 更确定,利用同行贡献连续) |
| **Optimizing General SpGEMM on GPU** | [ACM](https://dl.acm.org/doi/10.1145/3774654) / [BIT](https://pure.bit.edu.cn/en/publications/optimizing-general-sparse-matrix-matrix-multiplication-on-the-gpu/) | hash load-factor / multiplier 启发式选取 | hash 实现的调参细节 |
| **NSparse**(Nagasaka) | [GitHub](https://github.com/EBD-CREST/nsparse) / [论文](https://www.semanticscholar.org/paper/High-Performance-and-Memory-Saving-Sparse-General-Nagasaka-Nukada/2d95653f4cd2a227ae2ffc1a745570322f53ec57) | merge-preprocessing 的 GPU SpGEMM,省内存 | 开源 GPU SpGEMM 库,可直接对照/借 merge 实现 |

> **路线选择**:hash(spECK / Register-Aware 的 hash 变体)对高度不规则、长行矩阵友好;分段归并(Segmented Merge)对"同行贡献密集"更稳。Register-Aware 在同一框架对比三者,是决策的最佳入口。

---

## B. 方法论 / 算法天花板——验证方向 + 偷数据流

这些**大多不是讲 sort**,而是讲**用哪种公式/数据流**——对应我们 gust/outer/colw/inner 的对比(已实测)。除 SpArch 外基本是 ASIC 提案,不能跑在 H100,但算法/数据流是天花板参考。

| 工作 | 会议/年 | 讲的轴 | 和本项目的关系 |
|---|---|---|---|
| **MatRaptor** | [MICRO 2020](https://www.csl.cornell.edu/~zhiruz/pdfs/matraptor-micro2020.pdf) | ESC 数据流(row-wise product) | **我们的 ESC = 它的 Expand-Select-Reduce**。可对照 row-wise product 数据流优化 |
| **Spada** | [ASPLOS 2023](https://people.iiis.tsinghua.edu.cn/~gaomy/pubs/spada.asplos23.pdf) / [sim](https://github.com/tsinghua-ideal/spada-sim) | **按稀疏模式自适应选公式** | 形式化了我们实测的"Gustavson 赢小矩阵、cuSPARSE 赢大矩阵"——可指导公式选择策略 |
| **InnerSP** | [PACT 2021](https://jaehyuk-huh.github.io/papers/pact2021_innersp.pdf) | row-wise **inner product** | 对应我们的 inner 法;讲怎么让 inner 别那么慢 |
| **GAMMA** | ISCA 2020 | Gustavson-based | 我们 gust 的算法原型 |
| **OuterSPACE** | ISCA 2018 | outer-product(输入/输出复用) | 对应我们的 outer 法 |
| **SpArch** | ISCA 2019 | 专用 **merging unit**(层次化) | **唯一真碰 merge 的一篇**;给你的 sort 瓶颈一个硬件天花板参考 |

> B 类对本项目**不是新优化点**,更像旁证:我们的 4 法对比已经覆盖了他们的公式之争。Spada 的"自适应选公式"思想值得偷。

---

## C. 最近(2024–2025)

| 工作 | 会议/年 | 备注 |
|---|---|---|
| **RASSM** | [ASPLOS 2025](https://www.asplos-conference.org/asplos2025/program.html) | residue-based + adaptive tiling,新切分思路 |
| **MAGNUS** | ICS 2025 | 给 SpGEMM 生成 data locality(访存局部性) |
| **Swift** | HPCA 2026 | 注意是 **SpMM**(稀疏×稠密),非 SpGEMM;以后碰 GNN 再看 |

---

## D. 瓶颈 #1(d2h 传输)——无顶会论文

**没有顶会工作专门解决这个**:它不是算法问题,是 CUDA API 工程问题(`cudaMallocHost` 反复锁页 + PCIe 下载 C)。

解法(已落在本仓库):
- `cudaMallocAsync` 流式内存池 / pinned arena(锁页一次复用);
- `KEEP_ON_DEVICE`(benchmark 路径跳过 d2h);
- 细节见 `suitesparse_crawl/transfer_optimization.md`。

这条路无可借鉴论文,直接做工程。

---

## E. 阅读路由(按目的)

| 想干什么 | 读哪篇 |
|---|---|
| 决定 sort 换成 hash 还是 merge、怎么放寄存器 | **Register-Aware**(NPC'18) |
| 找现成的 hash accumulator 实现 | **spECK** |
| 找免 sort 的归并替代 | **Segmented Merge** |
| 系统理解公式选择 / 偷自适应策略 | **Spada**(ASPLOS'23)+ **MatRaptor**(MICRO'20) |
| 看 merge 的硬件天花板 | **SpArch**(ISCA'19) |
| SpGEMM 全景综述 | [A Systematic Survey of SpGEMM](https://dl.acm.org/doi/10.1145/3571157) |

---

## F. 对本项目的落地优先级(与 `profiling_analysis.md` §4 对齐)

1. **#1 d2h**:`KEEP_ON_DEVICE` / pinned 池——工程,无论文,最大收益(~60%)。
2. **#2 sort**:按 Register-Aware 的结论,把 ESC 的 `sort_by_key` 换成 hash 或分段归并——大矩阵 ~30–40% 收益,手写法翻盘 cuSPARSE 的关键。
3. 公式选择:借 Spada 思想做 per-matrix 自适应(可选,锦上添花)。

---

## 来源

- [MatRaptor (MICRO 2020)](https://www.csl.cornell.edu/~zhiruz/pdfs/matraptor-micro2020.pdf)
- [Spada (ASPLOS 2023)](https://people.iiis.tsinghua.edu.cn/~gaomy/pubs/spada.asplos23.pdf) · [spada-sim](https://github.com/tsinghua-ideal/spada-sim)
- [spECK (hash accumulator)](https://www.tugraz.at/fileadmin/user_upload/Institute/ICG/Downloads/team_steinberger/Publications/spECK.pdf)
- [Register-Aware SpGEMM (sort/merge/hash 对比)](https://pacman.cs.tsinghua.edu.cn/npc2018/papers/register-aware.pdf)
- [Segmented Merge](https://www.ssslab.cn/assets/papers/2021-ji-segmerge.pdf)
- [Optimizing General SpGEMM on GPU](https://dl.acm.org/doi/10.1145/3774654)
- [InnerSP (PACT 2021)](https://jaehyuk-huh.github.io/papers/pact2021_innersp.pdf)
- [NSparse library](https://github.com/EBD-CREST/nsparse)
- [A Systematic Survey of SpGEMM](https://dl.acm.org/doi/10.1145/3571157)
- [ASPLOS 2025 program (RASSM)](https://www.asplos-conference.org/asplos2025/program.html)

> 注:spECK / GAMMA / OuterSPACE / SpArch 的**确切会议年份**个别来自检索摘要,落地引用前建议核对原文。算法结论不依赖具体年份。
