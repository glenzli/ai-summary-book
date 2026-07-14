# 资料源与外部输入

本文件只列正文实际使用的外部输入，并说明用途。机构网页口径核对日期为 **2026-07-14**；动态文档在引用时仍应记录访问版本。

## 浮点与数值分析

- IEEE, [*IEEE Standard for Floating-Point Arithmetic, IEEE 754-2019*](https://standards.ieee.org/ieee/754/6210/), DOI `10.1109/IEEESTD.2019.8766229`。正文 3.A 使用 binary64 格式、`roundTiesToEven`、次正规数、无穷、NaN 与异常语义；本书不重证完整标准。
- Nicholas J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002, DOI `10.1137/1.9780898718027`。第 3 章标准舍入模型、$\gamma_k$ 记号和求和误差分析的专著定位；本书重证所用有限版本。
- David Goldberg, [*What Every Computer Scientist Should Know About Floating-Point Arithmetic*](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html), *ACM Computing Surveys* 23(1), 1991, DOI `10.1145/103162.103163`。浮点格式、舍入和异常的经典背景。

## 并发与分布式系统

- Leslie Lamport, [*Time, Clocks, and the Ordering of Events in a Distributed System*](https://www.microsoft.com/en-us/research/publication/time-clocks-ordering-events-distributed-system/), *Communications of the ACM* 21(7), 1978, DOI `10.1145/359545.359563`。第 4 章 happens-before 与分布式事件偏序的外部来源。
- Maurice P. Herlihy and Jeannette M. Wing, [*Linearizability: A Correctness Condition for Concurrent Objects*](https://www.cs.cmu.edu/~wing/publications/HerlihyWing90.pdf), *ACM TOPLAS* 12(3), 1990, DOI `10.1145/78969.78972`。外部输入 4.A；本书不重证线性化局部性等完整理论。
- NVIDIA, [*CUDA C++ Best Practices Guide*](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)；GPU 并行浮点、设备执行与优化边界的版本化工程资料。

## RNG 与机器学习框架

- PyTorch, [*Reproducibility Notes*](https://docs.pytorch.org/docs/stable/notes/randomness.html)；随机源、跨版本/平台限制与确定性算法的官方边界。
- PyTorch, [`torch.use_deterministic_algorithms`](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html)；第 4、5 章关于确定性开关覆盖、替代实现和报错行为的来源。
- TensorFlow, [*Random number generation*](https://www.tensorflow.org/guide/random_numbers)；全局/操作级 seed 与状态语义。
- NumPy, [*Random sampling documentation*](https://numpy.org/doc/stable/reference/random/index.html)；Generator、BitGenerator、SeedSequence 与状态接口。

## 内容身份、构建与 provenance

- W3C, [*PROV-DM: The PROV Data Model*](https://www.w3.org/TR/prov-dm/) and [*PROV-O: The PROV Ontology*](https://www.w3.org/TR/prov-o/), W3C Recommendations, 2013。外部输入 7.A 的 Entity、Activity、Agent，以及使用、生成与 Activity--Agent 关联关系。
- Open Containers Initiative, [*OCI Image Format Specification*](https://github.com/opencontainers/image-spec)；镜像清单、层和内容摘要的规范边界。
- Chris Lamb and Stefano Zacchiroli, [*Reproducible Builds: Increasing the Integrity of Software Supply Chains*](https://arxiv.org/abs/2104.06020), *IEEE Software* 39(2), 2022, DOI `10.1109/MS.2021.3073045`；独立逐位构建与供应链证据边界。
- [Reproducible Builds project documentation](https://reproducible-builds.org/docs/)；时间戳、路径、文件顺序等构建非确定输入的工程资料。
- NixOS, [*Nix Reference Manual*](https://nixos.org/manual/nix/stable/)；声明式构建、存储路径和 impurity 边界。
- Software Heritage, [*Persistent Identifiers*](https://www.softwareheritage.org/save-and-reference-research-software/)；软件归档标识。

## 机构复现术语

- ACM, [*Artifact Review and Badging -- Current*](https://www.acm.org/publications/policies/artifact-review-and-badging-current)，当前页面与 v1.1 徽章说明。外部输入 8.A：Repeatability（同团队/同设置）、Reproducibility（不同团队/同设置、使用作者制品）、Replicability（不同团队/不同设置、独立制品），以及 Results Reproduced/Replicated 的准确差别。页面亦记录 ACM/NISO 协调后的术语交换。
- National Academies of Sciences, Engineering, and Medicine, [*Reproducibility and Replicability in Science*](https://nap.nationalacademies.org/catalog/25303/reproducibility-and-replicability-in-science), 2019, DOI `10.17226/25303`。外部输入 8.B：同输入数据/计算步骤/方法/代码/分析条件的 computational reproducibility，与新数据研究同一问题的 replicability。
- Joint Committee for Guides in Metrology, [*International Vocabulary of Metrology (VIM), 3rd ed., JCGM 200:2012*](https://jcgm.bipm.org/vim/en/index.html)。外部输入 8.C：2.20--2.21 repeatability condition/measurement repeatability 与 2.24--2.25 reproducibility condition/measurement reproducibility。

## 统计推断

- Donald J. Schuirmann, [*A Comparison of the Two One-Sided Tests Procedure and the Power Approach for Assessing the Equivalence of Average Bioavailability*](https://pubmed.ncbi.nlm.nih.gov/3450848/), *Journal of Pharmacokinetics and Biopharmaceutics* 15(6), 1987, DOI `10.1007/BF01068419`。第 9 章 TOST 的原始方法来源；正文另证连续对称位置枢轴下的 $1-2\alpha$ 区间对应。
- Sture Holm, *A Simple Sequentially Rejective Multiple Test Procedure*, *Scandinavian Journal of Statistics* 6(2), 1979, DOI `10.2307/4615733`。定理 9.5 的原始来源；正文给出完整有限证明。
- Yoav Benjamini and Yosef Hochberg, [*Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing*](https://cris.tau.ac.il/en/publications/controlling-the-false-discovery-rate-a-practical-and-powerful-app-2/), *JRSS B* 57(1), 1995, DOI `10.1111/j.2517-6161.1995.tb02031.x`。外部输入 9.B 的独立情形；本书不重证其证明，也不把结论无条件推广到任意依赖。
- E. L. Lehmann and Joseph P. Romano, *Testing Statistical Hypotheses*, 3rd ed., Springer, 2005, DOI `10.1007/0-387-27605-X`。复合假设、检验大小、枢轴与多重检验的标准专著定位；正文只在列明模型下使用这些结论。
- George Casella and Roger L. Berger, *Statistical Inference*, 2nd ed., Duxbury, 2002, Chapter 5。外部输入 9.A 的正态样本均值、样本方差与 Student-$t$ 枢轴定位；本书直接陈述所用的一样本/配对差版本。

## 可计算性

- Alan M. Turing, [*On Computable Numbers, with an Application to the Entscheidungsproblem*](https://www.cs.virginia.edu/~robins/Turing_Paper_1936.pdf), *Proceedings of the London Mathematical Society* s2-42(1), 1936, DOI `10.1112/plms/s2-42.1.230`。外部输入 11.A 的停机不可判定边界；正文给出全域程序等价不可判定的归约。

## 外部边界总表

本书不重证 IEEE 754 完整规范、具体语言内存模型、线性化完整理论、框架跨版本行为、W3C PROV 全部约束、Student-$t$/渐近统计的完整分布理论、BH 在各依赖类下的推广或停机问题原始不可判定证明。正文只在明确假设下使用列出的有限版本。
