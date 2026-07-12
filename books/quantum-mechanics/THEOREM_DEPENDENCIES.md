# 外部输入定理依赖图

## 外部输入清单

| 标签 | 定理 | 用途 |
|---|---|---|
| QM-EXT-1 | Hilbert 空间上的谱定理 | 第 3、4、5 章定义一般可观测量和函数演算 |
| QM-EXT-2 | Stone 定理 | 第 5 章从强连续酉群得到自伴 Hamiltonian |
| QM-EXT-3 | Kato-Rellich 定理 | 第 6 章在实值乘法势的相对界假设下判断 $-\Delta+V$ 自伴 |
| QM-EXT-4 | Stone-von Neumann 定理 | 第 11 章正则对易关系的唯一性 |
| QM-EXT-5 | Wigner 定理 | 第 10 章对称性由酉或反酉算子实现 |
| QM-EXT-6 | Stinespring/Kraus 表示 | 第 18、19 章完全正映射和量子信道 |
| QM-EXT-7 | 绝热定理 | 第 16 章慢变 Hamiltonian 的近似演化 |
| QM-EXT-8 | 短程势波算子与渐近完备性 | 第 14 章以 $P_{\mathrm{ac}}(H_0)$ 定义波算子并识别值域 $\mathcal H_{\mathrm{ac}}(H)$ |
| QM-EXT-9 | Lie--Trotter 乘积公式 | 第 20 章路径积分的离散化入口 |
| QM-EXT-10 | 球谐函数完备性与 Coulomb 谱理论 | 第 22 章中心势分离变量和氢原子能级 |
| QM-EXT-11 | 磁 Schrodinger 算子自伴性 | 第 23 章最小耦合 Hamiltonian 的严格定义 |
| QM-EXT-12 | 有限维 $\mathfrak{su}(2)$ 表示分解 | 第 26 章角动量张量积与 Clebsch-Gordan 系数 |
| QM-EXT-13 | Wigner-Eckart 定理 | 第 26 章球张量矩阵元和选择定则 |
| QM-EXT-14 | Sturm-Liouville/Fourier-Hermite 完备性 | 第 6、7 章方势阱和谐振子本征函数完备性；第 23 章 Landau 纤维的完整谱 |
| QM-EXT-15 | WKB 转折点连接公式 | 第 13 章半经典量子化条件 |
| QM-EXT-16 | 光学定理与 partial wave 展开 | 第 14 章散射截面和中心势相移公式 |
| QM-EXT-17 | Friedrichs 扩张和闭二次型表示 | 附录 A 与半有界 Hamiltonian 自伴实现 |
| QM-EXT-18 | Lindblad 生成元定理 | 第 17 章 Markov 开放系统边界 |
| QM-EXT-19 | Uhlmann 定理与混合态纯化唯一性 | 第 17、19 章纯化和保真度边界 |
| QM-EXT-20 | Kato 解析扰动理论 | 第 12 章有限维扰动公式的无限维边界 |

## 依赖边界

- 第 1--2 章主要内部闭合，有限维谱分解给出证明。
- 第 3--5 章依赖 QM-EXT-1 与 QM-EXT-2。
- 第 6 章使用 QM-EXT-3 处理满足相对界条件的实值乘法势；第 6--7 章使用 QM-EXT-14 说明本征函数完备性，第 7 章再由 Hermite 加权移位在书内推出算子定义域与共同核心。
- 第 10--11 章分别依赖 QM-EXT-5 与 QM-EXT-4。
- 第 12--14、16、18、20 章各自依赖解析扰动、WKB 连接、散射、绝热、完全正映射和 Trotter 相关外部输入。
- 第 22--23 章依赖球谐完备性、Coulomb Hamiltonian 谱理论、磁 Schrodinger 算子的自伴性和 Landau 纤维中的 Hermite 完备性。
- 第 24--25 章主要内部闭合；连续谱黄金规则的严格极限依赖谱测度和散射理论。
- 第 26 章依赖 $\mathfrak{su}(2)$ 有限维表示分解和 Wigner-Eckart 定理。
- 第 17、19 章内部证明有限维密度算子、纯化、迹距离等基本事实；Lindblad 与 Uhlmann 型结构定理作为外部输入边界。
