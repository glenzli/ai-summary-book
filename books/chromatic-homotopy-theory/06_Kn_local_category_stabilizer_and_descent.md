# 第六章：$K(n)$-局部范畴、Morava stabilizer group 与 descent

第五章把所有高度排成 tower；若把视野固定在第 $n$ 层，新的问题是怎样描述整个 $K(n)$-局部范畴。Morava $E_n$ 提供可计算的完备系数环，extended stabilizer group $\mathbb G_n$ 编码其连续对称，而 homotopy fixed points 把带群作用的局部坐标重新下降为谱。由此，$K(n)$-local sphere 的同伦群问题转化为连续群上同调与 descent spectral sequence。本章使用第三章的 $E_n$、$K(n)$ 及其变形解释；Goerss--Hopkins--Miller、Devinatz--Hopkins 连续同伦固定点理论和 Morava module descent 均以精确外部输入出现。

## 6.1 $K(n)$-局部范畴

**定义 6.1.** $K(n)$-局部谱范畴记为
$$
\mathbf{Sp}_{K(n)}=\{X\in\mathbf{Sp}_{(p)}\mid X\simeq L_{K(n)}X\}.
$$
它是 $\mathbf{Sp}_{(p)}$ 的反射全子范畴，反射函子为 $L_{K(n)}$。

**命题 6.2.** 若 $X$ 是 $K(n)$-local，且 $Y\simeq X$，则 $Y$ 是 $K(n)$-local。

**证明.** $Y\simeq X\simeq L_{K(n)}X\simeq L_{K(n)}Y$，其中最后一步由函子性和等价保持性得到。证毕。

**定义 6.3.** $K(n)$-local sphere 定义为
$$
\mathbb S_{K(n)}=L_{K(n)}\mathbb S.
$$

**警告 6.4.** $\mathbb S_{K(n)}$ 不是普通球谱，也不是 $E_n$。它是高度 $n$ 局部世界的单位对象，计算极其困难。

## 6.2 Morava stabilizer group

**定义 6.5.** 设 $\Gamma$ 是 $\mathbb F_{p^n}$ 上高度 $n$ 的形式群。Morava stabilizer group $\mathbb S_n$ 是 $\Gamma$ 的 automorphism group。extended Morava stabilizer group 为
$$
\mathbb G_n=\mathbb S_n\rtimes \operatorname{Gal}(\mathbb F_{p^n}/\mathbb F_p).
$$

**外部输入 6.6.** $\mathbb G_n$ 作为 profinite group 连续作用在 $E_n$ 上，且该作用提升到 $\mathbb E_\infty$-ring spectra 层面。

**警告 6.7.** $\mathbb G_n$ 的作用不是普通离散群作用。homotopy fixed points、连续群上同调和 descent spectral sequence 必须保留 profinite topology。

## 6.3 Devinatz-Hopkins descent

**外部输入定理 6.8.** 对合适的 $X$，特别是有限谱 $X$，存在等价
$$
L_{K(n)}X\simeq (E_n\otimes X)^{h\mathbb G_n},
$$
其中右侧为连续 $\mathbb G_n$-作用下的 homotopy fixed points。一般谱版本需要额外模型和完备性条件。

**推论 6.9.** $K(n)$-local sphere 满足
$$
\mathbb S_{K(n)}\simeq E_n^{h\mathbb G_n}.
$$

**证明.** 在定理 6.8 中取 $X=\mathbb S$，得到
$$
L_{K(n)}\mathbb S\simeq (E_n\otimes\mathbb S)^{h\mathbb G_n}\simeq E_n^{h\mathbb G_n}.
$$
证毕。

**外部输入 6.10 (Morava descent spectral sequence).** 在适当连续性和收敛条件下，有谱序列
$$
H_c^s(\mathbb G_n;(E_n)_tX)\Longrightarrow \pi_{t-s}L_{K(n)}X.
$$

**使用说明.** 这里 $H_c^s$ 是连续群上同调。$(E_n)_tX$ 的拓扑和 $\mathbb G_n$-module 结构必须按具体模型说明。不能把它替换成离散群上同调。

## 6.4 低高度检查

**例 6.11 (高度 0).** $K(0)=H\mathbb Q$，$L_{K(0)}$ 是有理化。此时 stabilizer group 语言退化，计算回到有理稳定同伦论。

**例 6.12 (高度 1).** 高度 $1$ 的 Morava E-theory 与 $p$-adic K-theory 密切相关，$\mathbb G_1$ 与 $\mathbb Z_p^\times$ 相关。$K(1)$-local sphere 可通过 $p$-adic Adams operations 的 homotopy fixed points 描述。完整分裂和 $J$-homomorphism 比较需要单独定位。

**警告 6.13.** 高度 $1$ 的直观图像不能直接推广到高度 $n\ge2$。$\mathbb G_n$ 的 cohomological dimension、torsion、finite subgroups 和 action 都更复杂。

## 6.5 Picard、dualizing 和未来章节接口

**定义 6.14.** $\mathbf{Sp}_{K(n)}$ 的 Picard group 是 invertible $K(n)$-local spectra 的同构类群，记作
$$
\operatorname{Pic}(\mathbf{Sp}_{K(n)}).
$$

**边界 6.15.** Gross-Hopkins duality、Brown-Comenetz dual、$K(n)$-local Picard group 和 exotic elements 是后续章节主题。当前章节只建立 descent 接口，不把这些结果用于基础证明。

## 6.6 连续同伦固定点的计算格式

**定义 6.16.** 对 profinite group $G$ 连续作用的谱 $Y$，其连续 homotopy fixed points 可由离散商的 tower 或连续 cochain 模型构造，记为
$$
Y^{hG}.
$$
具体模型依赖 $Y$ 的拓扑化 action，本书不把它等同于 naive limit $\lim_{BG}Y$。

**命题 6.17.** 若 $G$ 是有限离散群，则连续 homotopy fixed points 退化为通常的 homotopy fixed points。

**证明.** 有限离散群的 profinite topology 是离散且紧的，连续 $G$-action 与普通 $G$-action 相同；连续 cochains 与普通 group cochains 相同。故两种构造一致。证毕。

**警告 6.18.** 对 $\mathbb G_n$，命题 6.17 不适用。$\mathbb G_n$ 是无限 profinite group，连续 cochains 是必要输入。

## 6.7 $K(n)$-local sphere 的计算策略

**步骤 6.19.** 计算 $\pi_*\mathbb S_{K(n)}$ 的标准策略是：

1. 用 Goerss-Hopkins-Miller 得到 $E_n$ 的 $\mathbb E_\infty$ 结构与 $\mathbb G_n$ action；
2. 用 Devinatz-Hopkins 得到
   $$
   \mathbb S_{K(n)}\simeq E_n^{h\mathbb G_n};
   $$
3. 建立连续群上同调谱序列
   $$
   H_c^s(\mathbb G_n;(E_n)_t)\Rightarrow \pi_{t-s}\mathbb S_{K(n)};
   $$
4. 计算 $H_c^s$，处理 differentials；
5. 解决 hidden extensions。

**警告 6.20.** 第 4 步通常是主要困难。即使 $E_2$ 页已知，differentials 和 extensions 仍可能改变最终同伦群。

## 6.8 从局部坐标到同伦下降

$K(n)$-local category 是单一高度的局部世界。Morava E-theory $E_n$ 带有 extended stabilizer group 的连续作用，$K(n)$-local sphere 可表示为 $E_n^{h\mathbb G_n}$。计算由连续群上同调谱序列控制。所有这些深层结构依赖 Goerss-Hopkins-Miller 和 Devinatz-Hopkins 外部输入。

## 练习

**练习 6.1.** 证明若 $X$ 是 $K(n)$-local，则 $F(Y,X)$ 对任意 $Y$ 也是 $K(n)$-local 是否成立？若要证明，写出需要的 closed symmetric monoidal localization 条件。

**练习 6.2.** 对 $n=1$，查阅 $\mathbb Z_p^\times$ 的结构，并说明其有限 torsion 部分如何依赖 $p=2$ 与奇素数。

**练习 6.3.** 在谱序列 6.10 中解释 $t-s$ 总次数从何而来。
