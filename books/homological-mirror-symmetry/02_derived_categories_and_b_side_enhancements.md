# 第二章：导出范畴、完美复形与 B-side 增强

## 本章目标

本章建立 HMS 的 B-side 语言：导出范畴、完美复形、coherent sheaves、dg enhancement、Fourier-Mukai transforms 和 matrix factorizations。重点不是复述代数几何全貌，而是把 B-side 写成可与 A-side Fukaya 范畴比较的增强范畴。

## 依赖前置知识

需要熟悉复形、同调、abelian category、局部自由层、张量积、拉回和推出。一般 derived algebraic geometry 不作为本章前置。

## 2.1 从复形到导出范畴

**定义 2.1.** 设 $\mathcal A$ 是 $k$-线性 abelian category。复形范畴 $\operatorname{Ch}(\mathcal A)$ 的对象是 cochain complexes
$$
\cdots\to C^{i-1}\xrightarrow{d^{i-1}}C^i\xrightarrow{d^i}C^{i+1}\to\cdots,
\qquad d^{i+1}d^i=0.
$$
链映射为保持次数并与微分交换的 morphisms。

**定义 2.2.** 链映射 $f:C\to D$ 称为 quasi-isomorphism，若对所有 $i$，诱导映射
$$
H^i(f):H^i(C)\to H^i(D)
$$
为同构。

**定义 2.3.** $\mathcal A$ 的导出范畴 $\mathrm D(\mathcal A)$ 是同伦范畴 $K(\mathcal A)$ 对所有 quasi-isomorphisms 的 Verdier localization：
$$
\mathrm D(\mathcal A)=K(\mathcal A)[\operatorname{Qis}^{-1}].
$$
若只允许 bounded complexes，则写作 $\mathrm D^b(\mathcal A)$。

**警告 2.4.** $\mathrm D(\mathcal A)$ 是三角范畴。HMS 中只给出三角范畴 $\mathrm D^b\operatorname{Coh}(X)$ 仍不够，因为 Hochschild invariants、Fourier-Mukai kernels 和 Morita theory 需要 dg 或 stable $\infty$ enhancement。

## 2.2 完美复形

**定义 2.5.** 设 $X$ 是 $k$-scheme。一个 $\mathcal O_X$-module 复形 $E$ 称为 perfect，若 Zariski 局部上 $E$ quasi-isomorphic 于有限长的有限秩局部自由 $\mathcal O_X$-module 复形。perfect complexes 的增强范畴记为 $\operatorname{Perf}(X)$。

**定义 2.6.** 若 $X$ 是 locally noetherian scheme，$\operatorname{Coh}(X)$ 表示 coherent sheaves 构成的 abelian category，$\mathrm D^b\operatorname{Coh}(X)$ 表示其 bounded derived category。

**命题 2.7.** 若 $X$ 是光滑 noetherian $k$-scheme，则每个 bounded coherent complex 是 perfect。因此在同伦范畴层面有
$$
H^0\operatorname{Perf}(X)\simeq \mathrm D^b\operatorname{Coh}(X).
$$

**证明草图.** 光滑 noetherian 假设给出局部环的有限全局维数。对任意 coherent sheaf $\mathcal F$ 和点 $x\in X$，存在邻域 $U$，使 $\mathcal F|_U$ 有有限长有限秩局部自由分解。对 bounded coherent complex，逐项取这样的局部分解并用截断拼接，得到局部有限长局部自由模型。因此该复形为 perfect。反向包含来自 perfect complex 的上同调 sheaves 仍 coherent 且 bounded。证毕。

**警告 2.8.** 若 $X$ 奇异，$\operatorname{Perf}(X)$ 与 $\mathrm D^b\operatorname{Coh}(X)$ 一般不同。奇异性正由 quotient
$$
\mathrm D_{\mathrm{sg}}(X)=\mathrm D^b\operatorname{Coh}(X)/\operatorname{Perf}(X)
$$
测量，这也是 Landau-Ginzburg B-side 和 matrix factorization 进入 HMS 的原因之一。

## 2.3 dg enhancement

**定义 2.9.** 三角范畴 $\mathcal T$ 的 dg enhancement 是一个 pretriangulated dg category $\mathcal C$ 和一个三角等价
$$
H^0(\mathcal C)\xrightarrow{\sim}\mathcal T.
$$

**例 2.10.** 对 scheme $X$，可以用 h-injective complexes of quasi-coherent sheaves、h-projective complexes，或 perfect complexes 的 dg model 给出 $\mathrm D_{\mathrm{qc}}(X)$ 与 $\operatorname{Perf}(X)$ 的 enhancement。具体模型的选择影响链级表示，但不应影响 Morita 型 HMS 命题。

**外部输入定理 2.11（enhancement 与 Morita 理论）.** 在适当假设下，$\operatorname{Perf}(X)$ 的 dg enhancement 是 B-side 的自然 Morita 对象；Fourier-Mukai kernels 与 dg functors 之间存在强联系。  
来源：Keller 的 dg categories，Huybrechts 的 Fourier-Mukai transforms，以及 Bondal-Orlov/Orlov 相关结果。后续 theorem locator 需要逐条定位。

## 2.4 Fourier-Mukai transforms

**定义 2.12.** 设 $X,Y$ 是光滑适当 $k$-schemes，$p_X,p_Y$ 为 $X\times Y$ 到两因子的投影。对 kernel
$$
K\in\operatorname{Perf}(X\times Y),
$$
Fourier-Mukai transform 定义为
$$
\Phi_K:\operatorname{Perf}(X)\to\operatorname{Perf}(Y),\qquad
\Phi_K(E)=\mathbf R p_{Y*}(p_X^\ast E\otimes^{\mathbf L}K).
$$

**命题 2.13.** 对角线结构层 $\mathcal O_\Delta\in\operatorname{Perf}(X\times X)$ 给出恒等 Fourier-Mukai transform。

**证明.** 令 $\Delta:X\to X\times X$ 为对角嵌入。由定义
$$
\Phi_{\mathcal O_\Delta}(E)=\mathbf R p_{2*}(p_1^\ast E\otimes^{\mathbf L}\Delta_\ast\mathcal O_X).
$$
投影公式给出
$$
p_1^\ast E\otimes^{\mathbf L}\Delta_\ast\mathcal O_X
\simeq \Delta_\ast(\Delta^\ast p_1^\ast E)
\simeq \Delta_\ast E.
$$
因为 $p_2\circ\Delta=\operatorname{id}_X$，所以 $\mathbf R p_{2*}\Delta_\ast E\simeq E$。证毕。

**外部输入定理 2.14（Fourier-Mukai 表示性）.** 对光滑适当 varieties，在适当假设下，许多 exact fully faithful functors between bounded derived categories of coherent sheaves 由 Fourier-Mukai kernels 表示。  
来源：Orlov 表示性定理及后续扩展；本书暂不内部证明。

**解释 2.15.** Fourier-Mukai kernel 是 B-side 等价的具体候选函子。HMS 中若能把 A-side 生成对象的 endomorphism algebra 与 B-side 某个生成对象的 endomorphism algebra 识别，就常能得到 Morita 层面的等价；若还能构造 kernel，则得到更几何的 B-side 解释。

## 2.5 Landau-Ginzburg B-side 与 matrix factorizations

**定义 2.16.** 设 $X$ 是 $k$-scheme，$W:X\to\mathbb A^1$ 是正则函数。一个 matrix factorization 是一对 $\mathbb Z/2$-分次 locally free sheaves $E_0,E_1$ 和 morphisms
$$
d_0:E_0\to E_1,\qquad d_1:E_1\to E_0
$$
使得
$$
d_1d_0=W\cdot\operatorname{id}_{E_0},\qquad
d_0d_1=W\cdot\operatorname{id}_{E_1}.
$$
它们组成的 dg 或 $\mathbb Z/2$-graded enhancement 记为 $\operatorname{MF}(X,W)$。

**例 2.17.** 若 $W=0$，matrix factorization 退化为 $\mathbb Z/2$-分次复形。若 $W$ 有孤立临界点，$\operatorname{MF}(X,W)$ 与奇点范畴有深刻关系；此关系是 Landau-Ginzburg mirror symmetry 的 B-side 基础之一。

**外部输入定理 2.18（Orlov 型关系）.** 在合适的正则性和适当性假设下，matrix factorizations 与 hypersurface 奇点的 singularity category 相关。  
来源：Orlov 的 Landau-Ginzburg/singularity category 理论；后续 theorem locator 需定位精确版本。

## 2.6 B-side 的 HMS 检查项

一个 B-side 数据包必须至少给出：

1. 几何对象 $X$ 或 $(X,W)$；
2. 采用的增强范畴：$\operatorname{Perf}(X)$、$\mathrm D^b\operatorname{Coh}(X)$ 的 enhancement、$\operatorname{MF}(X,W)$ 或 singularity category；
3. 是否光滑、适当、奇异、栈化或非紧；
4. 生成对象或 tilting object；
5. 候选 Fourier-Mukai kernel 或 endomorphism algebra；
6. 与 A-side 比较的系数、分次和 grading 数据。

## 本章小结

B-side 的主体不是孤立的三角范畴，而是可增强的导出几何对象。光滑情况下 $\operatorname{Perf}(X)$ 与 $\mathrm D^b\operatorname{Coh}(X)$ 在同伦范畴层面一致；奇异或 Landau-Ginzburg 情况下 matrix factorizations 和 singularity categories 成为核心。Fourier-Mukai transforms 提供了 B-side 等价的几何模型。

## 练习

**练习 2.1.** 证明 quasi-isomorphism 在导出范畴中变成同构。

**练习 2.2.** 对仿射 scheme $X=\operatorname{Spec}A$，说明 perfect complexes 与 perfect dg $A$-modules 的关系。

**练习 2.3.** 逐步验证命题 2.13 中使用的投影公式和对角线恒等式。

**练习 2.4.** 给出一个奇异 scheme 的例子，并解释为什么 $\operatorname{Perf}(X)$ 与 $\mathrm D^b\operatorname{Coh}(X)$ 不应混同。
