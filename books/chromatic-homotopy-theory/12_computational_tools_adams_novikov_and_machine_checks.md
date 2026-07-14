# 第十二章：计算工具、Adams--Novikov 与谱序列核验

前十一章说明了色层对象为何存在，真正计算稳定 stems 时却要在多种谱序列之间传递：Adams--Novikov 从 $BP_*BP$-comodule 开始，change of rings 把局部代数转向 stabilizer group，homotopy fixed point 与 Tate 谱序列再处理 descent 和群作用。任何一次转译都可能引入 differential、extension 或收敛问题，所以一张 $E_2$ 页面从来不等于最终同伦群。本章以一个可复核的计算流程组织这些工具，说明输入对象、分次、收敛口径和交叉校验分别放在哪里。第二章的 $BP$、第三章的 $E_n$、第六章的 descent 与附录 B/C 的谱序列和 Hopf algebroid 约定贯穿全章。

## 12.1 Adams-Novikov spectral sequence

**外部输入 12.1.** 对合适的 $p$-局部谱 $X$，存在 $BP$-based Adams-Novikov spectral sequence
$$
E_2^{s,t}\cong \operatorname{Ext}^{s,t}_{BP_*BP}(BP_*,BP_*X)
\Longrightarrow \pi_{t-s}X^\wedge_p
$$
或其有限谱/局部化变体。收敛条件必须随 $X$ 声明。

**警告 12.2.** $\operatorname{Ext}_{BP_*BP}$ 是 Hopf algebroid comodules 范畴中的 Ext。把它当作 $BP_*$-modules 范畴中的 Ext 会得到错误结果。

**定义 12.3.** Adams-Novikov filtration 是 abutment $\pi_*X^\wedge_p$ 上由谱序列给出的 filtration。元素的 filtration 表示其在 Adams-Novikov resolution 中出现的层级。

**警告 12.4.** $E_\infty$ 页面只给出 associated graded。确定同伦群还需要解决 hidden additive extensions 和 hidden multiplicative extensions。

## 12.2 Chromatic spectral sequence

**外部输入 12.5.** Chromatic spectral sequence 将 $BP_*BP$-comodule 的 Ext 按 invariant prime ideals
$$
I_n=(p,v_1,\ldots,v_{n-1})
$$
和局部上同调分层。它把 Adams-Novikov $E_2$ 页分解为高度层贡献。

**定义 12.6.** 记
$$
I_n=(p,v_1,\ldots,v_{n-1})\subset BP_*.
$$
这个理想在 $BP_*BP$ Hopf algebroid 意义下 invariant。

**警告 12.7.** $I_n$ 的 invariant 性不是普通环论陈述，而是左右 unit 和 coaction 相容的 Hopf algebroid 陈述。完整验证见附录 C。

## 12.3 Morava change of rings

**外部输入 12.8.** Morava change-of-rings theorem 将某些 localized/completed $BP_*BP$-comodule Ext 群与 Morava stabilizer group 的连续群上同调联系起来：
$$
\operatorname{Ext}_{\text{height }n}\quad \leadsto\quad H_c^*(\mathbb G_n; (E_n)_*X).
$$
精确形式依赖 completed Hopf algebroid、height $n$ localization 和 $X$ 的有限性。

**使用限制 12.9.** 不能把 change-of-rings 写成普通环扩张下的 Ext 换环。这里涉及 Hopf algebroid、formal group moduli 的局部化和 profinite stabilizer group。

## 12.4 Homotopy fixed point spectral sequence

**外部输入 12.10.** 若 $G$ 是 profinite group 连续作用在谱 $Y$ 上，在合适条件下有 homotopy fixed point spectral sequence
$$
H_c^s(G;\pi_tY)\Longrightarrow \pi_{t-s}Y^{hG}.
$$

**例 12.11.** 取 $Y=E_n\otimes X$、$G=\mathbb G_n$，得到 Morava descent spectral sequence
$$
H_c^s(\mathbb G_n;(E_n)_tX)\Longrightarrow \pi_{t-s}L_{K(n)}X.
$$

**警告 12.12.** 如果 $G$ 是 profinite group，$H_c^s$ 是连续群上同调；如果 $G$ 是有限离散群，才退化为普通群上同调。这个区别影响 $E_2$ 页。

## 12.5 Tate spectral sequence

**定义 12.13.** 对有限群 $G$ 作用的谱 $Y$，Tate construction $Y^{tG}$ 是 norm map
$$
Y_{hG}\to Y^{hG}
$$
的 cofiber。

**外部输入 12.14.** Tate spectral sequence 在合适有界性条件下形如
$$
\widehat H^s(G;\pi_tY)\Longrightarrow \pi_{t-s}Y^{tG}.
$$

**说明 12.15.** Tate vanishing 是 redshift、cyclotomic spectra 和 higher semiadditivity 中反复出现的技术点，但每次 vanishing 都有独立假设。

## 12.6 可复现的计算记录

**数据约定 12.16.** 一次可复现的 chromatic 计算至少记录：

1. 素数 $p$；
2. 高度 $n$；
3. 谱或环谱对象；
4. 使用的谱序列；
5. $E_2$ 页来源；
6. differential 列表；
7. hidden extensions；
8. 收敛定理；
9. 与已知低 stems 或已知 $K(n)$-local 结果的交叉验证；
10. 若使用软件，记录输入数据、版本和可复现脚本。

**警告 12.17.** 机器计算可以降低表格错误率，但不能替代数学证明。软件输出必须绑定到明确的 chain complex、resolution 或 spectral sequence model。

## 12.7 从对象到稳定 stems

**方法 12.18.** 对谱 $X$ 进行 Adams--Novikov 型计算时，逻辑次序如下：

1. 选择素数 $p$ 和 completion；
2. 计算或引用 $BP_*X$ 作为 $BP_*BP$-comodule；
3. 选择 resolution 或 cobar complex；
4. 计算
   $$
   \operatorname{Ext}^{s,t}_{BP_*BP}(BP_*,BP_*X);
   $$
5. 建立 differential；
6. 得到 $E_\infty$ 页；
7. 解 hidden additive extensions；
8. 解 hidden multiplicative extensions；
9. 与已知 $K(n)$-local 或低高度信息交叉验证。

**命题 12.19.** 若第 7 步未完成，则不能从 $E_\infty$ 页唯一恢复 $\pi_*X$ 的加法群。

**证明.** $E_\infty$ 页给出的是 filtered group 的 associated graded。不同 filtered groups 可以有同一个 associated graded，例如 $\mathbb Z/4$ 与带两层 filtration 的 $\mathbb Z/2\oplus\mathbb Z/2$ 都可给出两个 $\mathbb Z/2$ graded pieces。因此 extension data 必需。证毕。

**例 12.19A（同一 associated graded 的两种提升）.** 对
$A=\mathbb Z/4$ 取过滤
$$
F^0A=A,\qquad F^1A=2A,\qquad F^2A=0;
$$
对 $B=\mathbb Z/2\oplus\mathbb Z/2$ 取
$$
F^0B=B,\qquad F^1B=0\oplus\mathbb Z/2,\qquad F^2B=0.
$$
两者的 $\operatorname{gr}^0$ 与 $\operatorname{gr}^1$ 都是
$\mathbb Z/2$，但 $A$ 含四阶元素而 $B$ 没有。因此即使所有 differential
已经确定，若不解这一加法 extension，谱序列仍不能区分目标群是 $A$ 还是 $B$。

## 12.8 与 chromatic tower 的交叉校验

**判据 12.20.** 若某个 Adams--Novikov 计算声称发现高度 $n$ 周期族，应检查：

1. 该族在 $K(n)_*$ 或 $E_n$-based descent 中是否可见；
2. 低高度 $K(m)$ 对 $m<n$ 是否消失或变成 torsion；
3. 是否存在 $v_n$ self-map 或 telescope 解释；
4. 是否与已知 chromatic spectral sequence filtration 相容。

**警告 12.21.** 命名一个族为 $v_n$-periodic 不等于证明它由某个 finite type $n$ spectrum 的 $v_n$ self-map 产生。

## 12.9 从谱序列页面到稳定 stems

Chromatic 计算的共同模式是：用 $BP$ 和 formal group moduli 组织全局信息，再用 Morava stabilizer group 和 descent 研究单一高度。Adams--Novikov、chromatic spectral sequence、change of rings 和 homotopy fixed point spectral sequence 是核心工具。一个完整计算必须交代每个谱序列的收敛、hidden extensions 和模型假设。

## 练习

**练习 12.1.** 解释为什么 Adams-Novikov 谱序列的 $E_2$ 页是 comodule Ext。

**练习 12.2.** 写出 $I_1$、$I_2$、$I_3$ 的定义，并说明 $I_1=(p)$。

**练习 12.3.** 对有限群 $C_p$，写出 Tate cohomology $\widehat H^*(C_p;M)$ 与 ordinary group cohomology 的关系。

**练习 12.4.** 设计一个稳定 stems 小表格的校验清单，至少包含 differential 和 hidden extension 两列。
