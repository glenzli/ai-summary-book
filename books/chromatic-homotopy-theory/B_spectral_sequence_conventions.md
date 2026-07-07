# 附录 B：谱序列、filtration 和收敛约定

## B.1 总约定

**约定 B.1.** 本书的 homological spectral sequence 默认写作
$$
E^r_{s,t}\Rightarrow \pi_{t-s}X
$$
或
$$
E_r^{s,t}\Rightarrow \pi_{t-s}X,
$$
具体上下标由来源决定。每次使用必须说明：

1. $E_2$ 页；
2. 微分次数；
3. abutment；
4. filtration 是 increasing 还是 decreasing；
5. 收敛类型。

**警告 B.2.** Adams-Novikov、homotopy fixed point、Tate、descent 和 chromatic spectral sequence 的指标约定不同。不得只写“由谱序列得到”而不说明总次数和收敛。

## B.2 强收敛和条件收敛

**定义 B.3.** 谱序列强收敛到带 filtration 的 graded group $G_*$，若 $E^\infty$ 与 $\operatorname{gr}G_*$ 识别，并且 filtration 完备、Hausdorff，且不存在隐藏的 $\lim^1$ 障碍。

**定义 B.4.** 条件收敛指存在自然候选 abutment 和 filtration，但需要额外证明完备性、Hausdorff 性或 derived limit 消失。

**警告 B.5.** 在 chromatic homotopy theory 中，条件收敛常是实际计算的风险点。尤其在连续群上同调和 homotopy fixed point spectral sequence 中，profinite topology 会影响收敛。

## B.3 Adams-Novikov 口径

**外部输入 B.6.** 对合适的谱 $X$，Adams-Novikov spectral sequence 具有形式
$$
\operatorname{Ext}^{s,t}_{MU_*MU}(MU_*,MU_*X)\Rightarrow \pi_{t-s}X^\wedge
$$
或 $BP$-based 版本
$$
\operatorname{Ext}^{s,t}_{BP_*BP}(BP_*,BP_*X)\Rightarrow \pi_{t-s}X^\wedge_p.
$$
精确 completion 和 convergence 条件必须按对象 $X$ 声明。

**警告 B.7.** $E_2$ 页的 Ext 群是 Hopf algebroid comodules 中的 Ext，不是普通环上模的 Ext。

## B.4 Morava descent 口径

**外部输入 B.8.** Morava descent spectral sequence 写作
$$
H_c^s(\mathbb G_n;(E_n)_tX)\Rightarrow \pi_{t-s}L_{K(n)}X.
$$

**检查表 B.9.** 使用 B.8 前必须记录：

- $\mathbb G_n$ 是 profinite group；
- $(E_n)_tX$ 的拓扑；
- 连续 cochains 的模型；
- $X$ 是否有限、dualizable 或满足其他完备性条件；
- 谱序列是否强收敛。

## B.5 Hidden extensions

**定义 B.10.** 若 $E^\infty$ 只给出 $\operatorname{gr}G_*$，而不能唯一恢复 $G_*$ 的加法或乘法结构，则剩余问题称为 hidden extension。

**警告 B.11.** 稳定 stems 和 $K(n)$-local sphere 计算中的 hidden extensions 是实质问题。表格中的 $E^\infty$ 页不能直接当作同伦群。

## 本附录小结

谱序列是 chromatic theory 的计算引擎，但也是错误高发区。本书所有谱序列调用必须说明指标、收敛和 hidden extension 状态。否则只能作为启发或边界说明。
