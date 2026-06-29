# 附录 G：复几何主定理的依赖链

## G.0 目标

本附录把第三卷主体中的定理按依赖链重写。读者应能看出每个结论依赖哪些经典输入、哪些凝聚/解析输入、哪些步骤在本书内已经证明。

## G.1 Dolbeault 计算链

目标结论：

$$
R\Gamma(X,\mathcal O(E))
\simeq
\Gamma(X,\mathcal A_X^{0,\bullet}(E))
$$

在导出范畴中成立。

依赖：

1. 输入 F.1：Dolbeault resolution。
2. fine sheaf acyclicity：$\mathcal A_X^{0,q}(E)$ acyclic。
3. acyclic resolution 计算 derived global sections。

本书证明：

1. F.2 从 F.1 和 fine acyclicity 推出 Dolbeault cohomology 计算。
2. 卷四 6.3.1 说明连续微分复形可凝聚化为凝聚向量空间复形。

凝聚/解析额外输入：

1. 输入 F.14(1)：Dolbeault 复形可提升到 analytic/liquid 语境。
2. 输入 F.14(2)：提升后仍计算导出全局截面。

## G.2 有限性链

目标结论：

$$
\dim_\mathbb C H^i(X,\mathcal F)<\infty.
$$

向量丛情形依赖：

1. F.2：Dolbeault cohomology 计算。
2. F.7：Dolbeault Laplacian 的 Fredholm/Hodge theorem。
3. harmonic forms 空间有限维。

相干层一般情形依赖：

1. 局部有限表示。
2. 有限 Stein 覆盖或解析空间的有限性定理。
3. Cartan B 与 Čech-to-derived spectral sequence。
4. Grauert finiteness 或 coherent finiteness 输入 F.6。

本书证明的部分：

1. 若有一个有限维向量空间组成的有界复形，则其同调有限维。
2. 若有 acyclic Stein 覆盖，则 Čech 复形计算上同调。

未证明部分：

1. 紧复空间存在足够好的有限 Stein 覆盖及其相干控制。
2. 椭圆正则性或 Grauert finiteness。

## G.3 Serre duality 链

目标结论：

$$
H^q(X,\mathcal O(E))^\vee
\cong
H^{n-q}(X,\mathcal O(E^\vee\otimes\omega_X)).
$$

依赖：

1. Dolbeault 计算链 G.1。
2. 积分配对
   $$
   \int_X\operatorname{tr}(\alpha\wedge\beta).
   $$
3. Stokes 定理和 Leibniz 规则，保证配对下降到同调。
4. Hodge star 或 elliptic theory，保证非退化。
5. 有限性，保证非退化配对等价于完美对偶。

本书证明：

1. 配对与 $\bar\partial$ 相容。
2. 在 Riemann surface 情形可写出具体形式。

未证明部分：

1. 完美性。
2. 对任意 coherent sheaf 的 Ext 形式 Serre duality。

凝聚/解析翻译：

1. $f^!$ 对应 dualizing object。
2. trace map 对应积分。
3. $f_!\dashv f^!$ 的 counit 给出抽象 Serre pairing。

## G.4 GAGA 链

目标结论：

$$
\operatorname{Coh}(X)
\simeq
\operatorname{Coh}(X^{an})
$$

并且

$$
H^i(X,\mathcal F)\cong H^i(X^{an},\mathcal F^{an}).
$$

依赖：

1. $X$ proper over $\mathbb C$。
2. algebraic coherent sheaves 的解析化。
3. properness 控制无穷远，使解析相干层代数化。
4. 上同调比较。

本书证明：

1. properness 不可省略的反例：$\mathbb A^1$ 的解析函数多于多项式。
2. $\mathbb P^1$ 上线丛上同调与经典公式相容的例子。

未证明部分：

1. 解析相干层代数化。
2. 上同调比较的一般证明。

凝聚/解析翻译：

1. algebraic 与 analytic coherent theory 被放入同一 analytic 派生框架。
2. $R\Gamma$ 比较在导出层面陈述，而不是逐个 $H^i$ 分开证明。

## G.5 Riemann-Roch 链

目标结论：

$$
\chi(X,E)=\int_X\operatorname{ch}(E)\operatorname{td}(T_X).
$$

依赖：

1. 有限性：$\chi(X,E)$ 有定义。
2. Chern character 的构造与加法性。
3. Todd class 的构造。
4. trace/integration map。
5. pushforward 与 characteristic classes 的相容性。

本书证明：

1. $\mathbb P^1$ 上 $\mathcal O(d)$ 的公式。
2. Chern character 低阶项的形式展开。

未证明部分：

1. splitting principle 的完整使用。
2. Grothendieck-Riemann-Roch 的 deformation to normal cone 或 K-theory 证明。
3. condensed/analytic trace 与 characteristic classes 的一般相容性。

## G.6 六函子链

第三卷第八章只作为展望，不把六函子形式作为已证明理论。若要严格证明，需要：

1. 对空间或解析栈的对象类别给出定义。
2. 构造 $f^*,Rf_*,f_!,f^!$。
3. 证明伴随关系。
4. 证明 base change。
5. 证明 projection formula。
6. 证明 Verdier/Serre duality。

第二卷提供 $f_!$ 和 $f^!$ 的仿射有限型入口，但不足以推出完整六函子形式。

## G.7 读者检查表

读第三卷任一主定理时，应回答：

1. 这是经典定理、凝聚/解析翻译，还是本书证明的推论？
2. 是否需要 compactness/properness？
3. 是否需要 coherent 假设？
4. 是否需要 finite-dimensionality？
5. 是否使用 Dolbeault、Cartan、Hodge、GAGA 或 HRR 输入？
6. 是否进入 liquid/analytic 派生范畴？
7. 忘记凝聚结构后是否恢复经典陈述？

若这些问题有一个无法回答，该定理在本书当前版本中就还没有达到致密教材标准。
