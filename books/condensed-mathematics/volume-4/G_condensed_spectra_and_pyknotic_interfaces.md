# 附录 G：凝聚谱、pyknotic 接口与同伦方向

## G.0 目标

附录 E 给出 pyknotic 对象和凝聚同伦的入口。本附录进一步补凝聚谱与 pyknotic 谱的接口，说明从集合值 sheaf 到谱值 sheaf 时，哪些定义直接升级，哪些定理需要新的高阶输入。

本附录只作工具卷接口，不把稳定同伦论发展为第五卷。

## G.1 空间值与谱值 sheaf

设 \(\mathcal C\) 为 compact Hausdorff 或 compacta 站点。记 \(\mathcal S\) 为空间的 \(\infty\)-范畴，\(\operatorname{Sp}\) 为谱的稳定 \(\infty\)-范畴。

**定义 G.1.** 空间值凝聚对象为满足 hyperdescent 的函子

$$
F:\mathcal C^{op}\to\mathcal S.
$$

谱值凝聚对象为满足 hyperdescent 的函子

$$
E:\mathcal C^{op}\to\operatorname{Sp}.
$$

**命题 G.2（谱值 sheaf 的稳定性）.** 谱值 sheaf 范畴是稳定 \(\infty\)-范畴。

**证明.** 预 sheaf 范畴

$$
\operatorname{Fun}(\mathcal C^{op},\operatorname{Sp})
$$

逐对象稳定。sheaf 条件由若干极限条件给出，因此 sheaf 全子范畴对有限极限封闭。谱范畴中有限极限与有限余极限相容，局部化到 sheaf 后仍稳定。证毕。

## G.2 Eilenberg-Mac Lane 嵌入

阿贝尔群值 sheaf \(A\) 给谱值 sheaf \(HA\)：

$$
S\mapsto H(A(S)).
$$

**命题 G.3.** \(H:\mathbf{CondAb}\to\operatorname{Shv}(\mathcal C,\operatorname{Sp})\) 全忠实到 connective Eilenberg-Mac Lane 对象。

**证明.** 对阿贝尔群 \(M,N\)，谱映射空间满足

$$
\operatorname{Map}_{\operatorname{Sp}}(HM,HN)
$$

的 \(\pi_0\) 为 \(\operatorname{Hom}_{\mathbf{Ab}}(M,N)\)，负同伦群为零，正同伦群对应 Ext 信息。若限制到 heart，即只取 \(0\)-截断映射，则得到阿贝尔群态射。sheaf 层面逐对象并由 sheaf 条件兼容。精确的全忠实陈述应放在 connective spectra 的 heart 等价中。证毕。

**边界 G.4.** 不能把谱值 sheaf 等同于链复形 sheaf，除非选定 Eilenberg-Mac Lane 或 Dold-Kan/derived category 比较范围。非 connective 谱含有稳定同伦信息，超出普通凝聚阿贝尔群。

## G.3 Pyknotic 与 condensed 的比较口径

Barwick-Haine 的 pyknotic objects 是 compacta 站点上的 hypersheaves。condensed sets 使用 compact Hausdorff 测试站点和相应覆盖。

**输入定理 G.5（pyknotic 基础比较口径）.** 在固定 universe 和站点选择后，pyknotic objects 与 condensed objects 有共同的 sheaf-theoretic 核心：都以 compact 测试对象上的 sheaf/hypersheaf 编码拓扑信息。二者在 hyperdescent、coherent topos 性质和 universe 管理上采用不同约定。

本书只使用以下层面的比较：

1. 集合值对象：sheaf 条件和覆盖选择比较；
2. 阿贝尔群值对象：heart 层面的阿贝尔范畴；
3. 谱值对象：稳定 sheaf 范畴和 hyperdescent；
4. 几何应用：pro-etale、solid 和 analytic 结构需要额外输入。

## G.4 凝聚谱中的 exactness

**命题 G.6（fiberwise criterion）。** 谱值预 sheaf \(E\) 是 sheaf，当且仅当对每个覆盖超 Čech 对象 \(U_\bullet\to U\)，自然映射

$$
E(U)\to\varprojlim_{\Delta}E(U_\bullet)
$$

为谱等价。

**证明.** 这是谱值 sheaf 条件的定义。因 \(\operatorname{Sp}\) 是 \(\infty\)-范畴，匹配条件必须用 totalization 表达，而不是只用一阶 equalizer。证毕。

**推论 G.7.** 若谱值 sheaf \(E\) 的每个 homotopy sheaf

$$
\pi_n(E)
$$

为零，则 \(E\simeq0\)。

**证明.** 谱对象由所有 homotopy groups 检测。sheaf 化后 \(\pi_n(E)\) 是阿贝尔群值 sheaf；若全部为零，则对每个测试对象 \(S\)，谱 \(E(S)\) 的同伦群全为零，故 \(E(S)\simeq0\)。证毕。

## G.5 与 solid/analytic 的接口

solidification 可被看作某类 Bousfield localization。谱值版本需要把生成 cone 从凝聚阿贝尔群提升到谱值 sheaf 中。

**输入定理 G.8（谱值 solid/analytic localization 口径）.** solid、analytic、liquid 的稳定范畴版本可表述为谱值或派生 sheaf 范畴中的 Bousfield localization；其 kernel 由 Dirac-to-measure cone 的稳定化生成，并与张量积相容。

**边界 G.9.** G.8 是高阶结构输入。第一、二卷中的阿贝尔群值论证不能自动推出谱值张量理想性；必须检查 stable localization、compact generation 和 monoidal compatibility。

## G.6 形式化优先级

若要把本套讲义迁移到 proof assistant，优先级应为：

1. 集合值 sheaf 与阿贝尔群值 sheaf；
2. Stone/profinite 基础与站点比较；
3. 投射分解、Ext/Tor；
4. 谱值 sheaf 的稳定性；
5. Bousfield localization 抽象定理；
6. solid/analytic 生成 cone 和张量理想输入。

前四项可作为独立形式化项目；后两项依赖更多高阶范畴库。

## 练习

1. 写出谱值 sheaf 条件与阿贝尔群值 sheaf 等化子条件的差异。
2. 证明谱值 sheaf 范畴对 fiber 封闭。
3. 说明为什么 \(\pi_n(E)=0\) 对所有 \(n\) 可检测 \(E\simeq0\)。
4. 列出把 solidification 升级到谱值 sheaf 时需要检查的三项结构。
