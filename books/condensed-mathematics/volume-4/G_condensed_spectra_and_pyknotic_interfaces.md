# 附录 G：凝聚谱、pyknotic 接口与同伦方向

## G.0 与第八章的分工

附录 E 给出 pyknotic 对象和凝聚同伦的入口，第八章已把 hyperdescent、循环凝聚谱计算、
Dirac cone 和六函子相容条件写入正文。本附录保留参考接口，集中说明从集合值 sheaf
到谱值 sheaf 时哪些定义直接升级，哪些结论需要额外高阶输入。

## G.1 空间值与谱值 sheaf

设 \(\mathcal C\) 为 compact Hausdorff 或 compacta 站点。记 \(\mathcal S\) 为空间的 \(\infty\)-范畴，\(\operatorname{Sp}\) 为谱的稳定 \(\infty\)-范畴。

**定义 G.1（hypercomplete 约定）.** 本附录把满足 hyperdescent 的空间值函子

$$
F:\mathcal C^{op}\to\mathcal S.
$$

称为超完备空间值凝聚对象，并把满足 hyperdescent 的谱值函子

$$
E:\mathcal C^{op}\to\operatorname{Sp}.
$$

称为超完备凝聚谱。若只要求覆盖的 Čech descent，则得到一般 sheaf 范畴；再作
hypercompletion 才进入本附录的约定。

**命题 G.2（谱值 sheaf 的稳定性）.** 谱值 sheaf 范畴是稳定 \(\infty\)-范畴。

**证明.** 预 sheaf 范畴

$$
\operatorname{Fun}(\mathcal C^{op},\operatorname{Sp})
$$

逐对象稳定。hyperdescent 等价的 fiber、loop 和 suspension 仍逐 hypercover 满足
hyperdescent，因为谱中的有限极限与有限余极限相同，并与 totalization 的相关极限
交换。因此 hypersheaf 全子范畴含零对象，对 fiber、loop 与 suspension 封闭；在稳定
范畴中这正说明它自身稳定。证毕。

## G.2 Eilenberg-Mac Lane 嵌入

阿贝尔群值 sheaf \(A\) 是谱值 hypersheaf 范畴 heart 中的 Eilenberg--Mac Lane
对象 \(HA\)。其 homotopy sheaves 只有 \(\pi_0(HA)=A\) 非零。一般不能把它的导出
截面逐对象定义为 \(H(A(S))\)：若 \(S\) 有高阶 sheaf cohomology，则
\(R\Gamma(S,HA)\) 还带有负次数 homotopy groups。

**命题 G.3.** \(H:\mathbf{CondAb}\to
\operatorname{Shv}^{\wedge}(\mathcal C,\operatorname{Sp})\) 全忠实到 Eilenberg--Mac
Lane 对象。

**证明.** 在导出 sheaf 范畴中，mapping spectrum 满足

$$
\pi_0\operatorname{Map}(HA,HB)
=\operatorname{Hom}_{\mathbf{CondAb}}(A,B),
\qquad
\pi_{-r}\operatorname{Map}(HA,HB)
=\operatorname{Ext}^r_{\mathbf{CondAb}}(A,B)
$$

对 \(r\ge1\)。mapping space 只看 mapping spectrum 的非负 homotopy，而 heart 对象间
这些群除 \(\pi_0\) 外为零，所以 heart 嵌入全忠实；Ext 信息位于 mapping spectrum 的
负次数。证毕。

**边界 G.4.** 不能把谱值 sheaf 等同于链复形 sheaf，除非选定 Eilenberg-Mac Lane 或 Dold-Kan/derived category 比较范围。非 connective 谱含有稳定同伦信息，超出普通凝聚阿贝尔群。

## G.3 Pyknotic 与 condensed 的比较口径

Barwick--Haine 的 pyknotic objects 是 compacta 站点上的 (hyper)sheaves。凝聚集合使用
compact Hausdorff 测试站点和相应覆盖；在 0-截断层，固定相同 universe 与站点后，
hypercompletion 不产生差别。

**输入定理 G.5（pyknotic 基础比较口径）.** 在固定 universe、compacta 站点和
hypercompletion 约定后，0-截断 pyknotic objects 与 condensed sets 等价。对空间值或
谱值对象，必须继续区分 sheaf 与 hypersheaf；不能只凭术语相近省略 hypercompletion。

本书只使用以下层面的比较：

1. 集合值对象：sheaf 条件和覆盖选择比较；
2. 阿贝尔群值对象：heart 层面的阿贝尔范畴；
3. 谱值对象：稳定 sheaf 范畴和 hyperdescent；
4. 几何应用：pro-etale、solid 和 analytic 结构需要额外输入。

**外部输入定理 G.5.1（Wolf）.** 对 coherent scheme $X$，hypercomplete pro-étale
$\infty$-topos 等价于 $\operatorname{Gal}(X)$ 在 pyknotic spaces 中的连续表示范畴。
这给出已知的 pro-étale--pyknotic 接口，但不把 pro-étale 站点与 compacta 站点直接
识别；其谱值稳定化与 solid/analytic localization 的相容还需另行检查。

## G.4 凝聚谱中的 exactness

**命题 G.6（hypercover criterion）。** 谱值预 sheaf \(E\) 是 hypersheaf，当且仅当
对每个 hypercover \(U_\bullet\to U\)，自然映射

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

**证明.** 在 hypersheaf 范畴的 Postnikov 完备 $t$-结构中，谱对象由所有 homotopy
sheaves 检测。若它们全部为零，则每个 Postnikov 截断都为零；hypercompleteness 说明
$E$ 是这些截断的极限，故 $E\simeq0$。若只取非 hypercomplete sheaf，这一检测结论
需要另加 left-completeness 假设。证毕。

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
