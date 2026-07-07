# 附录 F：卷积 correspondence、proper base change 与结合性检查

## 本章目标

本附录给出 Hecke category、Steinberg variety、geometric Satake 和 Coulomb branch 中反复使用的卷积检查表。

## F.1 通用模板

**定义 F.1.** 卷积数据由对象空间 $X$、乘法 correspondence
$$
X\times X\xleftarrow{p}Z\xrightarrow{m}X
$$
和 sheaf/homology theory 给出。对 sheaves，
$$
\mathcal F\star\mathcal G=m_!p^\ast(\mathcal F\boxtimes\mathcal G).
$$
对 Borel-Moore homology，卷积通过 fiber product 上的 pull-intersect-push 定义。

**检查表 F.2.** 每次定义卷积必须记录：

1. $X,Z$ 的几何类型；
2. $p,m$ 是否 representable、proper、smooth 或 ind-proper；
3. 使用 $m_!$ 还是 $m_\ast$；
4. 是否需要 shift 或 Tate twist；
5. associativity 使用哪个三重 correspondence；
6. 单位对象或单位类的支撑。

## F.2 结合性命题

**命题 F.3.** 若卷积 correspondence 来自结合乘法，且六函子 formalism 对相关 Cartesian squares 满足 base change，则卷积 functor 结合。

**证明.** 三重卷积由三重 correspondence 控制。两种加括号方式分别对应两个 fiber product 的迭代。fiber product 的 canonical associativity 给出两个几何对象的自然同构；base change 把 functor 复合化为沿三重 correspondence 的单次 pull-push；乘法结合律使目标映射一致。因此得到自然 associator。$\square$

**定义 F.4.** 卷积单位由单位 correspondence 给出。若存在 $e:\mathrm{pt}\to X$ 表示单位轨道，则 sheaf 版本的单位通常为
$$
\mathbf 1=e_!E[\dim e]
$$
或其 perverse normalization；homology 版本的单位为单位分支的 fundamental class。

**命题 F.5.** 若单位 correspondence 与左右乘法图满足 base change，则 $\mathbf 1\star\mathcal F\simeq\mathcal F\simeq\mathcal F\star\mathbf 1$。

**证明.** 左单位的 convolution correspondence 是
$$
X\xleftarrow{\ \simeq\ }\mathrm{pt}\times X\xrightarrow{\ \mathrm{id}\ }X
$$
的相应 quotient 或 stacky 版本。pullback 沿同构不改变对象，pushforward 沿恒等映射也不改变对象。右单位相同。若存在 quotient stack 或 ind-scheme，需用相同论证加上 equivariant descent。$\square$

## F.3 四个核心实例

| 场景 | $X$ | correspondence | 输出 |
| --- | --- | --- | --- |
| finite Hecke | $B\backslash G/B$ | $G\times^B G$ | Hecke category |
| Springer | $\widetilde{\mathcal N}\times_{\mathcal N}\widetilde{\mathcal N}$ | 三重 Springer fiber product | $\mathbb C[W]$ 等 |
| affine Satake | $G[[z]]\backslash G((z))/G[[z]]$ | Beilinson-Drinfeld/affine Grassmannian convolution | tensor category |
| BFN Coulomb | $\mathcal R$ over affine Grassmannian | Borel-Moore convolution | Coulomb branch algebra |

**例 F.6.** finite Hecke 情形中，单位对象支撑在闭点 $B/B\subset G/B$。若 $X_w=BwB/B$，则
$$
\mathbf 1\star j_{w!}E_{X_w}[\ell(w)]\simeq j_{w!}E_{X_w}[\ell(w)].
$$
这是第四章标准对象卷积的最小检查。

**例 F.7.** affine Grassmannian 情形中，单位对象支撑在中性 lattice $L^+G/L^+G$。convolution Grassmannian 的 fiber over this unit 退化为原 Grassmannian，因此 Satake category 的 tensor unit 是中性 orbit 的 skyscraper perverse sheaf。

## F.4 常见错误

**警告 F.8.** 不能只写
$$
\mathcal F\star\mathcal G=m_\ast(\mathcal F\boxtimes\mathcal G)
$$
而省略 correspondence。多数几何表示论卷积并不是 $X$ 上已有乘法的直接 pushforward，而是先经过 contracted product 或 fiber product。

**警告 F.9.** $m_!$ 与 $m_\ast$ 的相等需要 properness 或 ind-properness。若映射非 proper，二者可能不同，Verdier duality 也会交换 standard 与 costandard objects。

## 本章小结

本附录给出卷积的统一类型检查表、结合性证明、单位对象检查和常见错误。后续章节不得只说“按卷积定义乘法”，而必须指出 correspondence、properness、base change 和所用 functor。
