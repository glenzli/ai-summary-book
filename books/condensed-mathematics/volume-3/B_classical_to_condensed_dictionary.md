# 附录 B：经典语言与凝聚语言对照

## B.0 目标

本附录把复几何中的经典术语与本书使用的 condensed/analytic 语言对应起来。

## B.1 函数空间

| 经典对象 | 凝聚/解析语言 |
| --- | --- |
| 拓扑向量空间 $V$ | 凝聚向量空间 $S\mapsto\operatorname{Cont}(S,V)$ |
| Banach/Fréchet 空间 | liquid 或 analytic 模对象 |
| 全纯函数空间 $\mathcal O(U)$ | analytic/liquid $\mathbb C$-模 |
| 光滑形式 $\mathcal A^{p,q}(U)$ | liquid 模中的函数空间对象 |

## B.2 层与导出范畴

| 经典对象 | 凝聚/解析语言 |
| --- | --- |
| 相干解析层 $\mathcal F$ | analytic ringed space 上的相干模 |
| $D_{\operatorname{coh}}(X)$ | analytic 派生范畴的相干子范畴 |
| $R\Gamma(X,\mathcal F)$ | analytic/liquid 派生全局截面 |
| Dolbeault resolution | liquid 模复形中的 resolution |

## B.3 对偶和推前

| 经典对象 | 凝聚/解析语言 |
| --- | --- |
| proper pushforward $Rf_*$ | proper 情形下与 $f_!$ 比较 |
| compact support cohomology | $f_!$ |
| dualizing sheaf $\omega_X$ | $f^!$ 作用于基域对象的结果 |
| Serre trace | $f_!\dashv f^!$ 的 counit |
| projection formula | $f_!(f^*M\otimes N)\simeq M\otimes f_!N$ |

## B.4 经典定理的翻译

| 经典定理 | 凝聚/解析表述 |
| --- | --- |
| coherent cohomology finite-dimensionality | $R\Gamma(X,\mathcal F)$ 是有限型/perfect 对象 |
| Serre duality | $f_!$ 与 $f^!$ 的对偶公式 |
| GAGA | algebraic 与 analytic 相干导出范畴比较 |
| Riemann-Roch | trace、Chern character 与 Todd class 的兼容性 |

## B.5 常见误读

1. 凝聚化不是忘记拓扑，而是用测试对象重新编码拓扑。
2. liquid 空间不是 Banach 空间的新名字，而是适合同调代数的分析范畴。
3. $f_!$ 不是普通 $f_*$，非 proper 情形中它记录支撑和边界条件。
4. GAGA 不是所有 analytic 与 algebraic 对象都等价；properness 是关键假设。
5. Riemann-Roch 不只是 Euler characteristic 公式，还包含 functorial trace 和 characteristic class 的兼容性。

## B.6 本附录小结

第三卷的核心工作是翻译：把经典复几何中的函数空间、层、上同调、对偶和特征类放入 condensed/analytic 的派生范畴框架。
