# 附录 T：单值性推出函数外延性的形式化输入

## T.0 目标

本附录处理 K.1.2 的缺口：第六章定理 6.11“单值性推出函数外延性”。当前版本不把该长证明完全重写为书内路径代数证明，而是把它登记为精确外部输入，并给出 UniMath 与 Coq-HoTT 的形式化入口。

本附录的结论是：

**外部定理 T.0.1.** 在单值基础中，univalence 蕴含依赖函数外延性：
$$
\prod_{T:\mathcal U}\prod_{P:T\to\mathcal U}
\prod_{f,g:\prod_{t:T}P(t)}
\mathsf{isEquiv}\bigl(\mathsf{happly}_{f,g}\bigr).
$$

这里 $\mathsf{happly}_{f,g}:(f=g)\to\prod_{t:T}f(t)=g(t)$ 由路径归纳定义。

## T.1 陈述的层级

本书区分三种函数外延性陈述。

**定义 T.1.1（普通函数外延性）。** 对 $X,Y:\mathcal U$ 和 $f,g:X\to Y$，若
$$
h:\prod_{x:X}f(x)=g(x),
$$
则有路径 $f=g$。

**定义 T.1.2（依赖函数外延性）。** 对 $T:\mathcal U$、$P:T\to\mathcal U$ 和
$$
f,g:\prod_{t:T}P(t),
$$
若
$$
h:\prod_{t:T}f(t)=g(t),
$$
则有路径 $f=g$。

**定义 T.1.3（强函数外延性）。** 映射
$$
\mathsf{happly}_{f,g}:(f=g)\to\prod_{t:T}f(t)=g(t)
$$
是等价。

强函数外延性蕴含依赖函数外延性，方法是取 $\mathsf{happly}_{f,g}$ 的逆映射。第六章公理 6.2 采用强函数外延性。

## T.2 UniMath 的形式化链条

UniMath 在
$$
\texttt{UniMath/Foundations/UnivalenceAxiom.v}
$$
中把相关陈述拆成若干 statement，并证明从 univalence 到函数外延性的蕴含。

在 commit
$$
\texttt{9ed7661d3ad33c74e35824efccf861b4fdc17323}
$$
中可核查以下入口：

| 入口 | 作用 |
|---|---|
| `univalenceStatement` | 陈述 $\prod X\,Y,\mathsf{isweq}(\mathsf{eqweqmap}_{X,Y})$ |
| `funextfunStatement` | 普通函数外延性 |
| `funextsecStatement` | 依赖函数外延性的函数方向 |
| `isweqtoforallpathsStatement` | $\mathsf{happly}$ 是 weak equivalence 的强形式 |
| `funextsecImplication` | 从强形式推出依赖函数外延性的函数方向 |
| `funextfunPreliminaryUAH` | 在 univalence 假设下证明普通函数外延性 |
| `funcontrUAH` | 在 univalence 假设下证明可收缩族的函数空间可收缩 |
| `funextcontrUAH` | 由上一项得到依赖函数的 contractible cone 陈述 |
| `isweqtoforallpathsUAH` | 在 univalence 假设下证明强依赖函数外延性 |
| `funextsecweqFromUnivalence` | 把 `univalenceStatement` 映到 `isweqtoforallpathsStatement` |

因此，UniMath 给出的形式化链条是：
$$
\mathsf{univalenceStatement}
\Longrightarrow
\mathsf{isweqtoforallpathsStatement}
\Longrightarrow
\mathsf{funextsecStatement}.
$$

这里第一步由 `funextsecweqFromUnivalence := isweqtoforallpathsUAH` 给出；第二步由 `funextsecImplication` 给出。

## T.3 证明路线的数学内容

UniMath 的证明路线可压缩为以下四步。

**步骤 1：等价预合成给函数空间等价。**  
若 $w:X\simeq X'$，则预合成
$$
(X'\to Y)\to(X\to Y),\qquad f\mapsto f\circ w
$$
是等价。UniMath 入口为 `isweqcompwithweqUAH`。证明核心是：由单值性把 $w$ 变成类型路径，再把预合成识别为沿该路径的 transport。

**步骤 2：从 path space 的投影得到普通函数外延性。**  
考虑路径空间对象
$$
\mathsf{pathsspace}(Y)\simeq \sum_{y_0,y_1:Y}(y_0=y_1).
$$
其两个端点投影在适当的路径空间上给出同伦。利用步骤 1 的函数空间等价，可把逐点路径提升为函数路径。UniMath 入口为 `apathpr1toprUAH` 和 `funextfunPreliminaryUAH`。

**步骤 3：普通函数外延性推出可收缩族的函数空间可收缩。**  
若每个 $P(x)$ 可收缩，则
$$
\prod_{x:X}P(x)
$$
可收缩。中心是逐点中心；唯一性由普通函数外延性逐点提升。UniMath 入口为 `funcontrUAH`。

**步骤 4：可收缩 cone 推出依赖函数外延性的强形式。**  
固定 $g:\prod_{t:T}P(t)$。考虑
$$
\sum_{f:\prod_{t:T}P(t)}\prod_{t:T} f(t)=g(t).
$$
它是 $g$ 的 path cone。步骤 3 证明该类型可收缩。另一方面
$$
\sum_{f:\prod_{t:T}P(t)}(f=g)
$$
由路径类型的基本性质可收缩。两个可收缩类型之间的规范映射是等价，得到 $\mathsf{happly}$ 是等价。UniMath 入口为 `funextcontrUAH` 和 `isweqtoforallpathsUAH`。

## T.4 Coq-HoTT 的入口

Coq-HoTT 在
$$
\texttt{theories/Types/Universe.v}
$$
中给出：

| 入口 | 作用 |
|---|---|
| `Univalence` | 单值性公理类 |
| `isequiv_equiv_path` | 类型路径到等价是等价 |
| `Univalence_implies_Funext` | 从 `Univalence` 得到 `Funext` 的实例入口 |
| `path_universe` | 等价到类型路径 |
| `transport_path_universe` | 沿 `path_universe` 的 transport 计算 |

Coq-HoTT 源码注释说明，实际证明可在 metatheory 文件 `UnivalenceImpliesFunext` 中找到；当前库由于 typeclass 设计，把 `Univalence_implies_Funext` 作为实例入口使用。对本书来说，这足以作为形式化对照，但若选择 Coq-HoTT 做逐行机器化，还必须核查相应 metatheory 文件与当前 build 的连接方式。

## T.5 本书采用的结论

本书从第六章以后可以采用以下两种口径之一：

1. 公理化 HoTT 口径：同时声明函数外延性与单值性。此时定理 6.11 作为外部冗余性说明。
2. 最小单值口径：只声明单值性，并引用外部定理 T.0.1 得到函数外延性。

当前正文采用第一种口径，因为它使前几章的证明依赖更清楚；附录 T 则保证第二种口径有形式化来源。

**剩余机器化义务。** 若后续将本书机器化，应选择一种口径，不应在同一形式化目标中同时把函数外延性作为公理又使用“单值性推出函数外延性”来降低公理依赖，除非同时记录 `Print Assumptions` 或对应依赖分析。
