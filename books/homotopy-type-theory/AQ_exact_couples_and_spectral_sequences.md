# 附录 AQ：Exact Couples、谱序列与收敛接口

本附录把“谱序列”从一句研究边界推进为严格代数接口。这里不证明特定拓扑谱序列的收敛定理，而是给出 exact couple、derived couple、页 $E_r$、微分和条件收敛的精确定义。

## AQ.1 Exact couple

**定义 AQ.1（graded group）.** 一个 graded group 为族
$$
G_\ast:\mathbb Z\to\mathsf{Group}.
$$
若每个 $G_n$ 为阿贝尔群，则称为 graded abelian group。

**定义 AQ.2（exact couple）.** exact couple 由 graded groups $D,E$ 和次数齐次同态
$$
i:D\to D,\qquad
j:D\to E,\qquad
k:E\to D
$$
组成，满足三处 exactness：
$$
\mathsf{im}(i)=\ker(j),\qquad
\mathsf{im}(j)=\ker(k),\qquad
\mathsf{im}(k)=\ker(i).
$$
次数记号可为
$$
\deg(i)=a,\quad\deg(j)=b,\quad\deg(k)=c.
$$

**定义 AQ.3（谱序列微分）.** 对 exact couple，定义
$$
d\coloneqq j\circ k:E\to E.
$$

**命题 AQ.4（$d^2=0$）.** 有
$$
d\circ d=0.
$$

**证明.** 展开：
$$
d^2=j\circ k\circ j\circ k.
$$
由 exactness $\mathsf{im}(j)=\ker(k)$，有 $k\circ j=0$。故
$$
j\circ(k\circ j)\circ k=j\circ0\circ k=0.
$$
$\square$

## AQ.2 Homology 与 derived couple

**定义 AQ.5（homology of a differential）.** 若 $d:E\to E$ 且 $d^2=0$，定义
$$
H(E,d)\coloneqq\ker(d)/\mathsf{im}(d)
$$
为集合商或阿贝尔群商。HoTT 中该商使用集合商 HIT 或已构造的阿贝尔群商。

**定义 AQ.6（derived couple）.** 给定 exact couple $(D,E,i,j,k)$，定义
$$
D'\coloneqq\mathsf{im}(i),\qquad
E'\coloneqq H(E,d).
$$
映射 $i',j',k'$ 由：

1.  $i'$ 是 $i$ 在 $\mathsf{im}(i)$ 上诱导的映射；
2.  $j'$ 把 $i(x)$ 送到 $[j(x)]$；
3.  $k'$ 把 homology 类 $[e]$ 送到 $k(e)$，并视为 $\mathsf{im}(i)$ 中的元素。

**命题 AQ.7（derived couple well-definedness）.** $j'$ 和 $k'$ 定义良好。

**证明（证明核）.** 对 $j'$，若 $i(x)=i(x')$，则 $x-x'\in\ker(i)=\mathsf{im}(k)$，取 $e:E$ 使 $k(e)=x-x'$。于是
$$
j(x)-j(x')=j(k(e))=d(e),
$$
所以 $[j(x)]=[j(x')]$。对 $k'$，若 $e$ 为 cycle，即 $d(e)=j(k(e))=0$，则 $k(e)\in\ker(j)=\mathsf{im}(i)$，故 $k(e):D'$；若 $e$ 改变一个 boundary $d(e_0)=j(k(e_0))$，则
$$
k(d(e_0))=k(j(k(e_0)))=0,
$$
因此 $k'$ 不变。$\square$

**定理 AQ.8（derived couple exactness）.** $(D',E',i',j',k')$ 仍是 exact couple。

**证明状态.** 这是 exact couple 的标准代数证明。三处 exactness 分别由 AQ.2 的三处 exactness 和 AQ.7 的商定义推出；HoTT 内逐项展开时需使用集合商归纳和子群/商群接口。本书把其作为书内代数证明核。

## AQ.3 谱序列页

**定义 AQ.9（由 exact couple 生成的谱序列）.** 从 exact couple 递归定义：
$$
(D_1,E_1,i_1,j_1,k_1)\coloneqq(D,E,i,j,k),
$$
$$
(D_{r+1},E_{r+1},i_{r+1},j_{r+1},k_{r+1})
\coloneqq
\mathsf{Derived}(D_r,E_r,i_r,j_r,k_r).
$$
第 $r$ 页微分为
$$
d_r\coloneqq j_r\circ k_r:E_r\to E_r.
$$

**命题 AQ.10（页递推）.** 有
$$
E_{r+1}\cong H(E_r,d_r).
$$

**证明.** 这是定义 AQ.6 和 AQ.9。$\square$

## AQ.4 过滤与收敛

**定义 AQ.11（filtered group）.** 群 $G$ 的递减过滤为子群族
$$
\cdots\subseteq F^{p+1}G\subseteq F^pG\subseteq\cdots\subseteq G.
$$
其 associated graded 为
$$
\mathsf{gr}^pG\coloneqq F^pG/F^{p+1}G.
$$

**定义 AQ.12（条件收敛）.** 谱序列 $E_r^{p,q}$ 条件收敛到 filtered group $G$，记作
$$
E_r^{p,q}\Rightarrow G^{p+q},
$$
若存在 $r=\infty$ 页
$$
E_\infty^{p,q}
$$
并有同构
$$
E_\infty^{p,q}\cong \mathsf{gr}^pG^{p+q}.
$$
此外需给出过滤的 exhaustiveness、separatedness 或 completeness 条件，具体取决于谱序列类型。

**警告 AQ.13（收敛不是形式结果）。** exact couple 给出页和微分，但不自动给出目标 $G$ 或收敛。收敛定理依赖过滤、完备性、界限条件和 exactness 假设；每个具体谱序列必须单独证明。

## AQ.5 HoTT 中的使用接口

**输入 AQ.14（HoTT 谱序列应用所需数据）.** 若要在 HoTT 中构造 Serre、Adams 或 Atiyah-Hirzebruch 型谱序列，至少需提供：

1.  一个 filtered chain/cochain/fiber tower 对象；
2.  由其 exact triangle 或 long exact sequence 产生的 exact couple；
3.  每页 $E_r$ 的双次数和微分次数；
4.  $E_2$ 或可计算页的识别；
5.  收敛目标和过滤；
6.  extension problems 的处理。

**当前状态 AQ.15.** 本书已经有 EM 上同调、smash product、fiber sequence 和长正合列接口；这足以陈述若干谱序列的输入格式。但要证明 Serre spectral sequence 或 Adams spectral sequence，仍需单独发展 filtered spectra、exact couples 的拓扑来源和收敛理论。
