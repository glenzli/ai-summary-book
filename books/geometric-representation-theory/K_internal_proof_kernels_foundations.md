# 内部证明核 I：基础几何、表示和卷积

本文件把当前书稿中可内部证明的基础命题扩写为更细证明核。正式章节可逐步吸收这些证明。

## K.1 齐性空间上的 equivariant vector bundles

**命题 K.1.** 令 $H$ 为代数群，$K\subset H$ 为 closed subgroup。则 $H$-equivariant vector bundles on $H/K$ 与有限维 algebraic $K$-representations 等价。

**证明.** 给定 $K$-表示 $V$，构造
$$
E_V=H\times^K V=(H\times V)/K,
$$
其中
$$
(h,v)\cdot k=(hk,k^{-1}v).
$$
投影 $E_V\to H/K$ 由 $[h,v]\mapsto hK$ 给出。因为 $H\to H/K$ 是 principal $K$-bundle，局部平凡化后 $E_V$ 同构于 $(H/K)\times V$，故为 vector bundle。左 $H$-作用
$$
h_0\cdot[h,v]=[h_0h,v]
$$
well-defined，因为它与右 $K$-作用交换。

反向地，给定 $H$-equivariant vector bundle $E\to H/K$，取基点 $eK$ 上的纤维 $E_{eK}$。稳定子为 $K$。对 $k\in K$，equivariance 给出线性自同构
$$
k:E_{eK}\to E_{keK}=E_{eK},
$$
从而得到 $K$-表示。若 $\phi:E\to E'$ 是 $H$-equivariant bundle map，则 $\phi_{eK}$ 是 $K$-linear。

最后检查两个构造互逆。由 $V$ 出发再取基点纤维得到
$$
(E_V)_{eK}\simeq V.
$$
由 $E$ 出发构造 $H\times^K E_{eK}$，映射
$$
[h,v]\mapsto h\cdot v\in E_{hK}
$$
well-defined：若 $[h,v]=[hk,k^{-1}v]$，则
$$
hk\cdot(k^{-1}v)=h\cdot v.
$$
它在每个纤维上为线性同构，并与 $H$-作用相容。$\square$

## K.2 Verma module 泛性质

**命题 K.2.** 对 $\lambda\in\mathfrak t^\ast$，Verma module
$$
M(\lambda)=U(\mathfrak g)\otimes_{U(\mathfrak b)}\mathbb C_\lambda
$$
表示 functor
$$
N\mapsto\{v\in N\mid \mathfrak n v=0,\ hv=\lambda(h)v,\ \forall h\in\mathfrak t\}.
$$

**证明.** 任一 $\mathfrak g$-module map $f:M(\lambda)\to N$ 由 $f(1\otimes1)$ 唯一决定。因为 $1\otimes1$ 满足 $\mathfrak n$ 杀掉和 $\mathfrak t$ 权为 $\lambda$，其像必须满足同样条件。

反向，给定 $v\in N$ 满足这些条件，定义
$$
\tilde f:U(\mathfrak g)\to N,\qquad u\mapsto uv.
$$
若 $b\in U(\mathfrak b)$，则
$$
\tilde f(ub)=u(bv).
$$
$\mathfrak b$ 在 $\mathbb C_\lambda$ 上的作用由 character 和 $\mathfrak n$ 平凡作用给出，所以 $\tilde f$ 与张量关系
$$
ub\otimes1=u\otimes b\cdot1
$$
相容，下降为 $M(\lambda)\to N$。两个构造互逆。$\square$

## K.3 Category $\mathcal O$ 的 extension 封闭性

**命题 K.3.** 若
$$
0\to M'\to M\to M''\to0
$$
是 weight $\mathfrak g$-modules 范畴中的短正合列，且 $M',M''\in\mathcal O$，则 $M\in\mathcal O$。若短正合列只给在所有 $\mathfrak g$-modules 中，则需要调用 category $\mathcal O$ 是 abelian subcategory 的外部输入。

**证明.** 有限生成性：取 $M'$ 的有限生成元集合和 $M''$ 的有限生成元在 $M$ 中的任意提升。这些元素生成 $M$。

权分解：短正合列在 weight modules 范畴中，因此
$$
0\to M'_\mu\to M_\mu\to M''_\mu
$$
正合，且
$$
\dim M_\mu\le \dim M'_\mu+\dim M''_\mu<\infty.
$$
这一步正是普通 $\mathfrak g$-module extension 与 weight-module extension 的区别；本命题只作内部证明核，完整 abelian 性在附录 D 中作为外部输入定位。

$\mathfrak n$ locally finite：对 $m\in M$，其像 $\bar m\in M''$ 生成有限维 $U(\mathfrak n)\bar m$。取该有限维空间的一组提升，$U(\mathfrak n)m$ 模去 $M'$ 后落在有限维空间中；与 $M'$ 的交由有限多个元素在 $M'$ 中生成，因 $M'$ locally finite 得有限维。故 $U(\mathfrak n)m$ 有限维。$\square$

## K.4 卷积结合性的抽象证明

**命题 K.4.** 设有可复合 correspondence
$$
X\times X\xleftarrow{p}Z\xrightarrow{m}X
$$
定义卷积。若三重 correspondence 存在，所有相关 base change 和 projection formula 成立，且 $m$ 来自结合乘法，则卷积结合。

**证明.** 对 $\mathcal F,\mathcal G,\mathcal H\in D(X)$，
$$
(\mathcal F\star\mathcal G)\star\mathcal H
$$
展开为沿两个 correspondence 连续执行 pull-push。用 base change 把中间的 pullback past pushforward 交换后，得到沿三重空间 $Z^{(3)}$ 的单次表达：
$$
m^{(3)}_!p^{(3)\ast}(\mathcal F\boxtimes\mathcal G\boxtimes\mathcal H).
$$
另一种加括号给出同一个三重空间和同一个总乘法 $m^{(3)}$。因此两者自然同构。结合约束的 pentagon identity 来自四重 correspondence 和乘法结合律的相干性。$\square$
