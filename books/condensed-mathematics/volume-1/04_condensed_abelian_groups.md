# 第四章：凝聚阿贝尔群

把凝聚集合逐点赋予阿贝尔群结构，看起来只是更换值域，实际却立刻暴露出 sheaf
范畴与逐点代数的差别。核可以逐点计算，余核预层却可能不满足粘合；一个态射在
sheaf 意义下满射，只要求截面在某个覆盖后可提升，而不要求每个测试空间上的映射都
满射。若忽略这一区别，后面的正合列、投射对象和导出函子都会从定义处出错。

因此我们从第三章的凝聚站点出发，用阿贝尔群值 sheaf 定义
$\mathbf{CondAb}$，并借助一般 sheaf of abelian groups 的阿贝尔范畴结构辨认核、
余核、像与自由对象。极不连通空间为何能把局部满射变成逐点满射，以及由此得到的
投射生成元，将在第六、七章证明；这里先把这些结论所需的类型语言固定下来。

## 4.1 定义

**定义 4.1.** 凝聚阿贝尔群是站点

$$
(\mathbf{CHaus},J_{\operatorname{surj}})
$$

上的阿贝尔群值 sheaf。也就是说，它是反变函子

$$
A:\mathbf{CHaus}^{\operatorname{op}}\to \mathbf{Ab}
$$

满足：忘记阿贝尔群结构后，复合函子

$$
\mathbf{CHaus}^{\operatorname{op}}
\xrightarrow{A}
\mathbf{Ab}
\to
\mathbf{Set}
$$

是凝聚集合。

凝聚阿贝尔群范畴记为

$$
\mathbf{CondAb}.
$$

换句话说，对每个有限联合满射覆盖 $\{S_i\to S\}_{i=1}^n$，序列

$$
A(S)\longrightarrow \prod_i A(S_i)
\rightrightarrows
\prod_{i,j}A(S_i\times_S S_j)
$$

是 $\mathbf{Ab}$ 中的等化子。

**注 4.2.** 因为 $\mathbf{Ab}\to \mathbf{Set}$ 保持极限，所以上述等化子条件可以在集合层面检查；但群结构不是装饰。后续同调代数依赖的是 $\mathbf{CondAb}$ 的阿贝尔范畴结构，而不是单纯集合值 sheaf。

## 4.2 凝聚集合中的群对象

**命题 4.3.** 凝聚阿贝尔群等价于凝聚集合范畴 $\mathbf{CondSet}$ 中的阿贝尔群对象。

**证明.** 若 $A$ 是阿贝尔群值 sheaf，则每个 $S$ 上有阿贝尔群 $A(S)$。加法、零元和取负给出自然变换

$$
A\times A\to A,\qquad *\to A,\qquad A\to A,
$$

满足阿贝尔群公理。由于这些公理是交换图，它们在每个 $S$ 上成立，因而在 $\mathbf{CondSet}$ 中成立。

反过来，若 $X$ 是 $\mathbf{CondSet}$ 中的阿贝尔群对象，则对每个 $S$，集合 $X(S)$ 由结构映射获得阿贝尔群结构；对每个态射 $S'\to S$，限制映射与群运算相容，故得到函子 $\mathbf{CHaus}^{\operatorname{op}}\to\mathbf{Ab}$。sheaf 条件由 $X$ 作为凝聚集合已满足。两种构造互逆。证毕。

## 4.3 拓扑阿贝尔群给出的例子

**定义 4.4.** 设 $G$ 是拓扑阿贝尔群。定义

$$
\underline G(S)=\operatorname{Cont}(S,G),
\qquad S\in\mathbf{CHaus}.
$$

由于 $G$ 的加法和取负连续，$\operatorname{Cont}(S,G)$ 自然成为阿贝尔群，逐点加法为

$$
(\varphi+\psi)(s)=\varphi(s)+\psi(s).
$$

**命题 4.5.** 对任意拓扑阿贝尔群 $G$，$\underline G$ 是凝聚阿贝尔群。

**证明.** 第三章命题 3.7 已证明底层集合值预层是 sheaf。逐点加法、零元、取负与限制映射相容，因此这是阿贝尔群值 sheaf。证毕。

**例 4.6.** 若 $M$ 是普通阿贝尔群，并赋予离散拓扑，则 $\underline M$ 是凝聚阿贝尔群。对连通非空紧 Hausdorff 空间 $S$，有

$$
\underline M(S)\cong M.
$$

对有限离散空间 $S=\{1,\dots,n\}$，有

$$
\underline M(S)\cong M^n.
$$

## 4.4 核与余核

凝聚阿贝尔群范畴是阿贝尔范畴。这是 sheaf of abelian groups 的一般事实。我们在本章给出构造方式，后续再讨论更精细的 exactness 检查。

**定理 4.7.** $\mathbf{CondAb}$ 是阿贝尔范畴。

**证明路线（外部输入）.** 对任意站点 $(\mathcal C,J)$，阿贝尔群值 sheaf 范畴 $\operatorname{Sh}(\mathcal C,J;\mathbf{Ab})$ 是阿贝尔范畴。

具体地，若 $f:A\to B$ 是 sheaf 态射，则核可逐点定义：

$$
(\ker f)(S)=\ker(A(S)\to B(S)).
$$

因为极限与 sheaf 条件相容，$\ker f$ 仍是 sheaf。

余核先在预层范畴中逐点取：

$$
S\longmapsto \operatorname{coker}(A(S)\to B(S)),
$$

然后进行 sheafification。这样得到的 sheaf 是 $\mathbf{CondAb}$ 中的余核。一般阿贝尔范畴公理可由 sheafification 的正合性和预层阿贝尔范畴结构推出。完整证明属于 sheaf 理论标准结果。证毕。

**警告 4.8.** 余核不能总是简单地逐点计算后就结束；逐点余核可能只是预层，不一定已经是 sheaf。因此在 sheaf 范畴中，满射也不能简单理解为每个 $S$ 上的映射都满。

## 4.5 满射的局部性质

**定义 4.9.** 态射 $f:A\to B$ 在 $\mathbf{CondAb}$ 中是满射，如果它是阿贝尔范畴意义下的 epimorphism，等价于其余核为零。

一般 sheaf 理论告诉我们，$f$ 为满射意味着 $B$ 的局部截面可以局部提升到 $A$。更具体地，对 $b\in B(S)$，通常不能要求存在 $a\in A(S)$ 使得 $f(a)=b$；只能要求存在覆盖 $\{S_i\to S\}$，使得每个 $b|_{S_i}$ 来自某个 $a_i\in A(S_i)$。

这一区别在凝聚数学中极其重要。许多计算需要找到足够好的测试对象，使得“局部提升”能变成“全局提升”。极不连通空间正是在这里出现。

## 4.6 自由凝聚阿贝尔群

存在忘却函子

$$
U:\mathbf{CondAb}\to \mathbf{CondSet}.
$$

它把凝聚阿贝尔群忘记为底层凝聚集合。

**定义 4.10.** 若存在凝聚阿贝尔群 $\mathbb Z[X]$ 与态射

$$
X\to U(\mathbb Z[X])
$$

满足对任意 $A\in\mathbf{CondAb}$ 有自然双射

$$
\operatorname{Hom}_{\mathbf{CondAb}}(\mathbb Z[X],A)
\cong
\operatorname{Hom}_{\mathbf{CondSet}}(X,U(A)),
$$

则称 $\mathbb Z[X]$ 为由凝聚集合 $X$ 生成的自由凝聚阿贝尔群。

**注 4.11.** 一般 sheaf 范畴中，自由阿贝尔群对象可由“逐点自由阿贝尔群再 sheafification”构造。后续会特别研究 $X=\underline S$ 时的对象 $\mathbb Z[\underline S]$，它们是凝聚阿贝尔群的重要生成对象。

## 4.7 逐点核与局部余核

阿贝尔群值 sheaf 给出

$$
\mathbf{CondAb}
=
\operatorname{Sh}(\mathbf{CHaus},J_{\operatorname{surj}};\mathbf{Ab}).
$$

拓扑阿贝尔群通过连续映射函子进入其中，自由阿贝尔群对象则为后续生成构造提供左
伴随。关键区别也已经明确：核由逐点核给出，余核须 sheafification，满射只保证局部
提升。要把这种局部性重新化为可计算的逐点代数，需要寻找所有覆盖都能分裂的测试
空间；第五章先缩小测试站点，第六章再找到承担这一作用的极不连通空间。

## 练习

**练习 4.1.** 证明命题 4.3 中从群对象到阿贝尔群值 sheaf 的构造确实给出反变函子到 $\mathbf{Ab}$。

**练习 4.2.** 设 $G$ 是拓扑阿贝尔群。证明 $\operatorname{Cont}(S,G)$ 的逐点加法使 $\underline G(S)$ 成为阿贝尔群，且限制映射是群同态。

**练习 4.3.** 设 $f:A\to B$ 是凝聚阿贝尔群态射。证明逐点定义的 $\ker f$ 满足 sheaf 条件。

**练习 4.4.** 查阅一般 sheaf theory，给出一个 sheaf 满射不是逐点满射的例子。
