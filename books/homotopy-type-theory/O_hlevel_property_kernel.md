# 附录 O：同伦层级性质的命题性

## 目标

本附录补齐附录 N.9 使用的标准事实：对任意固定同伦层级 $n$，类型 $A$ 具有 $h$-level $n$ 这一性质本身是命题。证明使用函数外延性；不使用单值性。

## O.1 预备引理

**引理 O.1（依赖函数命题稳定性）.** 若 $B:A\to\mathcal U$ 且
$$
\prod_{a:A}\mathsf{isProp}(B(a)),
$$
则
$$
\mathsf{isProp}\Bigl(\prod_{a:A}B(a)\Bigr).
$$

**证明.** 这是附录 F.5。给定两个函数 $f,g:\prod_{a:A}B(a)$，逐点由 $B(a)$ 的命题性得到 $f(a)=g(a)$，再由函数外延性得到 $f=g$。$\square$

**引理 O.2（可收缩性是命题）.** 对任意 $A:\mathcal U$，
$$
\mathsf{isProp}(\mathsf{isContr}(A)).
$$

**证明.** 这是附录 D.11。证明用 $\Sigma$ 路径刻画比较两个收缩中心和两个收缩同伦，后者使用函数外延性。$\square$

## O.2 一般同伦层级

回忆定义 4.12：
$$
\mathsf{isOfHLevel}_0(A)\coloneqq\mathsf{isContr}(A),
$$
$$
\mathsf{isOfHLevel}_{n+1}(A)
\coloneqq
\prod_{x,y:A}\mathsf{isOfHLevel}_n(x=y).
$$

**定理 O.3（同伦层级性质是命题）.** 对任意 $n:\mathbb N$ 和 $A:\mathcal U$，
$$
\mathsf{isProp}(\mathsf{isOfHLevel}_n(A)).
$$

**证明.** 对 $n$ 作自然数归纳。

基步 $n\equiv0$：此时
$$
\mathsf{isOfHLevel}_0(A)\equiv\mathsf{isContr}(A),
$$
由引理 O.2 得到命题性。

归纳步：假设对任意类型 $X$，有
$$
\mathsf{isProp}(\mathsf{isOfHLevel}_n(X)).
$$
需证
$$
\mathsf{isProp}(\mathsf{isOfHLevel}_{n+1}(A)).
$$
展开定义，目标为
$$
\mathsf{isProp}
\Bigl(
\prod_{x,y:A}\mathsf{isOfHLevel}_n(x=y)
\Bigr).
$$
先对固定 $x:A$ 应用引理 O.1：对每个 $y:A$，归纳假设作用于类型 $x=y$ 给出
$$
\mathsf{isProp}(\mathsf{isOfHLevel}_n(x=y)).
$$
因此
$$
\prod_{y:A}\mathsf{isOfHLevel}_n(x=y)
$$
是命题。再对 $x:A$ 应用引理 O.1，得到整个双重依赖函数类型是命题。$\square$

**推论 O.4（isSet 是命题）.** 对任意 $A:\mathcal U$，
$$
\mathsf{isProp}(\mathsf{isSet}(A)).
$$

**证明.** $\mathsf{isSet}(A)$ 是 $\mathsf{isOfHLevel}_2(A)$ 的展开。由定理 O.3 取 $n=2$。$\square$

## O.3 对附录 N 的应用

**命题 O.5（code fibers 是集合的相干性）.** 附录 N.9 中，圆归纳证明
$$
\prod_{x:\mathbb S^1}\mathsf{isSet}(\mathsf{code}(x))
$$
的路径构造子相干由推论 O.4 给出。

**证明.** 基点处由附录 M.5 得到 $\mathsf{isSet}(\mathbb Z)$。沿 $\mathsf{loop}$ 的 transport 后，目标仍是 $\mathsf{isSet}(\mathbb Z)$ 中两个元素相等；由推论 O.4，该类型是命题，所以任意两项相等。$\square$
