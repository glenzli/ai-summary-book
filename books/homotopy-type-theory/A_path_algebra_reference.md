# 附录 A：路径代数参考表

## 目标

本附录列出常用路径代数公式，供正文证明和证明蓝图引用。所有公式原则上可由路径归纳证明；若涉及函数路径，则需要函数外延性。

## A.1 基本路径公式

对 $p:x=y$、$q:y=z$、$r:z=w$：

- $p\cdot\mathsf{refl}_y=p$。
- $\mathsf{refl}_x\cdot p=p$。
- $(p\cdot q)\cdot r=p\cdot(q\cdot r)$。
- $p^{-1}\cdot p=\mathsf{refl}_y$。
- $p\cdot p^{-1}=\mathsf{refl}_x$。
- $(p^{-1})^{-1}=p$。
- $(p\cdot q)^{-1}=q^{-1}\cdot p^{-1}$。

## A.2 函数作用于路径

对 $f:A\to B$：

- $\mathsf{ap}_f(\mathsf{refl}_x)=\mathsf{refl}_{f(x)}$。
- $\mathsf{ap}_f(p^{-1})=(\mathsf{ap}_f(p))^{-1}$。
- $\mathsf{ap}_f(p\cdot q)=\mathsf{ap}_f(p)\cdot\mathsf{ap}_f(q)$。
- $\mathsf{ap}_{g\circ f}(p)=\mathsf{ap}_g(\mathsf{ap}_f(p))$。
- $\mathsf{ap}_{\mathsf{id}}(p)=p$。

## A.3 Transport

对 $P:A\to\mathcal U$：

- $\mathsf{transport}^{P}(\mathsf{refl}_x,u)=u$。
- $\mathsf{transport}^{P}(p^{-1},\mathsf{transport}^{P}(p,u))=u$。
- $\mathsf{transport}^{P}(q,\mathsf{transport}^{P}(p,u))=\mathsf{transport}^{P}(p\cdot q,u)$。

对依赖函数 $f:\prod_{x:A}P(x)$：

- $\mathsf{transport}^{P}(p,f(x))=f(y)$ 与 $\mathsf{apd}_f(p)$ 相关。

## A.4 $\Sigma$ 路径

对 $(a,b),(a',b'):\sum_{x:A}B(x)$，路径可由
$$
p:a=a'
$$
和
$$
\mathsf{transport}^{B}(p,b)=b'
$$
给出。完整地，有等价
$$
((a,b)=(a',b'))\simeq
\sum_{p:a=a'}\mathsf{transport}^{B}(p,b)=b'.
$$

## A.5 证明策略

- 若目标依赖一条路径，优先对该路径作路径归纳。
- 若目标是函数相等，先尝试构造逐点路径，再使用函数外延性。
- 若目标是 $\Sigma$ 类型中的路径，使用 $\Sigma$ 路径刻画。
- 若目标是等价保持性质，先考虑 fiber 可收缩定义，再考虑单值性 transport。
