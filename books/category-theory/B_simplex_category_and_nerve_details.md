# 附录 B：单纯形范畴与 nerve 细节

## 本章目标

本附录记录单纯形范畴 $\Delta$ 的生成元、单纯恒等式和普通范畴 nerve 的内角唯一填充证明。

## B.1 生成元

**定义 B.1.** 对 $0\le i\le n$，面映射

$$
\delta^i:[n-1]\to[n]
$$

是漏掉 $i$ 的保序单射。对 $0\le i\le n$，退化映射

$$
\sigma^i:[n+1]\to[n]
$$

是把 $i$ 与 $i+1$ 都送到 $i$ 的保序满射。

**命题 B.2.** $\Delta$ 中任意保序映射可唯一写成若干退化映射后接若干面映射的复合，并满足指标递增的规范形式。

**证明.** 设 $\alpha:[m]\to[n]$ 保序。令 $\operatorname{im}(\alpha)=\{r_0<\cdots<r_p\}$。存在唯一保序满射

$$
q:[m]\to[p]
$$

满足 $\alpha(i)=r_{q(i)}$，并存在唯一保序单射

$$
i:[p]\to[n],\qquad a\mapsto r_a
$$

使 $\alpha=i q$。保序单射 $i$ 由目标中漏掉的元素唯一决定，因此可唯一写成面映射的规范复合；保序满射 $q$ 由相邻元素被识别的位置唯一决定，因此可唯一写成退化映射的规范复合。两部分唯一，故整体规范分解唯一。$\square$

## B.2 单纯恒等式

**命题 B.3.** 面和退化映射满足：

$$
\delta^j\delta^i=\delta^i\delta^{j-1}\qquad (i<j),
$$

$$
\sigma^j\sigma^i=\sigma^i\sigma^{j+1}\qquad (i\le j),
$$

以及

$$
\sigma^j\delta^i=
\begin{cases}
\delta^i\sigma^{j-1},& i<j,\\
\operatorname{id},& i=j\text{ or }i=j+1,\\
\delta^{i-1}\sigma^j,& i>j+1.
\end{cases}
$$

**证明.** 每个等式都是两个保序映射之间的等式。逐点检查 $[n]$ 中元素的像即可。三类公式分别表达“漏掉两个元素的次序无关但指标需修正”“合并两个相邻对的次序无关但指标需修正”“先漏掉再合并的相互作用”。$\square$

## B.3 nerve 的内角唯一填充

**定理 B.4.** 对普通范畴 $\mathcal C$，$N(\mathcal C)$ 对所有内角 $\Lambda_i^n\to N(\mathcal C)$ 有唯一填充 $\Delta^n\to N(\mathcal C)$。

**证明.** 一个 $n$-单纯形 $\Delta^n\to N(\mathcal C)$ 等价于函子 $[n]\to\mathcal C$。这又等价于对象 $X_0,\dots,X_n$ 与态射 $f_{ab}:X_a\to X_b$，$0\le a\le b\le n$，满足

$$
f_{aa}=\operatorname{id}_{X_a},\qquad
f_{bc}\circ f_{ab}=f_{ac}.
$$

给定内角 $\Lambda_i^n\to N(\mathcal C)$，它给出除第 $i$ 个面外所有 $(n-1)$-面上的这类数据。特别地，它包含所有相邻边 $f_{a,a+1}$。定义所有长边为复合

$$
f_{ab}=f_{b-1,b}\circ\cdots\circ f_{a,a+1}.
$$

普通范畴的结合律保证这些复合满足 $f_{bc}f_{ab}=f_{ac}$。由此得到一个函子 $[n]\to\mathcal C$，即填充。

唯一性：任意填充在相邻边上必须等于给定内角数据；而函子 $[n]\to\mathcal C$ 的所有长边由相邻边的复合唯一决定。因此填充唯一。$\square$

## B.4 本章小结

单纯恒等式是单纯集计算的代数基础。普通范畴 nerve 的内角唯一填充说明：quasi-category 是普通范畴的弱化，其中复合存在但不再严格唯一。

## 练习

**练习 B.1.** 对 $[2]\to[4]$ 的一个保序映射写出满射-单射分解。

**练习 B.2.** 逐点验证公式 $\delta^j\delta^i=\delta^i\delta^{j-1}$。

**练习 B.3.** 对 $n=3,i=1$ 手工画出 $\Lambda_1^3$ 的缺失面。
