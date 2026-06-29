# 附录 AG：结构 transport 与代数 SIP 证明核

本附录补全附录 I.6、I.7 和 P.11 的证明细节。核心思想是：沿单值性路径 $\mathsf{ua}(e)$ transport 代数结构，等价于用 $e$ 把结构从源类型共轭到目标类型。

设 $e:A\simeq B$，其底层函数仍记为 $e:A\to B$，逆函数记为 $e^{-1}:B\to A$。

## AG.1 常元与运算的 transport

**引理 AG.1（常元 transport）.** 对结构族 $S(X)\coloneqq X$，有
$$
\mathsf{transport}^{S}(\mathsf{ua}(e),a)=e(a).
$$

**证明.** 对等价 $e$ 作等价归纳。由单值性，把问题化到 $e$ 为恒等等价，$\mathsf{ua}(e)$ 为反身路径。此时 transport judgmentally 为恒等，结论为 reflexivity。$\square$

**引理 AG.2（一元运算 transport）.** 对结构族 $S(X)\coloneqq X\to X$，若 $f:A\to A$，则
$$
\mathsf{transport}^{S}(\mathsf{ua}(e),f)
=
\lambda b.\,e(f(e^{-1}(b))).
$$

**证明.** 对 $e$ 作等价归纳。恒等等价情形中右边化为 $\lambda b.\,f(b)$，由函数外延性与反身路径得到。$\square$

**引理 AG.3（二元运算 transport）.** 对结构族 $S(X)\coloneqq X\to X\to X$，若 $\mu:A\to A\to A$，则
$$
\mathsf{transport}^{S}(\mathsf{ua}(e),\mu)
=
\lambda b_1\,b_2.\,e(\mu(e^{-1}(b_1),e^{-1}(b_2))).
$$

**证明.** 对 $e$ 作等价归纳。恒等等价情形中两边均为 $\mu$；由两次函数外延性得到函数路径。$\square$

**引理 AG.4（有限 arity 运算 transport）.** 对任意有限 arity $k$ 的运算结构
$$
S(X)\coloneqq X^k\to X,
$$
沿 $\mathsf{ua}(e)$ 的 transport 把运算 $\omega:A^k\to A$ 送到
$$
(b_1,\ldots,b_k)\mapsto
e(\omega(e^{-1}b_1,\ldots,e^{-1}b_k)).
$$

**证明.** 对 $k$ 归纳，或把 $X^k\to X$ 展开为迭代函数类型后重复 AG.1-AG.3 的等价归纳证明。$\square$

## AG.2 有限代数签名

**定义 AG.5（命题性公理代数签名）.** 一个有限代数签名由有限个常元、有限 arity 运算和公理族组成，其中每条公理在给定底层类型与运算解释后是命题。

对签名 $\Sigma$，记结构族为
$$
\mathsf{Alg}_{\Sigma}(A).
$$
它由有限 $\Sigma$-类型和有限乘积组成：
$$
\sum_{\text{常元解释}}\sum_{\text{运算解释}}\mathsf{Axioms}(A,\text{解释}).
$$

**定义 AG.6（传统结构同构）.** 两个 $\Sigma$-代数 $(A,s)$ 与 $(B,t)$ 的传统同构由等价 $e:A\simeq B$ 加上如下保持条件组成：

1.  每个常元 $c_A$ 满足 $e(c_A)=c_B$；
2.  每个 $k$ 元运算 $\omega_A$ 满足
    $$
    e(\omega_A(a_1,\ldots,a_k))
    =
    \omega_B(e(a_1),\ldots,e(a_k));
    $$
3.  公理证明不作为额外数据比较。

**定理 AG.7（传统同构等价于规范结构等价）.** 对命题性公理代数签名 $\Sigma$，传统结构同构等价于附录 I.2 的规范结构等价
$$
\sum_{e:A\simeq B}
\mathsf{transport}^{\mathsf{Alg}_{\Sigma}}(\mathsf{ua}(e),s)=t.
$$

**证明.** 展开 $\mathsf{Alg}_{\Sigma}$ 的有限 $\Sigma$-类型。由 $\Sigma$ 路径刻画，结构相等逐项分解为常元、运算和公理分量的相等：

1.  常元分量由 AG.1 化为 $e(c_A)=c_B$；
2.  运算分量由 AG.4 化为传统的保持运算公式；公式方向可由对等价 $e$ 作用或用 $e^{-1}$ 改写互相转换；
3.  公理分量位于命题中，由 I.5 自动消去。

有限乘积和有限 $\Sigma$ 的路径比较反复使用附录 D.6 的 $\Sigma$ 路径刻画。由此得到传统同构与规范结构等价的双向构造；互逆性逐项由路径归纳和公理分量命题性给出。$\square$

## AG.3 结构范畴单值性

**定理 AG.8（命题性公理代数结构范畴单值性）.** 设 $\mathsf{AlgCat}_{\Sigma}$ 是命题性公理代数签名 $\Sigma$ 的结构范畴：对象为 $\Sigma$-代数，态射为保持全部常元和运算的函数。若同构定义为底层等价加保持结构条件，则 $\mathsf{AlgCat}_{\Sigma}$ 是单值范畴。

**证明.** 对对象 $X=(A,s)$、$Y=(B,t)$，附录 I.3 给出对象路径与规范结构等价的等价：
$$
(X=Y)\simeq
\sum_{e:A\simeq B}
\mathsf{transport}^{\mathsf{Alg}_{\Sigma}}(\mathsf{ua}(e),s)=t.
$$
由 AG.7，右侧等价于传统结构同构。范畴同构类型 $X\cong Y$ 也等价于传统结构同构：一个范畴同构给出保持结构的函数及其保持结构的逆，因而底层函数是等价；反向由底层等价及结构保持条件给出保持结构的逆函数。保持结构证明和逆律证明的高阶相容位于 Hom 集合的路径类型或命题性公理分量中，因此由集合性/命题性消去。

上述复合等价在反身路径上把 $\mathsf{refl}_X$ 送到恒等结构同构。由路径归纳，它与 $\mathsf{idtoiso}_{X,Y}$ 相同。因此 $\mathsf{idtoiso}_{X,Y}$ 是等价，$\mathsf{AlgCat}_{\Sigma}$ 单值。$\square$

**推论 AG.9（群范畴单值性）.** 若群定义为集合、乘法、单位、逆和命题性群律，则群范畴是单值范畴。

**证明.** 群是命题性公理代数签名的实例；应用 AG.8。$\square$
