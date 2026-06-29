# 附录 J：一元代数签名的结构等同性

## 目标

本附录把附录 I 的结构等同性原则专门化到一元代数签名。这样，群、幺半群、环的乘法部分、格、半环等常见一底层类型结构都可用同一套证明：对象路径等价于底层等价保持全部运算，并且命题性公理不会增加额外数据。

本附录仍不处理多载体结构、依赖结构和高阶范畴结构；那些应在单值范畴论卷中单独展开。

## J.1 有限元组

**定义 J.1.** 对自然数 $n$ 和类型 $A$，记
$$
A^n\coloneqq \mathsf{Fin}(n)\to A.
$$
这里 $\mathsf{Fin}(n)$ 是有 $n$ 个元素的有限类型。

**定义 J.2.** 若 $e:A\simeq B$，定义
$$
e^{(n)}:A^n\simeq B^n
$$
为逐点作用：
$$
e^{(n)}(\vec a)(i)\coloneqq e(\vec a(i)).
$$
其逆由 $e^{-1}$ 逐点给出。

**证明.** 等价的逆和复合已在附录 G 中建立；逐点左右逆由函数外延性和 $e$ 的左右逆给出。$\square$

## J.2 一元代数签名

**定义 J.3（一元代数签名）.** 一个一元代数签名 $\Sigma$ 由类型 $\mathsf{Op}_\Sigma$ 和 arity 函数
$$
\mathsf{ar}:\mathsf{Op}_\Sigma\to\mathbb N
$$
组成。

**定义 J.4（代数结构）.** 对类型 $A:\mathcal U$，$\Sigma$-运算结构定义为
$$
\mathsf{AlgOps}_\Sigma(A)
\coloneqq
\prod_{\omega:\mathsf{Op}_\Sigma}(A^{\mathsf{ar}(\omega)}\to A).
$$

若还给定命题性公理族
$$
\mathsf{Law}_\Sigma:\prod_{A:\mathcal U}\mathsf{AlgOps}_\Sigma(A)\to\mathcal U
$$
并满足每个 $\mathsf{Law}_\Sigma(A,\alpha)$ 是命题，则完整代数结构为
$$
\mathsf{Alg}_\Sigma
\coloneqq
\sum_{A:\mathcal U}\sum_{\alpha:\mathsf{AlgOps}_\Sigma(A)}
\mathsf{Law}_\Sigma(A,\alpha).
$$

## J.3 同态与同构

**定义 J.5.** 给定 $(A,\alpha)$ 与 $(B,\beta)$，函数 $f:A\to B$ 保持运算，若
$$
\mathsf{Pres}_\Sigma(f,\alpha,\beta)
\coloneqq
\prod_{\omega:\mathsf{Op}_\Sigma}\prod_{\vec a:A^{\mathsf{ar}(\omega)}}
f(\alpha_\omega(\vec a))
=
\beta_\omega(\lambda i.\,f(\vec a(i))).
$$

**定义 J.6.** $\Sigma$-代数同构由底层等价和运算保持性组成：
$$
(A,\alpha)\cong_\Sigma(B,\beta)
\coloneqq
\sum_{e:A\simeq B}\mathsf{Pres}_\Sigma(e,\alpha,\beta).
$$
若结构还包含命题性公理，公理证明分量不进入同构数据。

## J.4 Transport 的计算

**定理 J.7（运算沿等价的 transport）.** 假设单值性。设 $e:A\simeq B$，$\alpha:\mathsf{AlgOps}_\Sigma(A)$。沿路径 $\mathsf{ua}(e):A=B$ transport 得到的运算
$$
\mathsf{transport}^{\mathsf{AlgOps}_\Sigma}(\mathsf{ua}(e),\alpha)
:\mathsf{AlgOps}_\Sigma(B)
$$
满足：对每个 $\omega$ 和 $\vec b:B^{\mathsf{ar}(\omega)}$，
$$
\bigl(\mathsf{transport}(\mathsf{ua}(e),\alpha)\bigr)_\omega(\vec b)
=
e\left(\alpha_\omega(\lambda i.\,e^{-1}(\vec b(i)))\right).
$$

**证明.** 对等价 $e$ 使用等价归纳。也就是由单值性给出的
$$
(A=B)\simeq(A\simeq B)
$$
把命题化到 $e$ 为恒等等价的情形。恒等情形中，$\mathsf{ua}(\mathsf{idEquiv})$ 与 $\mathsf{refl}_A$ 相容，transport 按反身路径计算为自身，右边也按恒等等价和其逆化简为 $\alpha_\omega(\vec b)$。$\square$

**说明 J.8.** 若采用公理化 HoTT，定理 J.7 的等式是路径而非 judgmental equality。若采用 cubical type theory，某些 transport 计算可具有更强的计算行为。

## J.5 一元代数 SIP

**定理 J.9（一元代数 SIP）.** 假设单值性与函数外延性。对任意一元代数签名 $\Sigma$ 和命题性公理族 $\mathsf{Law}_\Sigma$，两个 $\Sigma$-代数对象
$$
(A,\alpha,\ell_A),\qquad(B,\beta,\ell_B)
$$
的路径等价于底层代数同构：
$$
((A,\alpha,\ell_A)=(B,\beta,\ell_B))
\simeq
\sum_{e:A\simeq B}\mathsf{Pres}_\Sigma(e,\alpha,\beta).
$$

**证明.** 先对总结构使用附录 I.3 的精确 SIP。得到路径等价于
$$
\sum_{e:A\simeq B}
\mathsf{transport}^{S}(\mathsf{ua}(e),(\alpha,\ell_A))=(\beta,\ell_B),
$$
其中 $S(A)\coloneqq\sum_{\alpha:\mathsf{AlgOps}_\Sigma(A)}\mathsf{Law}_\Sigma(A,\alpha)$。

由 $\Sigma$ 路径刻画，该结构相等由两个分量组成：

1.  运算分量相等；
2.  公理证明分量相等。

第二分量位于命题 $\mathsf{Law}_\Sigma(B,\beta)$ 中，因此由命题性自动给出。第一分量由函数外延性展开为对每个 $\omega$ 和输入元组的等式。再用定理 J.7 展开 transport，正得到 $\mathsf{Pres}_\Sigma(e,\alpha,\beta)$。$\square$

## J.6 群和幺半群

**例 J.10（幺半群）.** 幺半群签名有一个二元运算 $\mu$ 和一个零元运算 $e$。公理为结合律和左右单位律。由于这些公理是集合层命题，定理 J.9 给出：幺半群对象路径等价于幺半群同构。

**例 J.11（群）.** 群签名有二元乘法、单位常元和一元逆运算，公理为结合律、单位律和逆元律，并包含底层类型是集合的证明。所有公理分量为命题，故定理 J.9 给出群对象路径等价于群同构。
