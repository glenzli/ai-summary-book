# 附录 V：圆的基本群同构

## V.0 目标

附录 N 已证明 loop space 等价：
$$
(\mathsf{base}=\mathsf{base})\simeq\mathbb Z.
$$
本附录补齐群结构层面的相容性，证明基本群
$$
\pi_1(\mathbb S^1,\mathsf{base})
$$
与整数加法群同构。

本附录使用：

1. 附录 M 的归纳整数、加法和 loop 幂；
2. 附录 N 的 encode-decode 等价；
3. 集合截断的递归原则；
4. 路径复合的结合律、单位律和逆律。

## V.1 基本群运算

**定义 V.1（基本群乘法）。** 对基点类型 $(X,x_0)$，定义
$$
\pi_1(X,x_0)\coloneqq\|x_0=x_0\|_0.
$$
乘法为集合截断递归诱导的运算
$$
[p]\cdot[q]\coloneqq[p\cdot q],
$$
单位为 $[\mathsf{refl}_{x_0}]$，逆元为 $[p]^{-1}\coloneqq[p^{-1}]$。

**命题 V.2（基本群律）。** $\pi_1(X,x_0)$ 是群。

**证明.** 每条群律由 loop space 中对应路径代数律下降到集合截断。以结合律为例，目标
$$
([p]\cdot[q])\cdot[r]=[p]\cdot([q]\cdot[r])
$$
经集合截断归纳化为
$$
[(p\cdot q)\cdot r]=[p\cdot(q\cdot r)],
$$
由路径代数结合律给出。左右单位律和逆律同理，分别使用路径复合单位律和逆律。由于目标是集合截断中的路径，递归/归纳合法性由集合截断的消去原则保证。$\square$

## V.2 loop 幂与 successor/predecessor

本节记
$$
\mathsf{pow}(z)\coloneqq\mathsf{loop}^z:\mathsf{base}=\mathsf{base}
$$
为定义 M.15 的 loop 幂。

**引理 V.3（successor 作用）。** 对任意 $z:\mathbb Z$，
$$
\mathsf{pow}(\mathsf{succ}_{\mathbb Z}(z))
=
\mathsf{pow}(z)\cdot\mathsf{loop}.
$$

**证明.** 对 $z$ 作归纳整数消去。

若 $z\equiv\mathsf{pos}(n)$，则分两种情形。$n\equiv0$ 时，两边分别为 $\mathsf{loop}$ 和 $\mathsf{refl}\cdot\mathsf{loop}$，由左单位律得证。$n\equiv\mathsf{succ}(k)$ 时，结论按 $\mathsf{pow}$ 定义化为反身路径。

若 $z\equiv\mathsf{neg}(0)$，则左边为 $\mathsf{pow}(\mathsf{pos}(0))\equiv\mathsf{refl}$，右边为 $\mathsf{loop}^{-1}\cdot\mathsf{loop}$，由逆律得证。

若 $z\equiv\mathsf{neg}(\mathsf{succ}(k))$，左边为 $\mathsf{pow}(\mathsf{neg}(k))$，右边展开为
$$
(\mathsf{pow}(\mathsf{neg}(k))\cdot\mathsf{loop}^{-1})\cdot\mathsf{loop}.
$$
由结合律和逆律化为 $\mathsf{pow}(\mathsf{neg}(k))$。$\square$

**引理 V.4（predecessor 作用）。** 对任意 $z:\mathbb Z$，
$$
\mathsf{pow}(\mathsf{pred}_{\mathbb Z}(z))
=
\mathsf{pow}(z)\cdot\mathsf{loop}^{-1}.
$$

**证明.** 对 $z$ 作归纳整数消去。

若 $z\equiv\mathsf{pos}(0)$，左边为 $\mathsf{pow}(\mathsf{neg}(0))=\mathsf{loop}^{-1}$，右边为 $\mathsf{refl}\cdot\mathsf{loop}^{-1}$，由左单位律得证。

若 $z\equiv\mathsf{pos}(\mathsf{succ}(k))$，左边为 $\mathsf{pow}(\mathsf{pos}(k))$，右边为
$$
(\mathsf{pow}(\mathsf{pos}(k))\cdot\mathsf{loop})\cdot\mathsf{loop}^{-1}.
$$
由结合律和逆律化简。

若 $z\equiv\mathsf{neg}(k)$，结论按 $\mathsf{pow}$ 和 $\mathsf{pred}_{\mathbb Z}$ 的定义化为反身路径。$\square$

## V.3 loop 幂保持加法

**定理 V.5（loop 幂与加法相容）。** 对任意 $z,w:\mathbb Z$，
$$
\mathsf{pow}(z+w)=\mathsf{pow}(z)\cdot\mathsf{pow}(w).
$$

**证明.** 对第二个变量 $w$ 作归纳整数消去。

若 $w\equiv\mathsf{pos}(0)$，则 $z+w\equiv z$，且
$$
\mathsf{pow}(\mathsf{pos}(0))\equiv\mathsf{refl}.
$$
目标化为
$$
\mathsf{pow}(z)=\mathsf{pow}(z)\cdot\mathsf{refl},
$$
由右单位律得证。

若 $w\equiv\mathsf{pos}(\mathsf{succ}(n))$，由定义 M.13，
$$
z+w\equiv\mathsf{succ}_{\mathbb Z}(z+\mathsf{pos}(n)).
$$
于是由引理 V.3，
$$
\mathsf{pow}(z+w)
=
\mathsf{pow}(z+\mathsf{pos}(n))\cdot\mathsf{loop}.
$$
用归纳假设得到
$$
(\mathsf{pow}(z)\cdot\mathsf{pow}(\mathsf{pos}(n)))\cdot\mathsf{loop}.
$$
由结合律，这等于
$$
\mathsf{pow}(z)\cdot
(\mathsf{pow}(\mathsf{pos}(n))\cdot\mathsf{loop})
=
\mathsf{pow}(z)\cdot\mathsf{pow}(\mathsf{pos}(\mathsf{succ}(n))).
$$

若 $w\equiv\mathsf{neg}(0)$，则 $z+w\equiv\mathsf{pred}_{\mathbb Z}(z)$。由引理 V.4，
$$
\mathsf{pow}(z+w)
=
\mathsf{pow}(z)\cdot\mathsf{loop}^{-1}
=
\mathsf{pow}(z)\cdot\mathsf{pow}(\mathsf{neg}(0)).
$$

若 $w\equiv\mathsf{neg}(\mathsf{succ}(n))$，由定义 M.13，
$$
z+w\equiv\mathsf{pred}_{\mathbb Z}(z+\mathsf{neg}(n)).
$$
由引理 V.4 和归纳假设，
$$
\mathsf{pow}(z+w)
=
(\mathsf{pow}(z)\cdot\mathsf{pow}(\mathsf{neg}(n)))\cdot\mathsf{loop}^{-1}.
$$
再由结合律化为
$$
\mathsf{pow}(z)\cdot
(\mathsf{pow}(\mathsf{neg}(n))\cdot\mathsf{loop}^{-1})
=
\mathsf{pow}(z)\cdot\mathsf{pow}(\mathsf{neg}(\mathsf{succ}(n))).
$$
四种情形完成。$\square$

## V.4 从 loop space 等价到基本群同构

**定义 V.6（整数到基本群）。** 定义
$$
\Phi:\mathbb Z\to\pi_1(\mathbb S^1,\mathsf{base})
$$
为
$$
\Phi(z)\coloneqq[\mathsf{pow}(z)].
$$

**定义 V.7（基本群到整数）。** 定义
$$
\Psi:\pi_1(\mathbb S^1,\mathsf{base})\to\mathbb Z
$$
为集合截断递归：
$$
\Psi([p])\coloneqq\mathsf{encode}_{\mathsf{base}}(p).
$$
这是合法的，因为 $\mathbb Z$ 是集合，见命题 M.5。

**命题 V.8（$\Phi$ 与 $\Psi$ 互逆）。** $\Phi$ 与 $\Psi$ 互为逆函数。

**证明.** 对 $z:\mathbb Z$，
$$
\Psi(\Phi(z))
=
\mathsf{encode}_{\mathsf{base}}(\mathsf{pow}(z))
=z
$$
由引理 N.6。

对 $[p]:\pi_1(\mathbb S^1,\mathsf{base})$，由集合截断归纳归约到 $p:\mathsf{base}=\mathsf{base}$。需证
$$
[\mathsf{pow}(\mathsf{encode}_{\mathsf{base}}(p))]=[p].
$$
由定理 N.8 的 decode-after-encode，
$$
\mathsf{decode}_{\mathsf{base}}(\mathsf{encode}_{\mathsf{base}}(p))=p.
$$
而 $\mathsf{decode}_{\mathsf{base}}(n)=\mathsf{pow}(n)$，故得到所需路径，并应用集合截断构造子。$\square$

**命题 V.9（$\Phi$ 保乘法）。** 对任意 $z,w:\mathbb Z$，
$$
\Phi(z+w)=\Phi(z)\cdot\Phi(w).
$$

**证明.** 左边为
$$
[\mathsf{pow}(z+w)],
$$
右边为
$$
[\mathsf{pow}(z)\cdot\mathsf{pow}(w)].
$$
由定理 V.5 给出括号内路径，应用集合截断构造子即得。$\square$

**定理 V.10（圆的基本群）。** 整数加法群与圆的基本群同构：
$$
(\mathbb Z,+,0,-)\cong \pi_1(\mathbb S^1,\mathsf{base}).
$$

**证明.** 由命题 V.8，$\Phi$ 是底层类型等价。由命题 V.9，$\Phi$ 保持乘法。单位保持可直接由
$$
\Phi(0)=[\mathsf{refl}_{\mathsf{base}}]
$$
给出。逆元保持不需另作归纳：在任意群中，若 $F$ 保持乘法和单位，则
$$
F(z^{-1})\cdot F(z)=F(z^{-1}+z)=F(0)=e
$$
且同理 $F(z)\cdot F(z^{-1})=e$，由逆元唯一性得到 $F(z^{-1})=F(z)^{-1}$。故 $\Phi$ 是群同构。$\square$

## V.5 边界

本附录依赖附录 W 对 M.14 的整数加法群律证明核。换言之，附录 W 证明整数对象自身形成交换群；本附录证明附录 N 的 encode-decode 等价与群运算相容，并由此得到圆的基本群同构。剩余边界仅是附录 N.8 所述的 propositional HIT computation 机器化插入问题。
