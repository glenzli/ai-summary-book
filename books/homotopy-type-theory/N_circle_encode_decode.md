# 附录 N：圆的 Encode-Decode 证明核

## 目标

本附录把第十一章的定理
$$
(\mathsf{base}=\mathsf{base})\simeq\mathbb Z
$$
拆成可核查的类型论构造。所用输入为：

1.  圆的 HIT 规则，见附录 L.14-L.16；
2.  归纳整数、successor 自等价和 loop 幂，见附录 M；
3.  函数外延性和单值性，见第六章；
4.  路径代数，见第二章和附录 A。

本附录默认
$$
\mathbb Z\coloneqq\mathbb Z_{\mathsf{ind}},
\qquad
0_{\mathbb Z}\coloneqq\mathsf{pos}(0).
$$

## N.1 单值性的 transport 计算

**引理 N.1（univalence transport 计算）.** 设 $A,B:\mathcal U$，$e:A\simeq B$。若
$$
\mathsf{ua}(e):A=B
$$
是单值性给出的路径，则
$$
\mathsf{transport}^{\lambda X.X}(\mathsf{ua}(e),a)=e(a)
$$
对任意 $a:A$ 成立。

**证明.** $\mathsf{idtoequiv}_{A,B}:(A=B)\to(A\simeq B)$ 的底层函数正是沿宇宙路径的 transport。单值性给出 $\mathsf{idtoequiv}$ 的逆方向 $\mathsf{ua}$，并有
$$
\mathsf{idtoequiv}(\mathsf{ua}(e))=e.
$$
展开 $\mathsf{idtoequiv}$ 的定义即得该 transport 计算。若采用 cubical type theory，该引理由 Glue/univalence 的计算规则给出。$\square$

## N.2 code 覆盖

**定义 N.2（圆的整数覆盖）.** 令
$$
s\coloneqq\mathsf{succEquiv}_{\mathbb Z}:\mathbb Z\simeq\mathbb Z.
$$
用圆递归原则 L.15 在宇宙 $\mathcal U$ 中定义
$$
\mathsf{code}:\mathbb S^1\to\mathcal U
$$
使基点数据为 $\mathbb Z$，路径数据为 $\mathsf{ua}(s)$。即有计算路径
$$
c_0:\mathsf{code}(\mathsf{base})=\mathbb Z,
$$
$$
c_\ell:\mathsf{ap}_{\mathsf{code}}(\mathsf{loop})
=
\mathsf{ua}(s).
$$
在采用 judgmental 点计算的 HIT 口径下可把 $c_0$ 当作定义相等；本附录公式为可读性把 $\mathsf{code}(\mathsf{base})$ 直接写作 $\mathbb Z$。若系统只给 propositional computation，所有基点处公式需插入 $c_0$ 的 transport。

**引理 N.3（沿 loop 的 transport）.** 对任意 $n:\mathbb Z$，
$$
\mathsf{transport}^{\mathsf{code}}(\mathsf{loop},n)
=
\mathsf{succ}_{\mathbb Z}(n),
$$
并且
$$
\mathsf{transport}^{\mathsf{code}}(\mathsf{loop}^{-1},n)
=
\mathsf{pred}_{\mathbb Z}(n).
$$

**证明.** 第一条由 $c_\ell$、引理 N.1 和 $s$ 的底层函数为 $\mathsf{succ}_{\mathbb Z}$ 得到。第二条使用宇宙路径的逆：
$$
\mathsf{ap}_{\mathsf{code}}(\mathsf{loop}^{-1})
=
\mathsf{ua}(s)^{-1}.
$$
沿逆路径的 transport 是沿原路径 transport 的逆函数；由附录 M.8，$\mathsf{pred}_{\mathbb Z}$ 是 $\mathsf{succ}_{\mathbb Z}$ 的逆。展开后得到所需等式。$\square$

## N.3 encode

**定义 N.4（encode）.** 对 $x:\mathbb S^1$ 和 $p:\mathsf{base}=x$，定义
$$
\mathsf{encode}_x(p)
\coloneqq
\mathsf{transport}^{\mathsf{code}}(p,0_{\mathbb Z})
:
\mathsf{code}(x).
$$
因此
$$
\mathsf{encode}:
\prod_{x:\mathbb S^1}(\mathsf{base}=x)\to\mathsf{code}(x).
$$

## N.4 loop 幂的 transport 计算

记附录 M.15 给出的 loop 幂为
$$
\mathsf{pow}(n)\coloneqq\mathsf{loop}^{n}:\mathsf{base}=\mathsf{base}.
$$

**引理 N.5（predecessor 与右复合 loop）.** 对任意 $n:\mathbb Z$，
$$
\mathsf{pow}(\mathsf{pred}_{\mathbb Z}(n))\cdot\mathsf{loop}
=
\mathsf{pow}(n).
$$

**证明.** 对 $n$ 作归纳整数消去。

若 $n\equiv\mathsf{pos}(0)$，左边为
$$
\mathsf{loop}^{-1}\cdot\mathsf{loop},
$$
由路径逆律等于 $\mathsf{refl}_{\mathsf{base}}=\mathsf{pow}(\mathsf{pos}(0))$。

若 $n\equiv\mathsf{pos}(\mathsf{succ}(k))$，则
$$
\mathsf{pred}_{\mathbb Z}(n)\equiv\mathsf{pos}(k),
$$
目标按 $\mathsf{pow}$ 的定义化为反身路径。

若 $n\equiv\mathsf{neg}(k)$，则
$$
\mathsf{pred}_{\mathbb Z}(n)\equiv\mathsf{neg}(\mathsf{succ}(k)),
$$
左边展开为
$$
(\mathsf{pow}(\mathsf{neg}(k))\cdot\mathsf{loop}^{-1})\cdot\mathsf{loop}.
$$
由结合律和逆律化为 $\mathsf{pow}(\mathsf{neg}(k))$。$\square$

**引理 N.6（loop 幂的 winding number）.** 对任意 $n:\mathbb Z$，
$$
\mathsf{encode}_{\mathsf{base}}(\mathsf{pow}(n))=n.
$$

**证明.** 按 $n$ 作归纳整数消去。

正半轴基步 $n\equiv\mathsf{pos}(0)$ 中，
$$
\mathsf{pow}(n)\equiv\mathsf{refl}_{\mathsf{base}},
$$
故 transport 为 $0_{\mathbb Z}$。

正半轴归纳步中，设结论对 $\mathsf{pos}(k)$ 成立。由于
$$
\mathsf{pow}(\mathsf{pos}(\mathsf{succ}(k)))
\equiv
\mathsf{pow}(\mathsf{pos}(k))\cdot\mathsf{loop},
$$
transport 的复合律给出
$$
\mathsf{encode}_{\mathsf{base}}(\mathsf{pow}(\mathsf{pos}(\mathsf{succ}(k))))
=
\mathsf{transport}^{\mathsf{code}}
(\mathsf{loop},
\mathsf{encode}_{\mathsf{base}}(\mathsf{pow}(\mathsf{pos}(k)))).
$$
用归纳假设和引理 N.3 化为
$$
\mathsf{succ}_{\mathbb Z}(\mathsf{pos}(k))
=
\mathsf{pos}(\mathsf{succ}(k)).
$$

负半轴基步 $n\equiv\mathsf{neg}(0)$ 中，$\mathsf{pow}(n)\equiv\mathsf{loop}^{-1}$，由引理 N.3 的逆 loop 计算得
$$
\mathsf{encode}_{\mathsf{base}}(\mathsf{loop}^{-1})
=
\mathsf{pred}_{\mathbb Z}(0_{\mathbb Z})
=
\mathsf{neg}(0).
$$

负半轴归纳步中，
$$
\mathsf{pow}(\mathsf{neg}(\mathsf{succ}(k)))
\equiv
\mathsf{pow}(\mathsf{neg}(k))\cdot\mathsf{loop}^{-1}.
$$
transport 复合律、归纳假设和引理 N.3 的逆 loop 计算给出
$$
\mathsf{pred}_{\mathbb Z}(\mathsf{neg}(k))
=
\mathsf{neg}(\mathsf{succ}(k)).
$$
$\square$

## N.5 decode

**定义 N.7（decode）.** 定义类型族
$$
D(x)\coloneqq\mathsf{code}(x)\to(\mathsf{base}=x).
$$
在基点处取
$$
d_0:\mathbb Z\to(\mathsf{base}=\mathsf{base}),
\qquad
d_0(n)\coloneqq\mathsf{pow}(n).
$$

要用圆的依赖消去原则 L.16 定义
$$
\mathsf{decode}:\prod_{x:\mathbb S^1}D(x),
$$
还需给出路径
$$
\delta:\mathsf{transport}^{D}(\mathsf{loop},d_0)=d_0.
$$
由函数外延性，只需对任意 $n:\mathbb Z$ 构造逐点路径。transport 穿过依赖函数类型的计算给出
$$
\mathsf{transport}^{D}(\mathsf{loop},d_0)(n)
=
\mathsf{transport}^{\lambda x.\mathsf{base}=x}
(\mathsf{loop},
d_0(\mathsf{transport}^{\mathsf{code}}(\mathsf{loop}^{-1},n))).
$$
引理 N.3 把内层 transport 化为 $\mathsf{pred}_{\mathbb Z}(n)$，而路径族 $\lambda x.\mathsf{base}=x$ 的 transport 满足
$$
\mathsf{transport}^{\lambda x.\mathsf{base}=x}(\mathsf{loop},q)
=
q\cdot\mathsf{loop}.
$$
所以逐点目标化为
$$
\mathsf{pow}(\mathsf{pred}_{\mathbb Z}(n))\cdot\mathsf{loop}
=
\mathsf{pow}(n),
$$
这正是引理 N.5。于是得到 $\delta$，并由圆依赖消去定义 $\mathsf{decode}$。

## N.6 两侧逆性

**定理 N.8（decode-after-encode）.** 对任意 $x:\mathbb S^1$ 和 $p:\mathsf{base}=x$，
$$
\mathsf{decode}_x(\mathsf{encode}_x(p))=p.
$$

**证明.** 对路径 $p$ 作路径归纳。归纳后 $x\equiv\mathsf{base}$ 且 $p\equiv\mathsf{refl}_{\mathsf{base}}$。左边化为
$$
\mathsf{decode}_{\mathsf{base}}(0_{\mathbb Z})
=
d_0(0_{\mathbb Z})
=
\mathsf{pow}(\mathsf{pos}(0))
\equiv
\mathsf{refl}_{\mathsf{base}}.
$$
故取反身路径。$\square$

**引理 N.9（code fibers 是集合）.** 对任意 $x:\mathbb S^1$，$\mathsf{code}(x)$ 是集合。

**证明.** 对 $x$ 作圆归纳。基点处由附录 M.5，$\mathbb Z$ 是集合。路径构造子处，需要证明把集合性沿 $\mathsf{loop}$ transport 后仍等于自身；这是附录 O.5，其核心是推论 O.4：$\mathsf{isSet}(A)$ 是命题。$\square$

**定理 N.10（encode-after-decode）.** 对任意 $x:\mathbb S^1$ 和 $c:\mathsf{code}(x)$，
$$
\mathsf{encode}_x(\mathsf{decode}_x(c))=c.
$$

**证明.** 设
$$
P(x)\coloneqq
\prod_{c:\mathsf{code}(x)}
(\mathsf{encode}_x(\mathsf{decode}_x(c))=c).
$$
由引理 N.9，每个 $\mathsf{code}(x)$ 是集合，所以每个等式类型
$$
\mathsf{encode}_x(\mathsf{decode}_x(c))=c
$$
是命题。由函数外延性和命题值依赖函数类型的稳定性，$P(x)$ 是命题。

对 $x$ 作圆归纳来构造 $P(x)$。路径构造子相干由 $P(\mathsf{base})$ 的命题性给出，因此只需基点情形。基点情形要求
$$
\prod_{n:\mathbb Z}
\mathsf{encode}_{\mathsf{base}}(\mathsf{decode}_{\mathsf{base}}(n))=n.
$$
而 $\mathsf{decode}_{\mathsf{base}}(n)=\mathsf{pow}(n)$，所以目标正是引理 N.6。$\square$

## N.7 圆的 loop space

**定理 N.11（圆的 loop space）.** 有等价
$$
(\mathsf{base}=\mathsf{base})\simeq\mathbb Z.
$$

**证明.** 取
$$
\mathsf{encode}_{\mathsf{base}}:
(\mathsf{base}=\mathsf{base})\to\mathbb Z
$$
和
$$
\mathsf{decode}_{\mathsf{base}}:
\mathbb Z\to(\mathsf{base}=\mathsf{base}).
$$
定理 N.8 给出
$$
\mathsf{decode}_{\mathsf{base}}\circ\mathsf{encode}_{\mathsf{base}}
\sim
\mathsf{id},
$$
定理 N.10 给出
$$
\mathsf{encode}_{\mathsf{base}}\circ\mathsf{decode}_{\mathsf{base}}
\sim
\mathsf{id}.
$$
因此 $\mathsf{encode}_{\mathsf{base}}$ 有准逆。由推论 G.7，$\mathsf{encode}_{\mathsf{base}}$ 是等价。$\square$

## N.8 计算口径核对

本证明使用规则 9.4-9.5 与附录 L 固定的圆输入：点构造子上的计算是 judgmental，loop 构造子上的计算是 propositional。因此
$$
\mathsf{code}(\mathsf{base})\equiv\mathbb Z
$$
以及消去截面在 $\mathsf{base}$ 上的值都不需要额外 transport；涉及 $\mathsf{loop}$ 的等式则通过 $\beta_{\mathsf{loop}}$、$\beta^P_{\mathsf{loop}}$ 和命题 N.3 的 transport 计算处理。若改用连点计算也仅为 propositional 的另一套 HIT 语法，必须先给出从该语法到本书规则包的翻译；那不是本定理当前口径中的遗留证明步骤。
