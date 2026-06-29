# 附录 AA：Rezk 完备化泛性质证明架构

本附录把 R.11 的外部输入进一步拆成标准定理架构。完整逐行形式化仍很长，但本附录给出足够精确的数学骨架：weak equivalence 的限制函子、自然变换的唯一下降、对象扩张的 contractible choice，以及 Rezk 完备化泛性质的归约。

设 $\mathcal C,\mathcal D,\mathcal E$ 为预范畴，其中 $\mathcal D,\mathcal E$ 为单值范畴。

## AA.1 Weak equivalence

**定义 AA.1（fully faithful）.** 函子
$$
F:\mathcal C\to\mathcal D
$$
是 fully faithful，若对任意 $x,y:\mathcal C$，函数
$$
F_{x,y}:\mathcal C(x,y)\to\mathcal D(Fx,Fy)
$$
是等价。

**定义 AA.2（essentially surjective）.** 函子 $F$ 是 essentially surjective，若
$$
\prod_{d:\mathcal D}\left\|
\sum_{c:\mathcal C}Fc\cong d
\right\|.
$$

**定义 AA.3（weak equivalence）.** $F$ 是 weak equivalence，若它 fully faithful 且 essentially surjective。

**注 AA.4。** 这里的 essentially surjective 是命题截断命题。不能从中全局选择每个 $d$ 的代表 $c$；这正是 Rezk 泛性质证明中需要单值目标和 contractible choice 的原因。

## AA.2 限制函子

**定义 AA.5（限制函子）.** 给定
$$
F:\mathcal C\to\mathcal D
$$
和目标范畴 $\mathcal E$，定义
$$
F^\ast:[\mathcal D,\mathcal E]\to[\mathcal C,\mathcal E]
$$
为预合成：
$$
F^\ast(H)\coloneqq H\circ F.
$$
在自然变换上，
$$
F^\ast(\alpha)_c\coloneqq \alpha_{Fc}.
$$

## AA.3 限制函子的 fully faithful 性

**命题 AA.6（自然变换由本质满像决定）.** 设 $F$ essentially surjective。若
$$
H,K:\mathcal D\to\mathcal E
$$
且 $\alpha,\beta:H\Rightarrow K$ 满足
$$
\prod_{c:\mathcal C}\alpha_{Fc}=\beta_{Fc},
$$
则
$$
\alpha=\beta.
$$

**证明（证明核）。** 由附录 X.4，自然变换相等只需逐对象证明
$$
\alpha_d=\beta_d.
$$
固定 $d:\mathcal D$。目标位于 Hom 集合
$$
\mathcal E(Hd,Kd)
$$
的路径空间中，因此是命题。故可对 essentially surjective 给出的命题截断消去，取代表
$$
(c,i):\sum_{c:\mathcal C}Fc\cong d.
$$
自然性沿同构 $i:Fc\cong d$ 给出交换方块：
$$
K(i)\circ\alpha_{Fc}=\alpha_d\circ H(i),
$$
$$
K(i)\circ\beta_{Fc}=\beta_d\circ H(i).
$$
由假设 $\alpha_{Fc}=\beta_{Fc}$，两式右侧相等。再用 $H(i)$ 是同构，右消去得到
$$
\alpha_d=\beta_d.
$$
由函数外延性和 X.4 得到 $\alpha=\beta$。$\square$

**命题 AA.7（限制函子在 Hom 上是嵌入）.** 若 $F$ essentially surjective，则对任意
$H,K:\mathcal D\to\mathcal E$，
$$
F^\ast:(H\Rightarrow K)\to(HF\Rightarrow KF)
$$
是 embedding。

**证明.** 若两个自然变换限制后相等，则由 AA.6 它们相等。由于自然变换类型是集合（X.5），路径纤维为命题，故是 embedding。$\square$

**命题 AA.8（限制函子的 Hom-surjectivity，证明架构）.** 若 $F$ fully faithful 且 essentially surjective，且 $\mathcal E$ 单值，则对任意
$H,K:\mathcal D\to\mathcal E$，任意自然变换
$$
\gamma:HF\Rightarrow KF
$$
可唯一扩张为自然变换
$$
\bar\gamma:H\Rightarrow K
$$
满足 $F^\ast(\bar\gamma)=\gamma$。

**证明架构。** 对 $d:\mathcal D$，考虑分量候选类型
$$
\mathsf{Comp}(d)\coloneqq
\sum_{u:\mathcal E(Hd,Kd)}
\prod_{(c,i:Fc\cong d)}
  \bigl(K(i)\circ \gamma_c=u\circ H(i)\bigr).
$$
该类型是 contractible：

1.  存在性可在命题截断下证明，因为“$\mathsf{Comp}(d)$ 可收缩”是命题；取代表 $(c,i)$ 后，定义
    $$
    u\coloneqq K(i)\circ\gamma_c\circ H(i)^{-1}.
    $$
2.  若取另一代表 $(c',i')$，则 $i'^{-1}\circ i:Fc\cong Fc'$。由 $F$ fully faithful 得到唯一同构 $h:c\cong c'$ 映到它。$\gamma$ 的自然性沿 $h$ 证明两种 $u$ 相等。
3.  任意两个候选 $u,u'$ 都由其对某个代表 $(c,i)$ 的兼容方程和同构消去相等；Hom 集合性消去所有高阶证明。

取 $\mathsf{Comp}(d)$ 的中心作为 $\bar\gamma_d$。自然性对 $f:d\to d'$ 的证明同样是 Hom 集合中的路径命题，可用 essentially surjective 截断消去，把 $d,d'$ 化到 $Fc,Fc'$，再由 $\gamma$ 的自然性和 $F$ 的 fully faithful 性验证。唯一性由 AA.6。$\square$

**定理 AA.9（限制函子 fully faithful）.** 若 $F$ 是 weak equivalence 且 $\mathcal E$ 单值，则
$$
F^\ast:[\mathcal D,\mathcal E]\to[\mathcal C,\mathcal E]
$$
是 fully faithful。

**证明.** Hom 上的 injectivity 由 AA.7；surjectivity 由 AA.8；自然变换类型是集合，故二者合成给出 Hom 等价。$\square$

## AA.4 限制函子的 essential surjectivity

**命题 AA.10（函子沿 weak equivalence 下降，证明架构）.** 若 $F:\mathcal C\to\mathcal D$ 是 weak equivalence，$\mathcal D$ 与 $\mathcal E$ 单值，则任意
$$
G:\mathcal C\to\mathcal E
$$
都存在命题截断意义下的
$$
\sum_{H:\mathcal D\to\mathcal E}HF\cong_{\mathsf{nat}}G.
$$

**证明架构。** 对每个 $d:\mathcal D$，定义扩张对象候选类型
$$
\mathsf{Obj}(d)\coloneqq
\sum_{e:\mathcal E}
\prod_{(c,i:Fc\cong d)}G(c)\cong e
$$
并附加代表相容性。与 AA.8 相同，该类型是 contractible：取代表 $(c,i)$ 时令 $e\coloneqq G(c)$；代表变换由 fully faithful 把 $\mathcal D$ 中的同构拉回 $\mathcal C$，再由 $G$ 送到 $\mathcal E$。因为 $\mathcal E$ 单值，同构给出对象路径，所以 contractible choice 可用于定义 $H(d)$，且不需要从 essentially surjective 截断中作全局选择。

对态射 $f:d\to d'$，用代表 $(c,i)$、$(c',i')$ 把 $f$ 转成
$$
Fc\to Fc'
$$
的态射，再由 fully faithful 唯一提升为
$$
c\to c',
$$
定义 $H(f)$ 为其 $G$-像并沿对象路径重写。函子律目标在 Hom 集合路径中，故可用截断消去和 Hom 集合性验证。

最后，$HF\cong_{\mathsf{nat}}G$ 在对象 $c$ 处由 $Fc$ 的代表 $(c,\mathsf{id})$ 给出；自然性由构造的态射部分计算得到。$\square$

**定理 AA.11（限制函子是等价）.** 若 $F:\mathcal C\to\mathcal D$ 是 weak equivalence，$\mathcal D,\mathcal E$ 单值，则
$$
F^\ast:[\mathcal D,\mathcal E]\to[\mathcal C,\mathcal E]
$$
是范畴论意义下的 weak equivalence；由于函子范畴单值（X.9），它给出对象类型层面的等价。

**证明.** Fully faithful 由 AA.9；essentially surjective 由 AA.10。函子范畴 $[\mathcal D,\mathcal E]$ 和 $[\mathcal C,\mathcal E]$ 的单值性由 X.9，故 weak equivalence 可视为本书所需的“合适意义下等价”。$\square$

## AA.5 Rezk 完备化泛性质

**定理 AA.12（Rezk 完备化泛性质，证明架构）.** 设
$$
\eta:\mathcal C\to\widehat{\mathcal C}
$$
为附录 R 的 Yoneda 本质像 Rezk 嵌入。对任意单值范畴 $\mathcal E$，预合成
$$
\eta^\ast:
[\widehat{\mathcal C},\mathcal E]\to[\mathcal C,\mathcal E]
$$
是等价。

**证明架构。** 附录 R.8 证明 $\eta$ fully faithful；R.10 证明 $\eta$ essentially surjective；R.7 证明 $\widehat{\mathcal C}$ 单值。由 AA.11 应用于
$$
F\coloneqq\eta,\qquad
\mathcal D\coloneqq\widehat{\mathcal C},
$$
得到 $\eta^\ast$ 是等价。$\square$

## AA.6 当前验证边界

本附录把 R.11 从单句外部输入降为可审查的证明架构。仍未完全逐行书内化的是 AA.8 和 AA.10 中 contractible choice 的全部 transport 计算，尤其是：

1.  不同代表 $(c,i)$ 与 $(c',i')$ 的相容路径；
2.  fully faithful 提升同构与函子 $G$ 的相容；
3.  单值目标中对象路径 transport 对 Hom 的具体作用；
4.  函子律和自然性证明分量的 Hom 集合性消去。

这些正是 UniMath/Cubical Agda 中 Rezk completion 形式化需要大量脚本处理的部分。本书后续若要求机器验证，应优先把 AA.8-AA.10 翻译到选定库，而不是重新发明 Rezk 完备化。
