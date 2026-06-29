# 附录 AH：Full Subcategory 与本质像证明核

本附录补全附录 R 中 Rezk 完备化构造的两个标准步骤：

1.  单值范畴的命题性 full subcategory 仍单值；
2.  函子的本质像嵌入 essentially surjective，若原函子 fully faithful 则本质像嵌入 fully faithful。

## AH.1 命题性 full subcategory

设 $\mathcal C$ 为预范畴，$P:\mathcal C_0\to\mathcal U$ 为对象谓词，并假设
$$
\prod_{x:\mathcal C_0}\mathsf{isProp}(P(x)).
$$

**定义 AH.1（full subcategory）.** 定义 $\mathcal C_P$：

1.  对象为
    $$
    \sum_{x:\mathcal C_0}P(x);
    $$
2.  Hom 继承自 $\mathcal C$：
    $$
    \mathcal C_P((x,p),(y,q))\coloneqq \mathcal C(x,y);
    $$
3.  恒等、复合和范畴律均来自 $\mathcal C$。

**引理 AH.2（full subcategory 的同构）.** 对对象 $(x,p),(y,q):\mathcal C_P$，有等价
$$
((x,p)\cong_{\mathcal C_P}(y,q))\simeq(x\cong_{\mathcal C}y).
$$

**证明.** 两侧的底层态射、逆态射和逆律完全相同，因为 Hom 定义为继承自 $\mathcal C$。谓词 $P$ 不出现在态射中。$\square$

**定理 AH.3（命题性 full subcategory 保持单值性）.** 若 $\mathcal C$ 是单值范畴，则 $\mathcal C_P$ 是单值范畴。

**证明.** 固定 $(x,p),(y,q):\mathcal C_P$。由 $\Sigma$ 路径刻画，
$$
((x,p)=(y,q))
\simeq
\sum_{r:x=y}\mathsf{transport}^{P}(r,p)=q.
$$
第二分量位于命题 $P(y)$ 中，因此自动唯一。故
$$
((x,p)=(y,q))\simeq(x=y).
$$
由 $\mathcal C$ 单值，
$$
(x=y)\simeq(x\cong_{\mathcal C}y).
$$
由 AH.2，
$$
(x\cong_{\mathcal C}y)\simeq((x,p)\cong_{\mathcal C_P}(y,q)).
$$
复合得到对象路径与同构的等价。反身路径对应恒等同构由路径归纳验证，因此该等价就是 $\mathsf{idtoiso}$，$\mathcal C_P$ 单值。$\square$

## AH.2 本质像

设
$$
F:\mathcal C\to\mathcal D
$$
为函子，且 $\mathcal D$ 为单值范畴。

**定义 AH.4（本质像）.** 定义 $F$ 的本质像 $\mathsf{Im}(F)$ 为 $\mathcal D$ 的 full subcategory，其对象谓词为
$$
P_F(d)\coloneqq
\left\|
\sum_{c:\mathcal C}F(c)\cong d
\right\|.
$$
由于命题截断是命题，$P_F$ 是命题性谓词。

**定理 AH.5（本质像单值）.** $\mathsf{Im}(F)$ 是单值范畴。

**证明.** 由 AH.3 应用于单值范畴 $\mathcal D$ 和命题性谓词 $P_F$。$\square$

**定义 AH.6（到本质像的核心限制）.** 定义函子
$$
\bar F:\mathcal C\to\mathsf{Im}(F)
$$
在对象上为
$$
c\mapsto(Fc,\ |(c,\mathsf{id}_{Fc})|),
$$
在态射上与 $F$ 相同。

**定理 AH.7（核心限制 essentially surjective）.** 函子 $\bar F$ essentially surjective。

**证明.** 取对象 $(d,r):\mathsf{Im}(F)$。需证明
$$
\left\|\sum_{c:\mathcal C}\bar F(c)\cong(d,r)\right\|.
$$
目标是命题截断，因此可对 $r$ 消去。取代表
$$
(c,i):\sum_{c:\mathcal C}F(c)\cong d.
$$
由 AH.2，$\mathcal D$ 中的同构 $i:Fc\cong d$ 给出 full subcategory 中的同构
$$
\bar F(c)\cong(d,r).
$$
送入命题截断即可。$\square$

**定理 AH.8（核心限制 fully faithful）.** 若 $F$ fully faithful，则 $\bar F$ fully faithful。

**证明.** 对 $c,c':\mathcal C$，$\mathsf{Im}(F)$ 中
$$
\mathsf{Im}(F)(\bar F(c),\bar F(c'))
$$
按 full subcategory 定义就是
$$
\mathcal D(Fc,Fc').
$$
而 $\bar F$ 在 Hom 上的函数就是 $F$ 在 Hom 上的函数。由 $F$ fully faithful，该函数是等价。$\square$

## AH.3 Yoneda 本质像

**推论 AH.9（Yoneda 本质像 Rezk 嵌入）.** 对 Yoneda 嵌入
$$
y:\mathcal C\to\mathsf{PSh}(\mathcal C),
$$
其本质像 $\mathsf{Im}(y)$ 是单值范畴；核心限制
$$
\eta:\mathcal C\to\mathsf{Im}(y)
$$
essentially surjective；若 Yoneda fully faithful，则 $\eta$ fully faithful。

**证明.** 预层范畴单值性由 R.2 / X.10-X.11 给出。应用 AH.5-AH.8。Yoneda fully faithful 由 U.11-U.12。$\square$

**说明 AH.10.** 附录 R 采用路径形式
$$
\left\|\sum_{c:\mathcal C}(y(c)=P)\right\|
$$
描述本质像，而 AH.4 采用同构形式
$$
\left\|\sum_{c:\mathcal C}y(c)\cong P\right\|.
$$
在单值预层范畴中二者等价，因此两种定义给出等价的 full subcategory。
