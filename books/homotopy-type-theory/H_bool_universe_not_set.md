# 附录 H：布尔类型与 Universe 非集合性

## 目标

本附录给出一个标准但重要的单值性后果：若 universe 含有布尔类型 $\mathbf 2$，并满足单值性，则该 universe 不是集合。证明只使用布尔类型消去、路径归纳、函数外延性不需要，单值性用于把布尔类型的非平凡自等价变成 universe 中的非平凡 loop。

## H.1 布尔类型的判别

布尔类型 $\mathbf 2$ 有构造子
$$
\mathsf{false}:\mathbf 2,\qquad \mathsf{true}:\mathbf 2.
$$

**定义 H.1（布尔判别族）.** 定义
$$
P:\mathbf 2\to\mathcal U
$$
为
$$
P(\mathsf{false})\coloneqq\mathbf 0,\qquad
P(\mathsf{true})\coloneqq\mathbf 1.
$$

**命题 H.2（布尔两点不同）.** 有
$$
\mathsf{false}\ne\mathsf{true}
$$
和
$$
\mathsf{true}\ne\mathsf{false}.
$$

**证明.** 若 $p:\mathsf{false}=\mathsf{true}$，则 transport
$$
\mathsf{transport}^{P}(p)
$$
把 $P(\mathsf{false})\equiv\mathbf 0$ 映到 $P(\mathsf{true})\equiv\mathbf 1$。这本身不给矛盾，因为没有 $\mathbf 0$ 中的点。改用逆方向：若 $q:\mathsf{true}=\mathsf{false}$，则
$$
\mathsf{transport}^{P}(q,\star):P(\mathsf{false})\equiv\mathbf 0,
$$
矛盾。于是 $\mathsf{true}\ne\mathsf{false}$。

若 $p:\mathsf{false}=\mathsf{true}$，则 $p^{-1}:\mathsf{true}=\mathsf{false}$，与上段矛盾。因此 $\mathsf{false}\ne\mathsf{true}$。$\square$

## H.2 取反是非平凡自等价

**定义 H.3（布尔取反）.** 定义
$$
\mathsf{negBool}:\mathbf 2\to\mathbf 2
$$
为
$$
\mathsf{negBool}(\mathsf{false})\coloneqq\mathsf{true},\qquad
\mathsf{negBool}(\mathsf{true})\coloneqq\mathsf{false}.
$$

**命题 H.4（取反是自逆）.** 对任意 $b:\mathbf 2$，
$$
\mathsf{negBool}(\mathsf{negBool}(b))=b.
$$

**证明.** 对 $b$ 作布尔消去。两个构造子情形都按定义计算为反身路径。$\square$

**命题 H.5（取反是等价）.** $\mathsf{negBool}:\mathbf 2\to\mathbf 2$ 是等价。

**证明.** 它的准逆为自身，左右逆同伦均由命题 H.4 给出。由附录 G.7，准逆推出等价。$\square$

**命题 H.6（取反不等于恒等函数）.** 有
$$
\mathsf{negBool}\ne\mathsf{id}_{\mathbf 2}.
$$

**证明.** 若 $p:\mathsf{negBool}=\mathsf{id}_{\mathbf 2}$，对 $p$ 使用 $\mathsf{happly}$ 并代入 $\mathsf{true}$，得到
$$
\mathsf{negBool}(\mathsf{true})=\mathsf{id}_{\mathbf 2}(\mathsf{true}).
$$
按定义化为
$$
\mathsf{false}=\mathsf{true},
$$
与命题 H.2 矛盾。$\square$

## H.3 Universe 不是集合

**定理 H.7（单值 universe 非集合）.** 设 $\mathbf 2:\mathcal U$，并且 $\mathcal U$ 满足单值性。则 $\mathcal U$ 不是集合：
$$
\neg\mathsf{isSet}(\mathcal U).
$$

**证明.** 设反设
$$
S:\mathsf{isSet}(\mathcal U).
$$
令
$$
e:\mathbf 2\simeq\mathbf 2
$$
为由 $\mathsf{negBool}$ 和命题 H.5 给出的自等价。由单值性得到 universe 中的路径
$$
\mathsf{ua}(e):\mathbf 2=\mathbf 2.
$$
另有反身路径 $\mathsf{refl}_{\mathbf 2}:\mathbf 2=\mathbf 2$。因为 $\mathcal U$ 是集合，路径空间 $\mathbf 2=\mathbf 2$ 是命题，所以
$$
\alpha:\mathsf{ua}(e)=\mathsf{refl}_{\mathbf 2}.
$$
对 $\alpha$ 作用函数 $\mathsf{idtoequiv}$，得到
$$
\mathsf{idtoequiv}(\mathsf{ua}(e))=\mathsf{idtoequiv}(\mathsf{refl}_{\mathbf 2}).
$$
左边由单值性的计算三角同伦等于 $e$，右边按定义等于恒等等价。因此得到
$$
e=\mathsf{idEquiv}_{\mathbf 2}.
$$
对该路径取第一投影，得到底层函数相等
$$
\mathsf{negBool}=\mathsf{id}_{\mathbf 2},
$$
这与命题 H.6 矛盾。故 $\mathcal U$ 不是集合。$\square$

**说明 H.8.** 该证明只表明含有布尔类型并满足单值性的 universe 具有非平凡 1-路径；它不声称 universe 的全部高阶结构已被计算。
