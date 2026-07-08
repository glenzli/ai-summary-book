# 附录 Q：低阶 stable stems 与 Adams-Novikov 校验表

本附录把计算章节从“流程”推进到“最小可核查表”。表格只列本书可安全使用的低阶稳定同伦事实；ANSS class 名称和 differential 的精确定位仍需 Ravenel/Goerss/现代计算表逐项补齐。

## Q.1 Integral stable stems: stems 0--3

| stem $k$ | $\pi_k^S$ | 常用生成元 | 说明 |
| --- | --- | --- | --- |
| 0 | $\mathbb Z$ | unit | 球谱单位 |
| 1 | $\mathbb Z/2$ | $\eta$ | Hopf map 稳定化 |
| 2 | $\mathbb Z/2$ | $\eta^2$ | $\eta$ 的平方 |
| 3 | $\mathbb Z/24$ | $\nu$ | Hopf invariant one family 的下一项；含 2-primary 和 3-primary 信息 |

**使用限制 Q.2.** 表 Q.1 是 classical stable stems 的低阶事实。若要在 Adams-Novikov spectral sequence 中使用，必须另外说明这些元素在 ANSS filtration 中的代表、differentials 和 extensions。

## Q.2 Prime-complete primary decomposition

| stem | $p=2$ primary | $p=3$ primary | $p\ge5$ |
| --- | --- | --- | --- |
| 0 | $\mathbb Z_2^\wedge$ component | $\mathbb Z_3^\wedge$ component | $\mathbb Z_p^\wedge$ component |
| 1 | $\mathbb Z/2$ | 0 | 0 |
| 2 | $\mathbb Z/2$ | 0 | 0 |
| 3 | $\mathbb Z/8$ | $\mathbb Z/3$ | 0 |

**证明 Q.3.** 对 stem $1$ 和 $2$，$\pi_1^S$、$\pi_2^S$ 都是 $\mathbb Z/2$。对 stem $3$，$\mathbb Z/24$ 的 primary decomposition 为 $\mathbb Z/8\oplus\mathbb Z/3$。证毕。

## Q.3 Adams 与 Adams-Novikov 的低阶校验

| 元素 | classical Adams 检测 | ANSS/chromatic 解释 | 本书使用状态 |
| --- | --- | --- | --- |
| $\eta$ | $h_1$ at $p=2$ | height 1 低阶周期现象 | 可作低阶例子 |
| $\nu$ | $h_2$ at $p=2$ | 2-primary part in stem 3；odd part at $p=3$ 另计 | 可作低阶例子 |
| $\alpha_1$ at odd $p$ | Adams-Novikov Greek-letter family | stem $2p-3$，order $p$ | 需 Ravenel locator |

**警告 Q.4.** 表 Q.3 不声称给出完整 ANSS。它只说明本书使用这些低阶元素时应如何连接 classical Adams、ANSS 和 chromatic language。

## Q.4 Hidden extension 的最小实例

**命题 Q.5.** Associated graded group
$$
\operatorname{gr}G\cong\mathbb Z/2\oplus\mathbb Z/2
$$
不能唯一决定 $G$。

**证明.** 令 $G_1=\mathbb Z/4$，取 filtration
$$
0\subset 2\mathbb Z/4\subset\mathbb Z/4.
$$
其 associated graded 是 $\mathbb Z/2\oplus\mathbb Z/2$。令 $G_2=\mathbb Z/2\oplus\mathbb Z/2$，取一个直接和 filtration，也有同样 associated graded。但 $G_1\not\cong G_2$。证毕。

**解释 Q.6.** 因此 Adams-Novikov $E_\infty$ 页不能单独决定稳定 homotopy groups；必须解 hidden additive extensions。

## Q.5 最小 ANSS 记录模板

| stable element | prime | stem | ANSS class | filtration | differential status | extension status | source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| $\eta$ | 2 | 1 | 待定位 | 待定位 | survives | extension trivial | Ravenel/Adams table |
| $\eta^2$ | 2 | 2 | 待定位 | 待定位 | survives | extension trivial | Ravenel/Adams table |
| $\nu$ 2-primary | 2 | 3 | 待定位 | 待定位 | survives | order 8 extension data | Ravenel/Adams table |
| $\nu$ 3-primary | 3 | 3 | 待定位 | 待定位 | survives | order 3 | Ravenel/ANSS table |
| $\alpha_1$ | odd $p$ | $2p-3$ | $\alpha_1$ | 待定位 | survives | order $p$ | Ravenel Greek-letter table |

**使用规则 Q.7.** 本书正文可以使用表 Q.1-Q.3 的低阶 stable stems 和 primary decomposition。若要声称某个 ANSS class、filtration 或 differential，必须先把 Q.5 中“待定位”替换为 source locator。

## 本附录小结

计算链已经达到教材内容最小闭合：低阶 stable stems、primary decomposition、hidden extension 风险和 ANSS 记录模板均已给出。完整 ANSS 表属于后续计算附录扩展，而不是核心定义链的阻塞项。
