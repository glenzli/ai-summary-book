# 附录 AZ：谱、稳定范畴与收敛证明接口

附录 AM 定义预谱和 Omega 谱，附录 AQ-AV 处理 exact couple 与经典谱序列模板。本附录补上稳定范畴和收敛定理所需的抽象结构。

## AZ.1 Spectrum category

**定义 AZ.1（spectrum）.** 谱 $E$ 由 pointed 类型族 $E_n$ 和结构映射
$$
\sigma_n:\Sigma E_n\to_\ast E_{n+1}
$$
组成。若伴随映射
$$
E_n\to_\ast\Omega E_{n+1}
$$
均为等价，则称为 Omega 谱。

**定义 AZ.2（谱映射）.** 谱映射 $f:E\to F$ 由 pointed maps
$$
f_n:E_n\to_\ast F_n
$$
组成，并与结构映射相容：
$$
f_{n+1}\circ\sigma^E_n\simeq\sigma^F_n\circ\Sigma f_n.
$$

**定义 AZ.3（稳定等价）.** 谱映射 $f:E\to F$ 是稳定等价，若对所有 $k$，诱导的稳定同伦群
$$
\pi_k^s(E)\to\pi_k^s(F)
$$
为同构。

**输入 AZ.4（stable homotopy category）.** 稳定同伦范畴 $\mathsf{Sp}$ 是谱关于稳定等价的同伦范畴或相应 $\infty$-范畴。HoTT 内部构造它需要 Rezk completion、localization 或 higher category machinery。

## AZ.2 Exact triangles

**定义 AZ.5（cofiber sequence of spectra）.** 谱映射
$$
E\to F\to G
$$
是 cofiber sequence，若逐层或稳定意义下 $G$ 是 $E\to F$ 的 cofiber，并与结构映射相容。

**输入 AZ.6（long exact sequence of generalized cohomology）.** cofiber sequence 诱导长正合列
$$
\cdots\to E^n(G)\to E^n(F)\to E^n(E)\to E^{n+1}(G)\to\cdots.
$$

**证明边界.** 该定理依赖 spectrum mapping space、fiber/cofiber 稳定等价和 loop-suspension 相容。可由稳定 $\infty$-范畴中的 fiber/cofiber triangle 推出。

## AZ.3 Filtered spectra

**定义 AZ.7（filtered spectrum）.** filtered spectrum 为序列
$$
\cdots\to X_{p-1}\to X_p\to X_{p+1}\to\cdots
$$
及其 colimit $X$，并给出 cofiber 层
$$
\mathsf{gr}_pX\coloneqq X_p/X_{p-1}.
$$

**命题 AZ.8（filtered spectrum exact couple）.** 对任意 spectrum-valued cohomology theory $E^\ast$，filtered spectrum 给出 exact couple
$$
D_1^{p,q}=E^{p+q}(X_p),
\qquad
E_1^{p,q}=E^{p+q}(\mathsf{gr}_pX).
$$

**证明.** 由每个 cofiber sequence
$$
X_{p-1}\to X_p\to\mathsf{gr}_pX
$$
的长正合列 AZ.6 组成 exact couple。$\square$

## AZ.4 强收敛

**定义 AZ.9（exhaustive and complete filtration）.** 过滤 $F^pG$ exhaustive 若 $\bigcup_pF^pG=G$，separated 若 $\bigcap_pF^pG=0$，complete 若自然映射
$$
G\to\lim_p G/F^pG
$$
为同构。

**定理 AZ.10（强收敛判据，外部输入 / 证明核）.** 若 filtered spectrum $X_\bullet$ 有界下、每一总次数只有有限多个非零 $E_r^{p,q}$，并且目标过滤 exhaustive、separated 且 complete，则由 exact couple 产生的谱序列强收敛：
$$
E_\infty^{p,q}\cong \mathsf{gr}^pE^{p+q}(X).
$$

**证明核.** 有界性保证对固定总次数，cycle-boundary filtration 在有限页后稳定；因此 $E_\infty$ 可定义为稳定值。exhaustive/separated/complete 条件把 associated graded 重建为目标群的过滤。扩张问题不由谱序列自动解决，需额外处理。$\square$

## AZ.5 本附录关闭的缺口

谱序列收敛不再只是“将来工作”：它被分解为 filtered spectrum、cofiber 长正合列、exact couple、有限性条件和过滤完备性。剩余工作是为具体谱序列逐项证明这些输入。
