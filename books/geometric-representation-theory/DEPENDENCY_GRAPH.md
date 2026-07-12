# 依赖图

本文件记录本书定义和证明依赖，防止倒用高级定理。

## 0. 基础层

```
universes, base field, coefficient field
  -> algebraic varieties and algebraic groups
  -> quotient stacks and equivariant categories
  -> derived categories and six functors
```

附录 A 必须先于第三章、第四章及所有后续 sheaf-theoretic 章节。

## 1. Flag 几何到 Hecke category

```
root datum and reductive group
  -> Borel subgroup and flag variety G/B
  -> Bruhat decomposition and Schubert stratification
  -> B-equivariant sheaves on G/B
  -> convolution correspondence
  -> Hecke category
  -> Kazhdan-Lusztig basis via IC sheaves
```

其中 Bruhat decomposition、Schubert closure normality/rational smoothness 等深层性质属于外部输入。第四章只在类型层面构造 convolution，KL 正性和 character formula 不作为内部证明。

## 2. Category O 到 localization

```
triangular decomposition of g
  -> Verma modules
  -> BGG category O
  -> central characters and dot action
  -> D_lambda-modules on G/B
  -> Beilinson-Bernstein localization
  -> KL character formula
```

第二章只构造代数侧。第七至第九章才能调用 D-module 和 localization。不得在第二章证明中预先使用 Beilinson-Bernstein。

## 3. Springer 和 Steinberg

```
nilpotent cone
  -> Springer resolution
  -> Steinberg variety
  -> convolution algebra in Borel-Moore homology
  -> Weyl group action
  -> Springer correspondence
```

Springer correspondence 是外部输入，后续可在第五章和第六章中作为核心 theorem locator 处理。

## 4. Affine Grassmannian 和 Satake

```
fpqc loop quotient LG/L^+G
  -> representability and reduced Betti Schubert exhaustion
  -> finite-support equivariant perverse category
  -> torsor descent of twisted external products
  -> properness on finite convolution supports
  -> stratified-semismall convolution t-exactness
  -> fusion and parity-corrected symmetric structure
  -> exact faithful global-cohomology fiber functor
  -> neutral Tannakian group H
  -> weight-functor torus morphism T^vee -> H
  -> MV-cycle/rank-one identification H = G^vee
```

第十二章内部证明 descent associativity、unit 和 $GL_2$ semismall dimension check；finite-support properness 与 semismall decomposition 是外部输入。第十三章内部只从 weight grading 构造 $T^\vee\to H$；fusion、fiber functor、neutral Tannaka 和 $H\simeq G^\vee$ 分别使用 `GSAT-CONV-1`、`GSAT-FIBER-1`、`TANNAKA-1`、`GSAT-1`，不得互相替代。

## 5. 前沿边界

```
Satake, Kac-Moody localization, factorization categories
  -> local geometric Langlands inputs
  -> global geometric Langlands categories
```

```
affine Grassmannian convolution and Borel-Moore homology
  -> BFN Coulomb branches
  -> quantized Coulomb branches
  -> symplectic duality interfaces
```

这些依赖当前只作为研究边界，不能用于证明基础章节中的核心定理。

## 6. 范畴化、辛几何和 Hall 方向

```
quiver representations
  -> Nakajima quiver varieties
  -> Hecke correspondences
  -> Kac-Moody and quantum group representations
```

```
Cartan datum
  -> KLR algebras
  -> induction and restriction functors
  -> Grothendieck groups
  -> quantum group integral forms and canonical bases
```

```
conical symplectic resolutions
  -> quantizations
  -> category O
  -> twisting and shuffling functors
  -> symplectic duality
```

```
affine Grassmannian convolution
  -> BFN space R
  -> equivariant Borel-Moore homology algebra
  -> Coulomb branch and quantization
```

```
quiver representation stacks
  -> short exact sequence correspondences
  -> Hall algebras and CoHA
  -> DT and wall-crossing interfaces
```

Nakajima、KLR/Rouquier、BLPW、BFN、CoHA 和 canonical basis 的核心定理均为外部输入。当前主体章节只建立定义、类型检查和接口，不把这些方向的深层定理倒用于基础章节。
