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
loop group G((z)) and arc group G[[z]]
  -> affine Grassmannian Gr_G
  -> G[[z]]-orbits indexed by dominant coweights
  -> perverse sheaves and convolution
  -> rigid symmetric tensor category
  -> Tannakian reconstruction
  -> Rep(G^vee)
```

Mirkovic-Vilonen theorem 是外部输入。Tannakian reconstruction 的一般定理也必须定位。

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
