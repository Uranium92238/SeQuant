using TensorKit
include("ccd-helper.jl")
nv = deserialize("nv.jlbin")
nocc = deserialize("nocc.jlbin")
cod = ℝ^nv ⊗ ℝ^nv 
dom = ℝ^nocc ⊗ ℝ^nocc
g_vvoo = TensorMap(deserialize("g_vvoo.jlbin"),cod,dom)
T2 = TensorMap(initialize_t2_only(),cod,dom)
T2 = initialize_t2_only()
@tensor R2[i,j,u,v] := T2[a,b,i,j] * g_vvoo[a,b,u,v]    
R2 = TensorMap(ones(Float64,nv,nv,nocc,nocc),ℝ^nv ⊗ ℝ^nv , ℝ^nocc ⊗ ℝ^nocc)
Scaled_R2 = copy(R2)
R2_storage = Array{typeof(R2)}(undef, 0)
push!(R2_storage, copy(R2))
push!(R2_storage, copy(T2))
g_vvvv = TensorMap(deserialize("g_vvvv.jlbin"),ℝ^nv⊗ℝ^nv,ℝ^nv⊗ℝ^nv)
push!(R2_storage, copy(g_vvvv))
length(R2_storage)
R_iter = copy(g_vvvv)
calculate_ECCD(T2)
convert(Array,T2)
R2u = copy(R2)
R2 = calculate_residual(T2,R2u,R2)
T2 = update_T2(T2,R2,Scaled_R2)
SquareDot(T2,T2)
T2
C = [1 2 3]
(T2+R2)*C[3]
push!(R2_storage, copy(R2))
C[1]*(R2_storage[1]+T2)
R2 = C[3]*(R2 + T2)


@tensor Trm1[a_1,a_2,i_1,i_2] := 0.5* g_vvoo[a_1,a_2,i_1,i_2]
Trm1 = nothing
GC.gc()
Trm1

@code_warntype initialize_cc_variables()
@code_warntype main("water_dump.fcidump")
@code_warntype calculate_residual_memeff(T2)