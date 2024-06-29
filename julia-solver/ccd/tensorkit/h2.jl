using TensorKit,TensorOperations,Serialization
Z::Float64 = 0.0
nocc = deserialize("nocc.jlbin")
nv = deserialize("nv.jlbin")
g_oovv = TensorMap(deserialize("g_oovv.jlbin"),ℝ^nocc ⊗ ℝ^nocc,ℝ^nv ⊗ ℝ^nv)
T2_vvoo = TensorMap(deserialize("T2_vvoo.jlbin"),ℝ^nv ⊗ ℝ^nv,ℝ^nocc ⊗ ℝ^nocc)
@tensor Z += 2*g_oovv[i_1, i_2, a_1, a_2]*T2_vvoo[a_1, a_2, i_1, i_2]
#Unload T2_vvoo[a_1, a_2, i_1, i_2]
#Unload g_oovv[i_1, i_2, a_1, a_2]
g_oovv = TensorMap(deserialize("g_oovv.jlbin"),ℝ^nocc ⊗ ℝ^nocc,ℝ^nv ⊗ ℝ^nv)
T2_vvoo = TensorMap(deserialize("T2_vvoo.jlbin"),ℝ^nv ⊗ ℝ^nv,ℝ^nocc ⊗ ℝ^nocc)
@tensor Z += -1*g_oovv[i_1, i_2, a_1, a_2]*T2_vvoo[a_1, a_2, i_2, i_1]
#Unload T2_vvoo[a_1, a_2, i_2, i_1]
#Unload g_oovv[i_1, i_2, a_1, a_2]
return Z