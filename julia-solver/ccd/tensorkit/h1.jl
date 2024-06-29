function addres_gvvoo(R2u,nv,nocc)
    g_vvoo = TensorMap(deserialize("g_vvoo.jlbin"),ℝ^nv ⊗ ℝ^nv,ℝ^nocc ⊗ ℝ^nocc)
    @tensor Trm1[a_1,a_2,i_1,i_2] := 0.5* g_vvoo[a_1,a_2,i_1,i_2]
    @tensor R2u[a_1,a_2,i_1,i_2] += Trm1[a_1,a_2,i_1,i_2]
    return R2u
end

function addres_gvoov(R2u,nv,nocc)
    g_voov = TensorMap(deserialize("g_voov.jlbin"),ℝ^nv ⊗ ℝ^nocc,ℝ^nocc ⊗ ℝ^nv)
    @tensor Trm2[a_1,a_2,i_1,i_2] := - g_voov[a_1,i_3,i_1,a_3] * T2[a_2,a_3,i_3,i_2]
    @tensor R2u[a_1,a_2,i_1,i_2] += Trm2[a_1,a_2,i_1,i_2]
    @tensor Trm5[a_1,a_2,i_1,i_2] := 2*g_voov[a_1,i_3,i_1,a_3] * T2[a_2,a_3,i_2,i_3]
    @tensor R2u[a_1,a_2,i_1,i_2] += Trm5[a_1,a_2,i_1,i_2]
    return R2u
end