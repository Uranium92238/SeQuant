include("ccd-helper.jl")
initialize_cc_variables()
T2 = initialize_t2_only()
g_oovv = deserialize("g_oovv.jlbin")
g_voov = deserialize("g_voov.jlbin")
g_vovo = deserialize("g_vovo.jlbin")
g_oooo = deserialize("g_oooo.jlbin")
g_vvvv = deserialize("g_vvvv.jlbin")
g_vvoo = deserialize("g_vvoo.jlbin")
fvv = deserialize("fvv.jlbin")
foo = deserialize("foo.jlbin")
@tensor begin
    Trm1[a_1,a_2,i_1,i_2] := 0.5* g_vvoo[a_1,a_2,i_1,i_2]
    Trm2[a_1,a_2,i_1,i_2] := - g_voov[a_1,i_3,i_1,a_3] * T2[a_2,a_3,i_3,i_2] 
    Trm3[a_1,a_2,i_1,i_2] := - g_vovo[a_2,i_3,a_3,i_2] * T2[a_1,a_3,i_1,i_3]
    Trm4[a_1,a_2,i_1,i_2] := - g_vovo[a_2,i_3,a_3,i_1] * T2[a_1,a_3,i_3,i_2]
    Trm5[a_1,a_2,i_1,i_2] := 2*g_voov[a_1,i_3,i_1,a_3] * T2[a_2,a_3,i_2,i_3]
    Trm6[a_1,a_2,i_1,i_2] := + 0.5*g_oooo[i_3,i_4,i_1,i_2] * T2[a_1,a_2,i_3,i_4]
    Trm7[a_1,a_2,i_1,i_2] := + fvv[a_2,a_3] * T2[a_1,a_3,i_1,i_2]
    Trm8[a_1,a_2,i_1,i_2] := + 0.5*g_vvvv[a_1,a_2,a_3,a_4] * T2[a_3,a_4,i_1,i_2]
    Trm9[a_1,a_2,i_1,i_2] := - foo[i_3,i_2] * T2[a_1,a_2,i_1,i_3]
end
@tensor B1[i_4,a_4,a_1,i_1] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_3,i_1,i_3])
@tensor B2[i_4,a_4,a_1,i_1] := 0.5*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_3,i_3,i_1])
@tensor B3[i_4,i_2] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_3,i_2])
@tensor B4[i_4,a_3,a_1,i_1] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_4,i_1,i_3])
@tensor B5[a_3,a_2] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_2,a_4,i_4,i_3])
@tensor B6[i_4,i_2] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_2,i_3])
@tensor B7[a_4,a_2] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_2,a_3,i_4,i_3])
@tensor B8[i_4,a_3,a_1,i_2] := 0.5*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_4,i_3,i_2])
@tensor B9[i_3,i_4,i_1,i_2] := 0.5* (g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_1,i_2])

@tensor R2u[a_1,a_2,i_1,i_2]:=  Trm1[a_1,a_2,i_1,i_2]+ Trm2[a_1,a_2,i_1,i_2]+ Trm3[a_1,a_2,i_1,i_2]+ Trm4[a_1,a_2,i_1,i_2] + Trm5[a_1,a_2,i_1,i_2]+ Trm6[a_1,a_2,i_1,i_2]+ Trm7[a_1,a_2,i_1,i_2]+ Trm8[a_1,a_2,i_1,i_2]+ Trm9[a_1,a_2,i_1,i_2]+ B1[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_2,i_4]- B1[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_4,i_2]+ B2[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_4,i_2]- B3[i_4,i_2] * T2[a_1,a_2,i_1,i_4]- B4[i_4,a_3,a_1,i_1] * T2[a_2,a_3,i_2,i_4]+ B4[i_4,a_3,a_1,i_1] * T2[a_2,a_3,i_4,i_2]+ B5[a_3,a_2] * T2[a_1,a_3,i_1,i_2]+ B6[i_4,i_2] * T2[a_1,a_2,i_1,i_4]- B7[a_4,a_2] * T2[a_1,a_4,i_1,i_2]+ B8[i_4,a_3,a_1,i_2] * T2[a_2,a_3,i_4,i_1]+ B9[i_3,i_4,i_1,i_2] * T2[a_1,a_2,i_3,i_4]
@tensor R2[a,b,i,j] := R2u[a,b,i,j] + R2u[b,a,j,i]

nv = deserialize("nv.jlbin")
nocc = deserialize("nocc.jlbin")
R2u = zeros(Float64,nv,nv,nocc,nocc)
R2 = zeros(Float64,nv,nv,nocc,nocc)
R2 = calculate_residual_new(T2,R2u,R2)
