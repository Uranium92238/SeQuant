using TensorOperations, Serialization
begin



    Z = 0.0
    nv = deserialize("nv.jlbin")
    no = deserialize("no.jlbin")
    g_oovv = deserialize("g_oovv.jlbin")
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor Z += 2 * g_oovv[i_1, i_2, a_1, a_2] * T2_vvoo[a_1, a_2, i_1, i_2]
    T2_vvoo = nothing
    g_oovv = nothing
    g_oovv = deserialize("g_oovv.jlbin")
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor Z += -1 * g_oovv[i_1, i_2, a_1, a_2] * T2_vvoo[a_1, a_2, i_2, i_1]
    T2_vvoo = nothing
    g_oovv = nothing
    f_ov = deserialize("f_ov.jlbin")
    T1_vo = deserialize("T1_vo.jlbin")
    @tensor Z += 2 * f_ov[i_1, a_1] * T1_vo[a_1, i_1]
    T1_vo = nothing
    f_ov = nothing
    I_ov = zeros(Float64, no, nv)
    g_oovv = deserialize("g_oovv.jlbin")
    T1_vo = deserialize("T1_vo.jlbin")
    @tensor I_ov[i_1, a_2] += g_oovv[i_1, i_2, a_1, a_2] * T1_vo[a_1, i_2]
    T1_vo = nothing
    g_oovv = nothing
    T1_vo = deserialize("T1_vo.jlbin")
    @tensor Z += -1 * I_ov[i_1, a_2] * T1_vo[a_2, i_1]
    T1_vo = nothing
    I_ov = nothing
    I_ov = zeros(Float64, no, nv)
    g_oovv = deserialize("g_oovv.jlbin")
    T1_vo = deserialize("T1_vo.jlbin")
    @tensor I_ov[i_2, a_2] += g_oovv[i_1, i_2, a_1, a_2] * T1_vo[a_1, i_1]
    T1_vo = nothing
    g_oovv = nothing
    T1_vo = deserialize("T1_vo.jlbin")
    @tensor Z += 2 * I_ov[i_2, a_2] * T1_vo[a_2, i_2]
    T1_vo = nothing
    I_ov = nothing
    return Z

end