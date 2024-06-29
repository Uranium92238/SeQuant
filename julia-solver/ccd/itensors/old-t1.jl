using ITensors,TensorOperations
using ITensors: permute!
include("ccd-helper.jl")
nv,nocc = 2,4;
function addres(R2u::Array{Float64,4},nv,nocc,T2::Array{Float64,4})
    println("TensorOperations is being used")
    g_voov = deserialize("g_voov.jlbin")
    @tensor Trm2[a_1,a_2,i_1,i_2] := - g_voov[a_1,i_3,i_1,a_3] * T2[a_2,a_3,i_3,i_2]
    @tensor R2u[a_1,a_2,i_1,i_2] += Trm2[a_1,a_2,i_1,i_2]
    # display(R2u)
    @tensor Trm5[a_1,a_2,i_1,i_2] := 2*g_voov[a_1,i_3,i_1,a_3] * T2[a_2,a_3,i_2,i_3]
    display(Trm5)
    @tensor R2u[a_1,a_2,i_1,i_2] += Trm5[a_1,a_2,i_1,i_2]
    return R2u
end

function addres(R2u::ITensor,nv,nocc,T2::ITensor)
    println("ITensor is being used")
    i_3 = Index(nocc,"i_3")
    a_3 = Index(nv,"a_3")
    g_voov = ITensor(deserialize("g_voov.jlbin"),a_1,i_3,i_1,a_3)
    T2 = replaceinds(T2, inds(T2) => (a_2, a_3, i_3, i_2))
    R2u = replaceinds(R2u, inds(R2u) => (a_1, a_2, i_1, i_2))
    Trm2 = ITensor(a_1,a_2,i_1,i_2)
    Trm5 = ITensor(a_1,a_2,i_1,i_2)
    Trm2 = g_voov * T2
    Trm2 = permute(Trm2,a_1,a_2,i_1,i_2,allow_alias=false)
    R2u += Trm2
    # @show R2u
    T2 = replaceinds(T2, inds(T2) => (a_2, a_3, i_2,i_3))
    Trm5 = 2*g_voov * T2
    Trm5 = permute(Trm5,a_1,a_2,i_1,i_2,allow_alias=false)
    @show Trm5
    R2u += Trm5
    return R2u
    println("All Good")
end

function accessindex(T2::ITensor)
    println(a_1)
end

function addres_gvvoo(R2u::Array{Float64,4},nv,nocc)
    g_vvoo = deserialize("g_vvoo.jlbin")
    @tensor Trm1[a_1,a_2,i_1,i_2] := 0.5* g_vvoo[a_1,a_2,i_1,i_2]
    @tensor R2u[a_1,a_2,i_1,i_2] += Trm1[a_1,a_2,i_1,i_2]
    return R2u
end

function addres_gvvoo(R2u::ITensor,nv,nocc)
    println("ITensor is being used for gvvoo")
    g_vvoo = ITensor(deserialize("g_vvoo.jlbin"),a_1,a_2,i_1,i_2)
    Trm1 = ITensor(a_1,a_2,i_1,i_2)
    Trm1 = 0.5 * g_vvoo
    Trm1 = permute(Trm1, a_1, a_2, i_1, i_2, allow_alias=false)
    R2u += Trm1
    return R2u
    println("gvvoo processing complete")
end

function addres_gvovo(R2u::Array{Float64,4},nv,nocc,T2::Array{Float64,4})
    g_vovo = deserialize("g_vovo.jlbin")
    @tensor Trm3[a_1,a_2,i_1,i_2] := - g_vovo[a_2,i_3,a_3,i_2] * T2[a_1,a_3,i_1,i_3]
    @tensor R2u[a_1,a_2,i_1,i_2] += Trm3[a_1,a_2,i_1,i_2]
    @tensor Trm4[a_1,a_2,i_1,i_2] := - g_vovo[a_2,i_3,a_3,i_1] * T2[a_1,a_3,i_3,i_2]
    @tensor R2u[a_1,a_2,i_1,i_2] += Trm4[a_1,a_2,i_1,i_2]
    return R2u
end

function addres_gvovo(R2u::ITensor, nv, nocc, T2::ITensor)
    println("ITensor is being used for gvovo")
    i_3 = Index(nocc, "i_3")
    a_3 = Index(nv, "a_3")
    g_vovo = ITensor(deserialize("g_vovo.jlbin"), a_2, i_3, a_3, i_2)
    T2_perm1 = replaceinds(T2, inds(T2) => (a_1, a_3, i_1, i_3))
    Trm3 = -g_vovo * T2_perm1
    Trm3 = permute(Trm3, a_1, a_2, i_1, i_2, allow_alias=false)
    R2u += Trm3
    T2_perm2 = replaceinds(T2, inds(T2) => (a_1, a_3, i_3, i_2))
    g_vovo = replaceinds(g_vovo, inds(g_vovo) => (a_2, i_3, a_3, i_1)) # Corrected: Replace indices of g_vovo for the second term
    Trm4 = -g_vovo * T2_perm2
    Trm4 = permute(Trm4, a_1, a_2, i_1, i_2, allow_alias=false)
    R2u += Trm4
    return R2u
    println("gvovo processing complete")
end

function addres_goooo(R2u,nv,nocc,T2)
    g_oooo = deserialize("g_oooo.jlbin")
    @tensor Trm6[a_1,a_2,i_1,i_2] := 0.5*g_oooo[i_3,i_4,i_1,i_2] * T2[a_1,a_2,i_3,i_4]
    @tensor R2u[a_1,a_2,i_1,i_2] += Trm6[a_1,a_2,i_1,i_2]
    return R2u
end

function addres_goooo(R2u::ITensor, nv, nocc, T2::ITensor)
    println("ITensor is being used for goooo")
    i_3 = Index(nocc, "i_3")
    i_4 = Index(nocc, "i_4")
    g_oooo = ITensor(deserialize("g_oooo.jlbin"), i_3, i_4, i_1, i_2)
    T2 = replaceinds(T2, inds(T2) => (a_1, a_2, i_3, i_4))
    Trm6 = 0.5 * g_oooo * T2
    Trm6 = permute(Trm6, a_1, a_2, i_1, i_2, allow_alias=false)
    R2u += Trm6
    return R2u
    println("goooo processing complete")
end



function addres_gvvvv(R2u::Array{Float64,4},nv,nocc,T2::Array{Float64,4})
    g_vvvv = deserialize("g_vvvv.jlbin")
    @tensor Trm8[a_1,a_2,i_1,i_2] := + 0.5*g_vvvv[a_1,a_2,a_3,a_4] * T2[a_3,a_4,i_1,i_2]
    @tensor R2u[a_1,a_2,i_1,i_2] += Trm8[a_1,a_2,i_1,i_2]
    return R2u
end

function addres_gvvvv(R2u::ITensor, nv, nocc, T2::ITensor)
    println("ITensor is being used for gvvvv")
    a_3 = Index(nv, "a_3")
    a_4 = Index(nv, "a_4")
    g_vvvv = ITensor(deserialize("g_vvvv.jlbin"), a_1, a_2, a_3, a_4)
    T2 = replaceinds(T2, inds(T2) => (a_3, a_4, i_1, i_2))
    Trm8 = 0.5 * g_vvvv * T2
    Trm8 = permute(Trm8, a_1, a_2, i_1, i_2, allow_alias=false)
    R2u += Trm8
    return R2u
    println("gvvvv processing complete")
end

function addres_fvv(R2u,nv,nocc,T2)
    fvv = deserialize("fvv.jlbin")
    @tensor Trm7[a_1,a_2,i_1,i_2] := fvv[a_2,a_3] * T2[a_1,a_3,i_1,i_2]
    @tensor R2u[a_1,a_2,i_1,i_2] += Trm7[a_1,a_2,i_1,i_2]
    return R2u
end

function addres_fvv(R2u::ITensor, nv, nocc, T2::ITensor)
    println("ITensor is being used for fvv")
    a_3 = Index(nv, "a_3")
    fvv = ITensor(deserialize("fvv.jlbin"), a_2, a_3)
    T2 = replaceinds(T2, inds(T2) => (a_1, a_3, i_1, i_2))
    Trm7 = fvv * T2
    Trm7 = permute(Trm7, a_1, a_2, i_1, i_2, allow_alias=false)
    R2u += Trm7
    return R2u
    println("fvv processing complete")
end


function addres_foo(R2u,nv,nocc,T2)
    foo = deserialize("foo.jlbin")
    @tensor Trm9[a_1,a_2,i_1,i_2] := - foo[i_3,i_2] * T2[a_1,a_2,i_1,i_3]
    @tensor R2u[a_1,a_2,i_1,i_2] += Trm9[a_1,a_2,i_1,i_2]
    return R2u
end

function addres_foo(R2u::ITensor, nv, nocc, T2::ITensor)
    println("ITensor is being used for foo")
    i_3 = Index(nocc, "i_3")
    foo = ITensor(deserialize("foo.jlbin"), i_3, i_2)
    T2 = replaceinds(T2, inds(T2) => (a_1, a_2, i_1, i_3))
    Trm9 = -foo * T2
    Trm9 = permute(Trm9, a_1, a_2, i_1, i_2, allow_alias=false)
    R2u += Trm9
    return R2u
    println("foo processing complete")
end

function addres_goovvb1(R2u::Array{Float64},nv,nocc,T2::Array{Float64})
    g_oovv = deserialize("g_oovv.jlbin")
    @tensor B1[i_4,a_4,a_1,i_1] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_3,i_1,i_3])
    @tensor R2u[a_1,a_2,i_1,i_2] +=  B1[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_2,i_4]- B1[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_4,i_2]
    return R2u
end

function addres_goovvb1(R2u::ITensor,nv,nocc,T2::ITensor)
    println("ITensor is being used for goovvb1")
    i_3 = Index(nocc, "i_3")
    a_3 = Index(nv, "a_3")
    i_4 = Index(nocc, "i_4")
    a_4 = Index(nv, "a_4")
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    B1 = ITensor(i_4, a_4, a_1, i_1)
    B1 = 2 * g_oovv * replaceinds(T2, inds(T2) => (a_1, a_3, i_1, i_3))
    tmp1 = replaceinds(B1, inds(B1) => (i_4, a_4, a_1, i_1))*replaceinds(T2, inds(T2) => (a_2, a_4, i_2, i_4)) - replaceinds(B1, inds(B1) => (i_4, a_4, a_1, i_1))*replaceinds(T2, inds(T2) => (a_2, a_4, i_4, i_2))
    R2u += tmp1
    return R2u
    println("goovvb1 processing complete")

end

function addres_goovvb2(R2u::Array{Float64},nv,nocc,T2::Array{Float64})
    g_oovv = deserialize("g_oovv.jlbin")
    @tensor B2[i_4,a_4,a_1,i_1] := 0.5*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_3,i_3,i_1])
    @tensor R2u[a_1,a_2,i_1,i_2] += B2[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_4,i_2]
    return R2u
end

function addres_goovvb2(R2u::ITensor, nv, nocc, T2::ITensor)
    println("ITensor is being used for goovvb2")
    i_3 = Index(nocc, "i_3")
    a_3 = Index(nv, "a_3")
    i_4 = Index(nocc, "i_4")
    a_4 = Index(nv, "a_4")
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    B2 = ITensor(i_4, a_4, a_1, i_1)
    B2 = 0.5 * g_oovv * replaceinds(T2, inds(T2) => (a_1, a_3, i_3, i_1))
    tmp2 = replaceinds(B2, inds(B2) => (i_4, a_4, a_1, i_1)) * replaceinds(T2, inds(T2) => (a_2, a_4, i_4, i_2))
    R2u += tmp2
    return R2u
    println("goovvb2 processing complete")
end

function addres_goovvb3(R2u::Array{Float64,4},nv,nocc,T2::Array{Float64,4})
    g_oovv = deserialize("g_oovv.jlbin")
    @tensor B3[i_4,i_2] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_3,i_2])
    @tensor R2u[a_1,a_2,i_1,i_2] +=  -B3[i_4,i_2] * T2[a_1,a_2,i_1,i_4]
    return R2u
end

function addres_goovvb3(R2u::ITensor, nv, nocc, T2::ITensor)
    println("ITensor is being used for goovvb3")
    i_3 = Index(nocc, "i_3")
    a_3 = Index(nv, "a_3")
    i_4 = Index(nocc, "i_4")
    a_4 = Index(nv, "a_4")
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    B3 = ITensor(i_4, i_2)
    B3 = 2 * g_oovv * replaceinds(T2, inds(T2) => (a_3, a_4, i_3, i_2))
    tmp3 = -replaceinds(B3, inds(B3) => (i_4, i_2)) * replaceinds(T2, inds(T2) => (a_1, a_2, i_1, i_4))
    tmp3 = permute(tmp3, a_1, a_2, i_1, i_2, allow_alias=false)
    R2u += tmp3
    return R2u
    println("goovvb3 processing complete")
end


function addres_goovvb4(R2u::Array{Float64,4},nv,nocc,T2::Array{Float64,4})
    g_oovv = deserialize("g_oovv.jlbin")
    @tensor B4[i_4,a_3,a_1,i_1] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_4,i_1,i_3])
    @tensor R2u[a_1,a_2,i_1,i_2] += -B4[i_4,a_3,a_1,i_1] * T2[a_2,a_3,i_2,i_4] + B4[i_4,a_3,a_1,i_1] * T2[a_2,a_3,i_4,i_2]
    return R2u
end

function addres_goovvb4(R2u::ITensor, nv, nocc, T2::ITensor)
    println("ITensor is being used for goovvb4")
    i_3 = Index(nocc, "i_3")
    a_3 = Index(nv, "a_3")
    i_4 = Index(nocc, "i_4")
    a_4 = Index(nv, "a_4")
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    B4 = ITensor(i_4, a_3, a_1, i_1)
    B4 = g_oovv * replaceinds(T2, inds(T2) => (a_1, a_4, i_1, i_3))
    tmp4 = -replaceinds(B4, inds(B4) => (i_4, a_3, a_1, i_1)) * replaceinds(T2, inds(T2) => (a_2, a_3, i_2, i_4))
    tmp4 += replaceinds(B4, inds(B4) => (i_4, a_3, a_1, i_1)) * replaceinds(T2, inds(T2) => (a_2, a_3, i_4, i_2))
    tmp4 = permute(tmp4, a_1, a_2, i_1, i_2, allow_alias=false)
    R2u += tmp4
    return R2u
    println("goovvb4 processing complete")
end

function addres_goovvb5(R2u::Array{Float64,4},nv,nocc,T2::Array{Float64,4})
    g_oovv = deserialize("g_oovv.jlbin")
    @tensor B5[a_3,a_2] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_2,a_4,i_4,i_3])
    @tensor R2u[a_1,a_2,i_1,i_2] += B5[a_3,a_2] * T2[a_1,a_3,i_1,i_2]
    return R2u
end

function addres_goovvb5(R2u::ITensor, nv, nocc, T2::ITensor)
    println("Adjusting ITensor operation for goovvb5 without defining a_1, a_2, i_1, i_2 within the function")
    i_3 = Index(nocc, "i_3")
    a_3 = Index(nv, "a_3")
    i_4 = Index(nocc, "i_4")
    a_4 = Index(nv, "a_4")
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    B5 = g_oovv * replaceinds(T2, inds(T2) => (a_2, a_4, i_4, i_3))
    B5 = permute(B5, a_3, a_2, allow_alias=false)
    R2u += B5 * replaceinds(T2, inds(T2) => (a_1, a_3, i_1, i_2))
    return R2u
    println("goovvb5 processing complete")
end

function addres_goovvb6(R2u::Array{Float64,4},nv,nocc,T2::Array{Float64,4})
    g_oovv = deserialize("g_oovv.jlbin")
    @tensor B6[i_4,i_2] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_2,i_3])
    @tensor R2u[a_1,a_2,i_1,i_2] += B6[i_4,i_2] * T2[a_1,a_2,i_1,i_4]
    return R2u
end

function addres_goovvb6(R2u::ITensor, nv, nocc, T2::ITensor)
    println("Adjusting ITensor operation for goovvb6")
    i_3 = Index(nocc, "i_3")
    a_3 = Index(nv, "a_3")
    i_4 = Index(nocc, "i_4")
    a_4 = Index(nv, "a_4")
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    B6 = g_oovv * replaceinds(T2, inds(T2) => (a_3, a_4, i_2, i_3))
    B6 = permute(B6, i_4, i_2, allow_alias=false)
    R2u += B6 * replaceinds(T2, inds(T2) => (a_1, a_2, i_1, i_4))
    return R2u
    println("goovvb6 processing complete")
end

function addres_goovvb7(R2u::Array{Float64,4},nv,nocc,T2::Array{Float64,4})
    g_oovv = deserialize("g_oovv.jlbin")
    @tensor B7[a_4,a_2] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_2,a_3,i_4,i_3])
    @tensor R2u[a_1,a_2,i_1,i_2] += - B7[a_4,a_2] * T2[a_1,a_4,i_1,i_2]
    return R2u
end

function addres_goovvb7(R2u::ITensor, nv, nocc, T2::ITensor)
    println("Adjusting ITensor operation for goovvb7")
    i_3 = Index(nocc, "i_3")
    a_3 = Index(nv, "a_3")
    i_4 = Index(nocc, "i_4")
    a_4 = Index(nv, "a_4")
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    B7 = 2 * g_oovv * replaceinds(T2, inds(T2) => (a_2, a_3, i_4, i_3))
    B7 = permute(B7, a_4, a_2, allow_alias=false)
    R2u += -B7 * replaceinds(T2, inds(T2) => (a_1, a_4, i_1, i_2))
    return R2u
    println("goovvb7 processing complete")
end

function addres_goovvb8(R2u::Array{Float64,4},nv,nocc,T2::Array{Float64,4})
    g_oovv = deserialize("g_oovv.jlbin")
    @tensor B8[i_4,a_3,a_1,i_2] := 0.5*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_4,i_3,i_2])
    @tensor R2u[a_1,a_2,i_1,i_2] +=  +B8[i_4,a_3,a_1,i_2] * T2[a_2,a_3,i_4,i_1]
    return R2u
end

function addres_goovvb8(R2u::ITensor, nv, nocc, T2::ITensor)
    println("Adjusting ITensor operation for goovvb8")
    i_3 = Index(nocc, "i_3")
    a_3 = Index(nv, "a_3")
    i_4 = Index(nocc, "i_4")
    a_4 = Index(nv, "a_4")
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    B8 = 0.5 * g_oovv * replaceinds(T2, inds(T2) => (a_1, a_4, i_3, i_2))
    B8 = permute(B8, i_4, a_3, a_1, i_2, allow_alias=false)
    R2u += B8 * replaceinds(T2, inds(T2) => (a_2, a_3, i_4, i_1))
    return R2u
    println("goovvb8 processing complete")
end


function addres_goovvb9(R2u,nv,nocc,T2)
    g_oovv = deserialize("g_oovv.jlbin")
    @tensor B9[i_3,i_4,i_1,i_2] := 0.5* (g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_1,i_2])
    @tensor R2u[a_1,a_2,i_1,i_2] += + B9[i_3,i_4,i_1,i_2] * T2[a_1,a_2,i_3,i_4]
    return R2u
end

function addres_goovvb9(R2u::ITensor, nv, nocc, T2::ITensor)
    println("Adjusting ITensor operation for goovvb9")
    i_3 = Index(nocc, "i_3")
    i_4 = Index(nocc, "i_4")
    a_3 = Index(nv, "a_3")
    a_4 = Index(nv, "a_4")
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    B9 = 0.5 * g_oovv * replaceinds(T2, inds(T2) => (a_3, a_4, i_1, i_2))
    B9 = permute(B9, i_3, i_4, i_1, i_2, allow_alias=false)
    R2u += B9 * replaceinds(T2, inds(T2) => (a_1, a_2, i_3, i_4))
    return R2u
    println("goovvb9 processing complete")
end

function calcresnew(T2::Array{Float64,4})
    nv = deserialize("nv.jlbin")
    nocc = deserialize("nocc.jlbin")
    R2u = zeros(Float64,nv,nv,nocc,nocc)
    R2 =  zeros(Float64,nv,nv,nocc,nocc)
    R2u = addres_gvvoo(R2u,nv,nocc)
    R2u = addres_gvoov(R2u,nv,nocc,T2)
    R2u = addres_gvovo(R2u,nv,nocc,T2)
    R2u = addres_goooo(R2u,nv,nocc,T2)
    R2u = addres_gvvvv(R2u,nv,nocc,T2)
    R2u = addres_fvv(R2u,nv,nocc,T2)
    R2u = addres_foo(R2u,nv,nocc,T2)
    # R2u = addres_goovv(R2u,nv,nocc,T2)  #In this all the BX terms are still in memory at once
    R2u = addres_goovvb1(R2u,nv,nocc,T2)
    R2u = addres_goovvb2(R2u,nv,nocc,T2)
    R2u = addres_goovvb3(R2u,nv,nocc,T2)
    R2u = addres_goovvb4(R2u,nv,nocc,T2)
    R2u = addres_goovvb5(R2u,nv,nocc,T2)
    R2u = addres_goovvb6(R2u,nv,nocc,T2)
    R2u = addres_goovvb7(R2u,nv,nocc,T2)
    R2u = addres_goovvb8(R2u,nv,nocc,T2)
    R2u = addres_goovvb9(R2u,nv,nocc,T2)
    @tensor R2[a,b,i,j] += R2u[a,b,i,j] + R2u[b,a,j,i]
    return R2
end

function calcresnew(T2::ITensor)
    nv = deserialize("nv.jlbin")
    nocc = deserialize("nocc.jlbin")
    R2u = ITensor(zeros(Float64,nv,nv,nocc,nocc),a_1,a_2,i_1,i_2)
    R2  = ITensor(zeros(Float64,nv,nv,nocc,nocc),a_1,a_2,i_1,i_2)
    R2u = addres_gvvoo(R2u,nv,nocc)
    R2u = addres_gvoov(R2u,nv,nocc,T2)
    R2u = addres_gvovo(R2u,nv,nocc,T2)
    R2u = addres_goooo(R2u,nv,nocc,T2)
    R2u = addres_gvvvv(R2u,nv,nocc,T2)
    R2u = addres_fvv(R2u,nv,nocc,T2)
    R2u = addres_foo(R2u,nv,nocc,T2)
    R2u = addres_goovvb1(R2u,nv,nocc,T2)
    R2u = addres_goovvb2(R2u,nv,nocc,T2)
    R2u = addres_goovvb3(R2u,nv,nocc,T2)
    R2u = addres_goovvb4(R2u,nv,nocc,T2)
    R2u = addres_goovvb5(R2u,nv,nocc,T2)
    R2u = addres_goovvb6(R2u,nv,nocc,T2)
    R2u = addres_goovvb7(R2u,nv,nocc,T2)
    R2u = addres_goovvb8(R2u,nv,nocc,T2)
    R2u = addres_goovvb9(R2u,nv,nocc,T2)
    R2 = R2 + permute(R2,a_2,a_1,i_2,i_1)
    return R2
end

########################################################
begin
    initialize_cc_variables()
    T2 = initialize_t2_only();
    # a_1 = Index(nv,"a_1")
    # a_2 = Index(nv,"a_2")
    # i_1 = Index(nocc,"i_1")
    # i_2 = Index(nocc,"i_2")
    iT2 = ITensor(initialize_t2_only(),a_1,a_2,i_1,i_2);
    R2u = zeros(Float64,nv,nv,nocc,nocc);
    iR2u = ITensor(copy(R2u),a_1,a_2,i_1,i_2);
end
a_1 = Index(nv,"a_1")
a_2 = Index(nv,"a_2")
i_1 = Index(nocc,"i_1")
i_2 = Index(nocc,"i_2")
T2 = initialize_t2_only();
R2 = calcresnew(T2)
iT2 = ITensor(initialize_t2_only(),a_1,a_2,i_1,i_2);
iR2 = calcresnew(iT2)

@show iR2

















####### clipbaord


function addres_gvoov(R2u::ITensor,nv,nocc,T2::ITensor)
    i_3 = Index(nocc,"i_3")
    a_3 = Index(nv,"a_3")
    g_voov = ITensor(deserialize("g_voov.jlbin"),a_1,i_3,i_1,a_3)
    T2 = replaceinds(T2, inds(T2) => (a_2, a_3, i_3, i_2))
    R2u = replaceinds(R2u, inds(R2u) => (a_1, a_2, i_1, i_2))
    Trm2 = ITensor(a_1,a_2,i_1,i_2)
    Trm5 = ITensor(a_1,a_2,i_1,i_2)
    Trm2 = g_voov * T2
    Trm2 = permute(Trm2,a_1,a_2,i_1,i_2,allow_alias=false)
    R2u += Trm2
    # @show R2u
    T2 = replaceinds(T2, inds(T2) => (a_2, a_3, i_2,i_3))
    Trm5 = 2*g_voov * T2
    Trm5 = permute(Trm5,a_1,a_2,i_1,i_2,allow_alias=false)
    R2u += Trm5
    return R2u
end