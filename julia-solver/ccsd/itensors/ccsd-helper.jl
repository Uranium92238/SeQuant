using DelimitedFiles, TensorOperations, LinearAlgebra, MKL, Serialization, ITensors
using ITensors: permute


# function SquareNorm(R2::Array{Float64,4}) 
#     # The norm taken by Julia and that of Molpro seem to be different due to symmetry of the tensor
#     # This is supposed to have no effect on the final Convergence  but...
#     # This was done to match the norm of Molpro. Same logic applies for the dot product....
#     nv = size(R2)[1]
#     no = size(R2)[3]
#     flatdims = nv*no
#     R = zeros(flatdims,flatdims)
#     for a in 1:nv , b in 1:nv , i in 1:no , j in 1:no
#         R[a+nv*(i-1),b+nv*(j-1)] = R2[a,b,i,j]
#     end
#     norm = 0.0
#     flag = 0
#     for X in 1:flatdims
#         for Y in 1:X
#             norm += R[X,Y]*R[X,Y]
#             flag +=1
#         end
#     end
#     # norm = sqrt(abs(dot(R2,R2))) #Use this for Julia Norm
#     return norm
# end

# function SquareDot(R2_i::Array{Float64,4},R2_j::Array{Float64,4})
#     nv = size(R2_i)[1]
#     no = size(R2_i)[3]
#     flatdims = nv*no
#     Ri = zeros(flatdims,flatdims)
#     Rj = zeros(flatdims,flatdims)
#     for a in 1:nv , b in 1:nv , i in 1:no , j in 1:no
#         Ri[a+nv*(i-1),b+nv*(j-1)] = R2_i[a,b,i,j]
#         Rj[a+nv*(i-1),b+nv*(j-1)] = R2_j[a,b,i,j]
#     end
#     dot = 0.0
#     flag = 0
#     for X in 1:flatdims
#         for Y in 1:X
#             dot += Ri[X,Y]*Rj[X,Y]
#             flag +=1
#         end
#     end
#     # @tensor ndot = R2_i[a,b,i,j]*R2_j[a,b,i,j]
#     return dot
# end



function main(pathtofcidump)
    linenum = 0
    norb = 0
    nelec = 0
    begin # Finds out from where the main integral data starts
        file = open(pathtofcidump, "r")
        for line in eachline(file)
            linenum += 1
            if strip(line) == "/"
                # println("The line number where integral data starts is: ", linenum)
                break
            end
        end
    end

    begin
        # Open the file
        file = open(pathtofcidump, "r")

        # Loop over each line in the file
        for line in eachline(file)
            # Check if the line starts with "&FCI NORB="
            if startswith(line, " &FCI NORB=")
                line = split(line, ",")
                words1 = split(line[1], " ")
                words2 = split(line[2], " ")
                norb = parse(Int, words1[end])
                nelec = parse(Int, words2[end])
                break
            end
        end

        # Close the file
        close(file)
    end
    data = readdlm(pathtofcidump, skipstart=linenum)
    hnuc = data[end, 1]


    data = copy(data[1:end-1, :])
    h = Array{Union{Missing,Float64}}(missing, norb, norb)
    g = Array{Union{Missing,Float64}}(missing, norb, norb, norb, norb)
    l = length(data[:, 1])
    non_redundant_indices = []
    for i in 1:l
        if (data[i, 4] == 0 && data[i, 5] == 0)
            I = round(Int, data[i, 2])
            J = round(Int, data[i, 3])
            h[I, J] = data[i, 1]
        else
            I = round(Int, data[i, 2])
            J = round(Int, data[i, 3])
            K = round(Int, data[i, 4])
            L = round(Int, data[i, 5])
            push!(non_redundant_indices, [I, J, K, L])
            g[I, J, K, L] = data[i, 1]
        end
    end
    for (I, J, K, L) in non_redundant_indices
        open("non_redundant_indices.txt", "a") do f
            println(f, I, J, K, L, "   ", g[I, J, K, L])
        end
        g[K, L, I, J] = g[I, J, K, L]
        g[J, I, L, K] = g[I, J, K, L]
        g[L, K, J, I] = g[I, J, K, L]
        g[J, I, K, L] = g[I, J, K, L]
        g[L, K, I, J] = g[I, J, K, L]
        g[I, J, L, K] = g[I, J, K, L]
        g[K, L, J, I] = g[I, J, K, L]

    end
    for I in 1:norb
        for J in 1:I
            h[J, I] = h[I, J]
        end
    end
    no = round(Int, nelec / 2)
    nv = norb - no
    h = convert(Array{Float64}, h)
    g = convert(Array{Float64}, g)
    K = zeros(nv, nv, no, no)
    for a in 1:nv, b in 1:nv, i in 1:no, j in 1:no
        K[a, b, i, j] = g[no+a, i, no+b, j]
    end
    H1 = 0.0
    H2 = 0.0
    for i in 1:no
        H1 = H1 + h[i, i]
        for j in 1:no
            H2 = H2 + 2 * g[i, i, j, j] - g[i, j, i, j]
        end
    end
    erhf = 2 * H1 + H2 + hnuc
    f = zeros(size(h)) # fock matrix Initialization
    for p in 1:norb, q in 1:norb
        s_pq = 0
        for i in 1:no
            s_pq += 2 * g[p, q, i, i] - g[p, i, i, q]
        end
        f[p, q] = h[p, q] + s_pq
    end
    serialize("h.jlbin", h)
    serialize("g.jlbin", g)
    serialize("hnuc.jlbin", hnuc)
    serialize("norbs.jlbin", norb)
    serialize("nelec.jlbin", nelec)
    serialize("no.jlbin", no)
    serialize("nv.jlbin", nv)
    serialize("K.jlbin", K)
    serialize("erhf.jlbin", erhf)
    serialize("f.jlbin", f)
end

function initialize_t2_only()
    nv = deserialize("nv.jlbin")
    no = deserialize("no.jlbin")
    ################## Amplitude Initialization ######################################
    K = deserialize("K.jlbin")
    f = deserialize("f.jlbin")
    T2::Array{Float64,4} = -K
    for a in 1:nv, b in 1:nv, i in 1:no, j in 1:no
        T2[a, b, i, j] = T2[a, b, i, j] / (f[no+a, no+a] + f[no+b, no+b] - f[i, i] - f[j, j])
        # if abs(T2[a,b,i,j]) < 10e-8
        #     T2[a,b,i,j] = 0.0
        # end
    end

    ##################################################################################
    return T2
end
function initialize_cc_variables()
    main("water_dump.fcidump")
    K = deserialize("K.jlbin")
    f = deserialize("f.jlbin")
    g = deserialize("g.jlbin")
    no = deserialize("no.jlbin")
    nv = deserialize("nv.jlbin")
    g_vvoo = zeros(nv, nv, no, no)
    g_voov = zeros(nv, no, no, nv)
    g_vovo = zeros(nv, no, nv, no)
    g_oovv = zeros(no, no, nv, nv)
    g_oooo = zeros(no, no, no, no)
    g_vvvv = zeros(nv, nv, nv, nv)
    g_oovo = zeros(no, no, nv, no)
    g_vovv = zeros(nv, no, nv, nv)
    g_vvvo = zeros(nv, nv, nv, no)
    f_vv = zeros(nv, nv)
    f_oo = zeros(no, no)
    f_vo = zeros(nv, no)
    f_ov = zeros(no, nv)
    g_vooo = zeros(nv, no, no, no)
    for a in 1:nv, b in 1:nv
        f_vv[a, b] = f[no+a, no+b]

        for i in 1:no, j in 1:no
            g_vvoo[a, b, i, j] = g[no+a, i, no+b, j]
            g_voov[a, i, j, b] = g[no+a, j, i, no+b]
            g_vovo[a, i, b, j] = g[no+a, no+b, i, j]
            g_oovv[i, j, a, b] = g[i, no+a, j, no+b]
            f_oo[i, j] = f[i, j]
        end
    end
    for i in 1:no, j in 1:no, k in 1:no, l in 1:no
        g_oooo[i, j, k, l] = g[i, k, j, l]
    end
    for a in 1:nv, b in 1:nv, c in 1:nv, d in 1:nv
        g_vvvv[a, b, c, d] = g[no+a, no+c, no+b, no+d]
    end
    for a in 1:nv, i in 1:no, j in 1:no, k in 1:no
        g_vooo[a, i, j, k] = g[no+a, j, i, k]
    end
    for i in 1:no, j in 1:no, a in 1:nv, k in 1:no
        g_oovo[i, j, a, k] = g[i, no+a, j, k]
    end
    for a in 1:nv, i in 1:no, b in 1:nv, c in 1:nv
        g_vovv[a, i, b, c] = g[no+a, no+b, i, no+c]
        g_vvvo[a, b, c, i] = g[no+a, no+c, no+b, i]
    end

    for i in 1:no, a in 1:nv
        # f_vo[a, i] = f[no+a, i]
        # f_ov[i, a] = f[i, no+a]
        f_vo[a, i] = 0.0
        f_ov[i, a] = 0.0
    end
    serialize("g_vvoo.jlbin", g_vvoo)
    serialize("g_voov.jlbin", g_voov)
    serialize("g_vovo.jlbin", g_vovo)
    serialize("g_oovv.jlbin", g_oovv)
    serialize("g_oooo.jlbin", g_oooo)
    serialize("g_vvvv.jlbin", g_vvvv)
    serialize("g_oovo.jlbin", g_oovo)
    serialize("g_vovv.jlbin", g_vovv)
    serialize("g_vvvo.jlbin", g_vvvo)
    serialize("K.jlbin", K)
    serialize("f_vv.jlbin", f_vv)
    serialize("f_oo.jlbin", f_oo)
    serialize("f_vo.jlbin", f_vo)
    serialize("f_ov.jlbin", f_ov)
    serialize("g_vooo.jlbin", g_vooo)
end






function calculate_residual(T2, R2u, R2)
    g_vvoo, g_voov, g_vovo, g_oovv, g_oooo, g_vvvv = deserialize("g_vvoo.jlbin"), deserialize("g_voov.jlbin"), deserialize("g_vovo.jlbin"), deserialize("g_oovv.jlbin"), deserialize("g_oooo.jlbin"), deserialize("g_vvvv.jlbin")
    f_vv, f_oo = deserialize("f_vv.jlbin"), deserialize("f_oo.jlbin")
    @tensor begin
        R2u[a_1, a_2, i_1, i_2] = 0.5 * g_vvoo[a_1, a_2, i_1, i_2] - g_voov[a_1, i_3, i_1, a_3] * T2[a_2, a_3, i_3, i_2] - g_vovo[a_2, i_3, a_3, i_2] * T2[a_1, a_3, i_1, i_3] - g_vovo[a_2, i_3, a_3, i_1] * T2[a_1, a_3, i_3, i_2] + 2 * g_voov[a_1, i_3, i_1, a_3] * T2[a_2, a_3, i_2, i_3] + 0.5 * g_oooo[i_3, i_4, i_1, i_2] * T2[a_1, a_2, i_3, i_4] + f_vv[a_2, a_3] * T2[a_1, a_3, i_1, i_2] + 0.5 * g_vvvv[a_1, a_2, a_3, a_4] * T2[a_3, a_4, i_1, i_2] - f_oo[i_3, i_2] * T2[a_1, a_2, i_1, i_3] + 2 * (g_oovv[i_3, i_4, a_3, a_4] * T2[a_1, a_3, i_1, i_3]) * T2[a_2, a_4, i_2, i_4] - 2 * (g_oovv[i_3, i_4, a_3, a_4] * T2[a_1, a_3, i_1, i_3]) * T2[a_2, a_4, i_4, i_2] + 0.5 * (g_oovv[i_3, i_4, a_3, a_4] * T2[a_1, a_3, i_3, i_1]) * T2[a_2, a_4, i_4, i_2] - 2 * (g_oovv[i_3, i_4, a_3, a_4] * T2[a_3, a_4, i_3, i_2]) * T2[a_1, a_2, i_1, i_4] - 1 * (g_oovv[i_3, i_4, a_3, a_4] * T2[a_1, a_4, i_1, i_3]) * T2[a_2, a_3, i_2, i_4] + (g_oovv[i_3, i_4, a_3, a_4] * T2[a_1, a_4, i_1, i_3]) * T2[a_2, a_3, i_4, i_2] + (g_oovv[i_3, i_4, a_3, a_4] * T2[a_2, a_4, i_4, i_3]) * T2[a_1, a_3, i_1, i_2] + (g_oovv[i_3, i_4, a_3, a_4] * T2[a_3, a_4, i_2, i_3]) * T2[a_1, a_2, i_1, i_4] - 2 * (g_oovv[i_3, i_4, a_3, a_4] * T2[a_2, a_3, i_4, i_3]) * T2[a_1, a_4, i_1, i_2] + 0.5 * (g_oovv[i_3, i_4, a_3, a_4] * T2[a_1, a_4, i_3, i_2]) * T2[a_2, a_3, i_4, i_1] + 0.5 * (g_oovv[i_3, i_4, a_3, a_4] * T2[a_3, a_4, i_1, i_2]) * T2[a_1, a_2, i_3, i_4]
        R2[a, b, i, j] = R2u[a, b, i, j] + R2u[b, a, j, i]
    end
    return R2
end

function calculate_residual_memeff_gc(T2)
    g_vvoo, g_voov, g_vovo, g_oovv, g_oooo, g_vvvv = deserialize("g_vvoo.jlbin"), deserialize("g_voov.jlbin"), deserialize("g_vovo.jlbin"), deserialize("g_oovv.jlbin"), deserialize("g_oooo.jlbin"), deserialize("g_vvvv.jlbin")
    f_vv, f_oo = deserialize("f_vv.jlbin"), deserialize("f_oo.jlbin")
    nv = deserialize("nv.jlbin")
    no = deserialize("no.jlbin")
    R2u = zeros(Float64, nv, nv, no, no)
    R2 = zeros(Float64, nv, nv, no, no)
    @tensor begin
        Trm1[a_1, a_2, i_1, i_2] := 0.5 * g_vvoo[a_1, a_2, i_1, i_2]
        R2u[a_1, a_2, i_1, i_2] += Trm1[a_1, a_2, i_1, i_2]
        @notensor Trm1 = nothing
        @notensor g_vvoo = nothing
        GC.gc()
        Trm2[a_1, a_2, i_1, i_2] := -g_voov[a_1, i_3, i_1, a_3] * T2[a_2, a_3, i_3, i_2]
        R2u[a_1, a_2, i_1, i_2] += Trm2[a_1, a_2, i_1, i_2]
        @notensor Trm2 = nothing
        GC.gc()
        Trm3[a_1, a_2, i_1, i_2] := -g_vovo[a_2, i_3, a_3, i_2] * T2[a_1, a_3, i_1, i_3]
        R2u[a_1, a_2, i_1, i_2] += Trm3[a_1, a_2, i_1, i_2]
        @notensor Trm3 = nothing
        GC.gc()
        Trm4[a_1, a_2, i_1, i_2] := -g_vovo[a_2, i_3, a_3, i_1] * T2[a_1, a_3, i_3, i_2]
        R2u[a_1, a_2, i_1, i_2] += Trm4[a_1, a_2, i_1, i_2]
        @notensor Trm4 = nothing
        @notensor g_vovo = nothing
        GC.gc()
        Trm5[a_1, a_2, i_1, i_2] := 2 * g_voov[a_1, i_3, i_1, a_3] * T2[a_2, a_3, i_2, i_3]
        R2u[a_1, a_2, i_1, i_2] += Trm5[a_1, a_2, i_1, i_2]
        @notensor Trm5 = nothing
        @notensor g_voov = nothing
        GC.gc()
        Trm6[a_1, a_2, i_1, i_2] := 0.5 * g_oooo[i_3, i_4, i_1, i_2] * T2[a_1, a_2, i_3, i_4]
        R2u[a_1, a_2, i_1, i_2] += Trm6[a_1, a_2, i_1, i_2]
        @notensor Trm6 = nothing
        @notensor g_oooo = nothing
        GC.gc()
        Trm7[a_1, a_2, i_1, i_2] := f_vv[a_2, a_3] * T2[a_1, a_3, i_1, i_2]
        R2u[a_1, a_2, i_1, i_2] += Trm7[a_1, a_2, i_1, i_2]
        @notensor Trm7 = nothing
        @notensor f_vv = nothing
        GC.gc()
        Trm8[a_1, a_2, i_1, i_2] := +0.5 * g_vvvv[a_1, a_2, a_3, a_4] * T2[a_3, a_4, i_1, i_2]
        R2u[a_1, a_2, i_1, i_2] += Trm8[a_1, a_2, i_1, i_2]
        @notensor Trm8 = nothing
        @notensor g_vvvv = nothing
        GC.gc()
        Trm9[a_1, a_2, i_1, i_2] := -f_oo[i_3, i_2] * T2[a_1, a_2, i_1, i_3]
        R2u[a_1, a_2, i_1, i_2] += Trm9[a_1, a_2, i_1, i_2]
        @notensor Trm9 = nothing
        @notensor f_oo = nothing
        GC.gc()
        B1[i_4, a_4, a_1, i_1] := 2 * (g_oovv[i_3, i_4, a_3, a_4] * T2[a_1, a_3, i_1, i_3])
        R2u[a_1, a_2, i_1, i_2] += B1[i_4, a_4, a_1, i_1] * T2[a_2, a_4, i_2, i_4] - B1[i_4, a_4, a_1, i_1] * T2[a_2, a_4, i_4, i_2]
        @notensor B1 = nothing
        GC.gc()
        B2[i_4, a_4, a_1, i_1] := 0.5 * (g_oovv[i_3, i_4, a_3, a_4] * T2[a_1, a_3, i_3, i_1])
        R2u[a_1, a_2, i_1, i_2] += B2[i_4, a_4, a_1, i_1] * T2[a_2, a_4, i_4, i_2]
        @notensor B2 = nothing
        GC.gc()
        B3[i_4, i_2] := 2 * (g_oovv[i_3, i_4, a_3, a_4] * T2[a_3, a_4, i_3, i_2])
        R2u[a_1, a_2, i_1, i_2] += -B3[i_4, i_2] * T2[a_1, a_2, i_1, i_4]
        @notensor B3 = nothing
        GC.gc()
        B4[i_4, a_3, a_1, i_1] := (g_oovv[i_3, i_4, a_3, a_4] * T2[a_1, a_4, i_1, i_3])
        R2u[a_1, a_2, i_1, i_2] += -B4[i_4, a_3, a_1, i_1] * T2[a_2, a_3, i_2, i_4] + B4[i_4, a_3, a_1, i_1] * T2[a_2, a_3, i_4, i_2]
        @notensor B4 = nothing
        GC.gc()
        B5[a_3, a_2] := (g_oovv[i_3, i_4, a_3, a_4] * T2[a_2, a_4, i_4, i_3])
        R2u[a_1, a_2, i_1, i_2] += B5[a_3, a_2] * T2[a_1, a_3, i_1, i_2]
        @notensor B5 = nothing
        GC.gc()
        B6[i_4, i_2] := (g_oovv[i_3, i_4, a_3, a_4] * T2[a_3, a_4, i_2, i_3])
        R2u[a_1, a_2, i_1, i_2] += B6[i_4, i_2] * T2[a_1, a_2, i_1, i_4]
        @notensor B6 = nothing
        GC.gc()
        B7[a_4, a_2] := 2 * (g_oovv[i_3, i_4, a_3, a_4] * T2[a_2, a_3, i_4, i_3])
        R2u[a_1, a_2, i_1, i_2] += -B7[a_4, a_2] * T2[a_1, a_4, i_1, i_2]
        @notensor B7 = nothing
        GC.gc()
        B8[i_4, a_3, a_1, i_2] := 0.5 * (g_oovv[i_3, i_4, a_3, a_4] * T2[a_1, a_4, i_3, i_2])
        R2u[a_1, a_2, i_1, i_2] += +B8[i_4, a_3, a_1, i_2] * T2[a_2, a_3, i_4, i_1]
        @notensor B8 = nothing
        GC.gc()
        B9[i_3, i_4, i_1, i_2] := 0.5 * (g_oovv[i_3, i_4, a_3, a_4] * T2[a_3, a_4, i_1, i_2])
        R2u[a_1, a_2, i_1, i_2] += +B9[i_3, i_4, i_1, i_2] * T2[a_1, a_2, i_3, i_4]
        @notensor B9 = nothing
        @notensor g_oovv = nothing
        GC.gc()
        R2[a, b, i, j] := R2u[a, b, i, j] + R2u[b, a, j, i]
        @notensor R2u = nothing
        GC.gc()
    end
    return R2
end

function calculate_ECCD()
    nv = deserialize("nv.jlbin")
    no = deserialize("no.jlbin")
    i_1 = Index(no, "i_1")
    i_2 = Index(no, "i_2")
    a_1 = Index(nv, "a_1")
    a_2 = Index(nv, "a_2")



    tmpvar = 0.0
    Z = ITensor(tmpvar)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_1, i_2, a_1, a_2)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_2, i_1, i_2)
    Z += 2 * g_oovv * T2_vvoo
    T2_vvoo = nothing
    g_oovv = nothing
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_1, i_2, a_1, a_2)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_2, i_2, i_1)
    Z += -1 * g_oovv * T2_vvoo
    T2_vvoo = nothing
    g_oovv = nothing
    return Z
end



function update_amplitudes(T2::ITensor, R2::ITensor, Scaled_R2::ITensor,T1::ITensor, R1::ITensor, Scaled_R1::ITensor)
    a_1, a_2, i_1, i_2 = inds(T2)
    f_vv = ITensor(deserialize("f_vv.jlbin"), a_1, a_2)
    f_oo = ITensor(deserialize("f_oo.jlbin"), i_1, i_2)
    nv = deserialize("nv.jlbin")::Int64
    no = deserialize("no.jlbin")::Int64
    shiftp,shifts = 0.20,0.15

    #Update Doubles
    for a in 1:nv, b in 1:nv, i in 1:no, j in 1:no
        Scaled_R2[a, b, i, j] = R2[a, b, i, j] / (f_vv[a, a] + f_vv[b, b] - f_oo[i, i] - f_oo[j, j] + shiftp)
    end
    for a in 1:nv, b in 1:nv, i in 1:no, j in 1:no
        T2[a, b, i, j] = T2[a, b, i, j] - Scaled_R2[a, b, i, j]
    end

    #Update Singles
    for a in 1:nv, i in 1:no
        Scaled_R1[a,i] = R1[a,i]/(f_vv[a,a] - f_oo[i,i]+shifts)
    end

    for a in 1:nv, i in 1:no
        T1[a, i] = T1[a, i] - Scaled_R1[a,i]
    end
    return T2,T1
end


function check_convergence(R1,R2, normtol, e_old, e_new, etol)
    rnorm = norm(R1) + norm(R2)
    if (rnorm < normtol && abs(e_new[1] - e_old[1]) < etol)
        return true, rnorm
    else
        return false, rnorm
    end

end

function calculate_R_iter(R_iter, R2)
    nv = deserialize("nv.jlbin")
    no = deserialize("no.jlbin")
    f_vv = deserialize("f_vv.jlbin")
    f_oo = deserialize("f_oo.jlbin")
    shiftp = 0.20
    for a in 1:nv, b in 1:nv, i in 1:no, j in 1:no
        R_iter[a, b, i, j] = R2[a, b, i, j] / (f_vv[a, a] + f_vv[b, b] - f_oo[i, i] - f_oo[j, j] + shiftp)
    end
    return R_iter
end


function just_show_Bmatrix(R_iter_storage, p, T2_storage, R2_storage)
    B = zeros(p + 1, p + 1)
    Bpp = dot(R_iter_storage[p], R_iter_storage[p])
    for i in 1:p
        for j in 1:i
            # B[i,j] = SquareDot(R_iter_storage[i],R_iter_storage[j])/Bpp
            B[i, j] = dot(R2_storage[i], R2_storage[j])
            B[j, i] = B[i, j]
        end
    end
    # display(@views B[1:p,1:p])
    B[p+1, 1:p] .= -1
    B[1:p, p+1] .= -1
    B[p+1, p+1] = 0
    Id = zeros(p + 1)
    Id[p+1] = -1
    C = B \ Id
    pop!(C) # As C[p+1] is the Lagrange multiplier
    flag = 0
    s = zeros(size(R_iter_storage[1]))
    rs = zeros(size(R_iter_storage[1]))
    r = zeros(size(R_iter_storage[1]))
    for i in 1:p
        s = s + C[i] .* (R_iter_storage[i] + T2_storage[i])   # t⁽ᵏ⁺¹⁾=∑cₖ(t⁽ᵏ⁾+Δt⁽ᵏ⁾)
        rs = rs + C[i] .* R_iter_storage[i]
        r = r + C[i] .* R2_storage[i]
    end
    errnorm = sqrt(abs(dot(rs, rs)))
    println("Current Norms of Scaled Residuals: ", [sqrt(abs(dot(R_iter_storage[i], R_iter_storage[i]))) for i in 1:p])
    println("Current error vector norm  = $(errnorm)")
    # display(B)
    # display(C)
end

function PerformDiis(p,T2_storage,R2_storage,T1_storage,R1_storage,dummycall)
    B = zeros(p + 1, p + 1)
    
    #Replacing Indices because dot products are only permissible between tensors with same indices
    a_1, a_2, i_1, i_2 = inds(T2_storage[1])
    for i in 1:p
        R2_storage[i] = replaceinds(R2_storage[i], inds(R2_storage[i]) => (a_1, a_2, i_1, i_2))
        R1_storage[i] = replaceinds(R1_storage[i], inds(R1_storage[i]) => (a_1, i_1))
    end
    #Populating B matrix
    for i in 1:p
        for j in 1:i
            B[i, j] = dot(R2_storage[i], R2_storage[j]) + dot(R1_storage[i], R1_storage[j])
            B[j, i] = B[i, j]
        end
    end
    B[p+1, 1:p] .= -1
    B[1:p, p+1] .= -1
    B[p+1, p+1] = 0
    Id = zeros(p + 1)
    Id[p+1] = -1

    #Solving system to find coefficients
    C = B \ Id
    pop!(C) # As C[p+1] is the Lagrange multiplier

    #Constructing new amplitudes as a linear combination of old amplitudes with coefficients C
    t2 = ITensor(zeros(size(R2_storage[1])), a_1, a_2, i_1, i_2)
    t1 = ITensor(zeros(size(R1_storage[1])), a_1, i_1)
    for i in 1:p
        t2 = t2 + C[i] .* T2_storage[i]
        t1 = t1 + C[i] .* T1_storage[i]
    end
    return t2,t1
end






function calc_R2()
    nv = deserialize("nv.jlbin")
    no = deserialize("no.jlbin")
    a_1 = Index(nv, "a_1")
    a_2 = Index(nv, "a_2")
    a_3 = Index(nv, "a_3")
    a_4 = Index(nv, "a_4")
    i_1 = Index(no, "i_1")
    i_2 = Index(no, "i_2")
    i_3 = Index(no, "i_3")
    i_4 = Index(no, "i_4")


    I_vvoo = ITensor(zeros(Float64, nv, nv, no, no), a_1, a_2, i_1, i_2)
    g_oooo = ITensor(deserialize("g_oooo.jlbin"), i_3, i_4, i_1, i_2)
    INTpp_vvoo = ITensor(deserialize("INTpp_vvoo.jlbin"), a_1, a_2, i_3, i_4)
    g_oooo = replaceinds(g_oooo, inds(g_oooo) => (i_3, i_4, i_1, i_2))
    INTpp_vvoo = replaceinds(INTpp_vvoo, inds(INTpp_vvoo) => (a_1, a_2, i_3, i_4))
    I_vvoo += 1 / 2 * g_oooo * INTpp_vvoo
    INTpp_vvoo = nothing
    g_oooo = nothing
    g_vvoo = ITensor(deserialize("g_vvoo.jlbin"), a_1, a_2, i_1, i_2)
    g_vvoo = replaceinds(g_vvoo, inds(g_vvoo) => (a_1, a_2, i_1, i_2))
    I_vvoo += 1 / 2 * g_vvoo
    g_vvoo = nothing
    g_vvvo = ITensor(deserialize("g_vvvo.jlbin"), a_2, a_1, a_3, i_1)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_3, i_2)
    g_vvvo = replaceinds(g_vvvo, inds(g_vvvo) => (a_2, a_1, a_3, i_1))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_3, i_2))
    I_vvoo += g_vvvo * T1_vo
    T1_vo = nothing
    g_vvvo = nothing
    g_voov = ITensor(deserialize("g_voov.jlbin"), a_2, i_3, i_2, a_3)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_3, i_3, i_1)
    g_voov = replaceinds(g_voov, inds(g_voov) => (a_2, i_3, i_2, a_3))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_3, i_3, i_1))
    I_vvoo += -1 * g_voov * T2_vvoo
    T2_vvoo = nothing
    g_voov = nothing
    g_vooo = ITensor(deserialize("g_vooo.jlbin"), a_1, i_3, i_1, i_2)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_2, i_3)
    g_vooo = replaceinds(g_vooo, inds(g_vooo) => (a_1, i_3, i_1, i_2))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_2, i_3))
    I_vvoo += -1 * g_vooo * T1_vo
    T1_vo = nothing
    g_vooo = nothing
    g_vovo = ITensor(deserialize("g_vovo.jlbin"), a_2, i_3, a_3, i_1)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_3, i_3, i_2)
    g_vovo = replaceinds(g_vovo, inds(g_vovo) => (a_2, i_3, a_3, i_1))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_3, i_3, i_2))
    I_vvoo += -1 * g_vovo * T2_vvoo
    T2_vvoo = nothing
    g_vovo = nothing
    g_voov = ITensor(deserialize("g_voov.jlbin"), a_2, i_3, i_2, a_3)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_3, i_1, i_3)
    g_voov = replaceinds(g_voov, inds(g_voov) => (a_2, i_3, i_2, a_3))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_3, i_1, i_3))
    I_vvoo += 2 * g_voov * T2_vvoo
    T2_vvoo = nothing
    g_voov = nothing
    g_vovo = ITensor(deserialize("g_vovo.jlbin"), a_1, i_3, a_3, i_1)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_2, a_3, i_2, i_3)
    g_vovo = replaceinds(g_vovo, inds(g_vovo) => (a_1, i_3, a_3, i_1))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_2, a_3, i_2, i_3))
    I_vvoo += -1 * g_vovo * T2_vvoo
    T2_vvoo = nothing
    g_vovo = nothing
    f_vv = ITensor(deserialize("f_vv.jlbin"), a_2, a_3)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_3, i_1, i_2)
    f_vv = replaceinds(f_vv, inds(f_vv) => (a_2, a_3))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_3, i_1, i_2))
    I_vvoo += f_vv * T2_vvoo
    T2_vvoo = nothing
    f_vv = nothing
    g_vvvv = ITensor(deserialize("g_vvvv.jlbin"), a_1, a_2, a_3, a_4)
    INTpp_vvoo = ITensor(deserialize("INTpp_vvoo.jlbin"), a_3, a_4, i_1, i_2)
    g_vvvv = replaceinds(g_vvvv, inds(g_vvvv) => (a_1, a_2, a_3, a_4))
    INTpp_vvoo = replaceinds(INTpp_vvoo, inds(INTpp_vvoo) => (a_3, a_4, i_1, i_2))
    I_vvoo += 1 / 2 * g_vvvv * INTpp_vvoo
    INTpp_vvoo = nothing
    g_vvvv = nothing
    f_oo = ITensor(deserialize("f_oo.jlbin"), i_3, i_2)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_2, i_1, i_3)
    f_oo = replaceinds(f_oo, inds(f_oo) => (i_3, i_2))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_2, i_1, i_3))
    I_vvoo += -1 * f_oo * T2_vvoo
    T2_vvoo = nothing
    f_oo = nothing
    I_oo = ITensor(zeros(Float64, no, no), i_4, i_2)
    g_oovo = ITensor(deserialize("g_oovo.jlbin"), i_3, i_4, a_3, i_2)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_3, i_3)
    g_oovo = replaceinds(g_oovo, inds(g_oovo) => (i_3, i_4, a_3, i_2))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_3, i_3))
    I_oo += g_oovo * T1_vo
    T1_vo = nothing
    g_oovo = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_2, i_1, i_4)
    I_oo = replaceinds(I_oo, inds(I_oo) => (i_4, i_2))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_2, i_1, i_4))
    I_vvoo += -2 * I_oo * T2_vvoo
    T2_vvoo = nothing
    I_oo = nothing
    I_vovo = ITensor(zeros(Float64, nv, no, nv, no), a_1, i_4, a_4, i_1)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_3, i_1, i_3)
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_3, i_4, a_3, a_4))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_3, i_1, i_3))
    I_vovo += g_oovv * T2_vvoo
    T2_vvoo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_2, a_4, i_2, i_4)
    I_vovo = replaceinds(I_vovo, inds(I_vovo) => (a_1, i_4, a_4, i_1))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_2, a_4, i_2, i_4))
    I_vvoo += 2 * I_vovo * T2_vvoo
    T2_vvoo = nothing
    I_vovo = nothing
    I_oooo = ITensor(zeros(Float64, no, no, no, no), i_3, i_4, i_1, i_2)
    g_oovo = ITensor(deserialize("g_oovo.jlbin"), i_3, i_4, a_3, i_1)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_3, i_2)
    g_oovo = replaceinds(g_oovo, inds(g_oovo) => (i_3, i_4, a_3, i_1))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_3, i_2))
    I_oooo += g_oovo * T1_vo
    T1_vo = nothing
    g_oovo = nothing
    INTpp_vvoo = ITensor(deserialize("INTpp_vvoo.jlbin"), a_1, a_2, i_4, i_3)
    I_oooo = replaceinds(I_oooo, inds(I_oooo) => (i_3, i_4, i_1, i_2))
    INTpp_vvoo = replaceinds(INTpp_vvoo, inds(INTpp_vvoo) => (a_1, a_2, i_4, i_3))
    I_vvoo += I_oooo * INTpp_vvoo
    INTpp_vvoo = nothing
    I_oooo = nothing
    I_oo = ITensor(zeros(Float64, no, no), i_4, i_2)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    INTpp_vvoo = ITensor(deserialize("INTpp_vvoo.jlbin"), a_3, a_4, i_3, i_2)
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_3, i_4, a_3, a_4))
    INTpp_vvoo = replaceinds(INTpp_vvoo, inds(INTpp_vvoo) => (a_3, a_4, i_3, i_2))
    I_oo += g_oovv * INTpp_vvoo
    INTpp_vvoo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_2, i_1, i_4)
    I_oo = replaceinds(I_oo, inds(I_oo) => (i_4, i_2))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_2, i_1, i_4))
    I_vvoo += -2 * I_oo * T2_vvoo
    T2_vvoo = nothing
    I_oo = nothing
    I_vovo = ITensor(zeros(Float64, nv, no, nv, no), a_1, i_3, a_4, i_1)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_3, i_1, i_4)
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_3, i_4, a_3, a_4))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_3, i_1, i_4))
    I_vovo += g_oovv * T2_vvoo
    T2_vvoo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_2, a_4, i_3, i_2)
    I_vovo = replaceinds(I_vovo, inds(I_vovo) => (a_1, i_3, a_4, i_1))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_2, a_4, i_3, i_2))
    I_vvoo += I_vovo * T2_vvoo
    T2_vvoo = nothing
    I_vovo = nothing
    I_vovo = ITensor(zeros(Float64, nv, no, nv, no), a_2, i_3, a_4, i_2)
    g_vovv = ITensor(deserialize("g_vovv.jlbin"), a_2, i_3, a_3, a_4)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_3, i_2)
    g_vovv = replaceinds(g_vovv, inds(g_vovv) => (a_2, i_3, a_3, a_4))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_3, i_2))
    I_vovo += g_vovv * T1_vo
    T1_vo = nothing
    g_vovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_4, i_1, i_3)
    I_vovo = replaceinds(I_vovo, inds(I_vovo) => (a_2, i_3, a_4, i_2))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_4, i_1, i_3))
    I_vvoo += 2 * I_vovo * T2_vvoo
    T2_vvoo = nothing
    I_vovo = nothing
    I_vovo = ITensor(zeros(Float64, nv, no, nv, no), a_2, i_3, a_4, i_2)
    g_vovv = ITensor(deserialize("g_vovv.jlbin"), a_2, i_3, a_3, a_4)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_3, i_2)
    g_vovv = replaceinds(g_vovv, inds(g_vovv) => (a_2, i_3, a_3, a_4))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_3, i_2))
    I_vovo += g_vovv * T1_vo
    T1_vo = nothing
    g_vovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_4, i_3, i_1)
    I_vovo = replaceinds(I_vovo, inds(I_vovo) => (a_2, i_3, a_4, i_2))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_4, i_3, i_1))
    I_vvoo += -1 * I_vovo * T2_vvoo
    T2_vvoo = nothing
    I_vovo = nothing
    I_vooo = ITensor(zeros(Float64, nv, no, no, no), a_2, i_3, i_1, i_2)
    g_voov = ITensor(deserialize("g_voov.jlbin"), a_2, i_3, i_2, a_3)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_3, i_1)
    g_voov = replaceinds(g_voov, inds(g_voov) => (a_2, i_3, i_2, a_3))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_3, i_1))
    I_vooo += g_voov * T1_vo
    T1_vo = nothing
    g_voov = nothing
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_1, i_3)
    I_vooo = replaceinds(I_vooo, inds(I_vooo) => (a_2, i_3, i_1, i_2))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_1, i_3))
    I_vvoo += -1 * I_vooo * T1_vo
    T1_vo = nothing
    I_vooo = nothing
    I_vv = ITensor(zeros(Float64, nv, nv), a_2, a_4)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    INTpp_vvoo = ITensor(deserialize("INTpp_vvoo.jlbin"), a_2, a_3, i_3, i_4)
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_3, i_4, a_3, a_4))
    INTpp_vvoo = replaceinds(INTpp_vvoo, inds(INTpp_vvoo) => (a_2, a_3, i_3, i_4))
    I_vv += g_oovv * INTpp_vvoo
    INTpp_vvoo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_4, i_1, i_2)
    I_vv = replaceinds(I_vv, inds(I_vv) => (a_2, a_4))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_4, i_1, i_2))
    I_vvoo += I_vv * T2_vvoo
    T2_vvoo = nothing
    I_vv = nothing
    I_oo = ITensor(zeros(Float64, no, no), i_3, i_2)
    g_oovo = ITensor(deserialize("g_oovo.jlbin"), i_3, i_4, a_3, i_2)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_3, i_4)
    g_oovo = replaceinds(g_oovo, inds(g_oovo) => (i_3, i_4, a_3, i_2))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_3, i_4))
    I_oo += g_oovo * T1_vo
    T1_vo = nothing
    g_oovo = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_2, i_1, i_3)
    I_oo = replaceinds(I_oo, inds(I_oo) => (i_3, i_2))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_2, i_1, i_3))
    I_vvoo += I_oo * T2_vvoo
    T2_vvoo = nothing
    I_oo = nothing
    I_vooo = ITensor(zeros(Float64, nv, no, no, no), a_1, i_3, i_1, i_2)
    g_oovo = ITensor(deserialize("g_oovo.jlbin"), i_3, i_4, a_3, i_1)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_3, i_4, i_2)
    g_oovo = replaceinds(g_oovo, inds(g_oovo) => (i_3, i_4, a_3, i_1))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_3, i_4, i_2))
    I_vooo += g_oovo * T2_vvoo
    T2_vvoo = nothing
    g_oovo = nothing
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_2, i_3)
    I_vooo = replaceinds(I_vooo, inds(I_vooo) => (a_1, i_3, i_1, i_2))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_2, i_3))
    I_vvoo += I_vooo * T1_vo
    T1_vo = nothing
    I_vooo = nothing
    I_vooo = ITensor(zeros(Float64, nv, no, no, no), a_1, i_4, i_1, i_2)
    g_oovo = ITensor(deserialize("g_oovo.jlbin"), i_3, i_4, a_3, i_2)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_3, i_3, i_1)
    g_oovo = replaceinds(g_oovo, inds(g_oovo) => (i_3, i_4, a_3, i_2))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_3, i_3, i_1))
    I_vooo += g_oovo * T2_vvoo
    T2_vvoo = nothing
    g_oovo = nothing
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_2, i_4)
    I_vooo = replaceinds(I_vooo, inds(I_vooo) => (a_1, i_4, i_1, i_2))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_2, i_4))
    I_vvoo += I_vooo * T1_vo
    T1_vo = nothing
    I_vooo = nothing
    I_vovo = ITensor(zeros(Float64, nv, no, nv, no), a_1, i_4, a_4, i_1)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_3, i_3, i_1)
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_3, i_4, a_3, a_4))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_3, i_3, i_1))
    I_vovo += g_oovv * T2_vvoo
    T2_vvoo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_2, a_4, i_4, i_2)
    I_vovo = replaceinds(I_vovo, inds(I_vovo) => (a_1, i_4, a_4, i_1))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_2, a_4, i_4, i_2))
    I_vvoo += 1 / 2 * I_vovo * T2_vvoo
    T2_vvoo = nothing
    I_vovo = nothing
    I_vooo = ITensor(zeros(Float64, nv, no, no, no), a_1, i_3, i_1, i_2)
    g_vovo = ITensor(deserialize("g_vovo.jlbin"), a_1, i_3, a_3, i_2)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_3, i_1)
    g_vovo = replaceinds(g_vovo, inds(g_vovo) => (a_1, i_3, a_3, i_2))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_3, i_1))
    I_vooo += g_vovo * T1_vo
    T1_vo = nothing
    g_vovo = nothing
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_2, i_3)
    I_vooo = replaceinds(I_vooo, inds(I_vooo) => (a_1, i_3, i_1, i_2))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_2, i_3))
    I_vvoo += -1 * I_vooo * T1_vo
    T1_vo = nothing
    I_vooo = nothing
    I_vovo = ITensor(zeros(Float64, nv, no, nv, no), a_1, i_3, a_3, i_1)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_4, i_1, i_4)
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_3, i_4, a_3, a_4))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_4, i_1, i_4))
    I_vovo += g_oovv * T2_vvoo
    T2_vvoo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_2, a_3, i_3, i_2)
    I_vovo = replaceinds(I_vovo, inds(I_vovo) => (a_1, i_3, a_3, i_1))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_2, a_3, i_3, i_2))
    I_vvoo += -2 * I_vovo * T2_vvoo
    T2_vvoo = nothing
    I_vovo = nothing
    I_vovo = ITensor(zeros(Float64, nv, no, nv, no), a_2, i_3, a_3, i_2)
    g_vovv = ITensor(deserialize("g_vovv.jlbin"), a_2, i_3, a_3, a_4)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_4, i_2)
    g_vovv = replaceinds(g_vovv, inds(g_vovv) => (a_2, i_3, a_3, a_4))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_4, i_2))
    I_vovo += g_vovv * T1_vo
    T1_vo = nothing
    g_vovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_3, i_1, i_3)
    I_vovo = replaceinds(I_vovo, inds(I_vovo) => (a_2, i_3, a_3, i_2))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_3, i_1, i_3))
    I_vvoo += -1 * I_vovo * T2_vvoo
    T2_vvoo = nothing
    I_vovo = nothing
    I_vovo = ITensor(zeros(Float64, nv, no, nv, no), a_1, i_3, a_3, i_2)
    g_vovv = ITensor(deserialize("g_vovv.jlbin"), a_1, i_3, a_3, a_4)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_4, i_2)
    g_vovv = replaceinds(g_vovv, inds(g_vovv) => (a_1, i_3, a_3, a_4))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_4, i_2))
    I_vovo += g_vovv * T1_vo
    T1_vo = nothing
    g_vovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_2, a_3, i_3, i_1)
    I_vovo = replaceinds(I_vovo, inds(I_vovo) => (a_1, i_3, a_3, i_2))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_2, a_3, i_3, i_1))
    I_vvoo += -1 * I_vovo * T2_vvoo
    T2_vvoo = nothing
    I_vovo = nothing
    I_vooo = ITensor(zeros(Float64, nv, no, no, no), a_1, i_3, i_1, i_2)
    g_oovo = ITensor(deserialize("g_oovo.jlbin"), i_3, i_4, a_3, i_2)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_3, i_1, i_4)
    g_oovo = replaceinds(g_oovo, inds(g_oovo) => (i_3, i_4, a_3, i_2))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_3, i_1, i_4))
    I_vooo += g_oovo * T2_vvoo
    T2_vvoo = nothing
    g_oovo = nothing
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_2, i_3)
    I_vooo = replaceinds(I_vooo, inds(I_vooo) => (a_1, i_3, i_1, i_2))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_2, i_3))
    I_vvoo += I_vooo * T1_vo
    T1_vo = nothing
    I_vooo = nothing
    I_vv = ITensor(zeros(Float64, nv, nv), a_2, a_4)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    INTpp_vvoo = ITensor(deserialize("INTpp_vvoo.jlbin"), a_2, a_3, i_4, i_3)
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_3, i_4, a_3, a_4))
    INTpp_vvoo = replaceinds(INTpp_vvoo, inds(INTpp_vvoo) => (a_2, a_3, i_4, i_3))
    I_vv += g_oovv * INTpp_vvoo
    INTpp_vvoo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_4, i_1, i_2)
    I_vv = replaceinds(I_vv, inds(I_vv) => (a_2, a_4))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_4, i_1, i_2))
    I_vvoo += -2 * I_vv * T2_vvoo
    T2_vvoo = nothing
    I_vv = nothing
    I_vovo = ITensor(zeros(Float64, nv, no, nv, no), a_1, i_4, a_3, i_1)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_4, i_1, i_3)
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_3, i_4, a_3, a_4))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_4, i_1, i_3))
    I_vovo += g_oovv * T2_vvoo
    T2_vvoo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_2, a_3, i_2, i_4)
    I_vovo = replaceinds(I_vovo, inds(I_vovo) => (a_1, i_4, a_3, i_1))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_2, a_3, i_2, i_4))
    I_vvoo += -1 * I_vovo * T2_vvoo
    T2_vvoo = nothing
    I_vovo = nothing
    I_vooo = ITensor(zeros(Float64, nv, no, no, no), a_1, i_4, i_1, i_2)
    g_oovo = ITensor(deserialize("g_oovo.jlbin"), i_3, i_4, a_3, i_2)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_3, i_1, i_3)
    g_oovo = replaceinds(g_oovo, inds(g_oovo) => (i_3, i_4, a_3, i_2))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_3, i_1, i_3))
    I_vooo += g_oovo * T2_vvoo
    T2_vvoo = nothing
    g_oovo = nothing
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_2, i_4)
    I_vooo = replaceinds(I_vooo, inds(I_vooo) => (a_1, i_4, i_1, i_2))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_2, i_4))
    I_vvoo += -2 * I_vooo * T1_vo
    T1_vo = nothing
    I_vooo = nothing
    I_vooo = ITensor(zeros(Float64, nv, no, no, no), a_1, i_3, i_1, i_2)
    g_vovv = ITensor(deserialize("g_vovv.jlbin"), a_1, i_3, a_3, a_4)
    INTpp_vvoo = ITensor(deserialize("INTpp_vvoo.jlbin"), a_3, a_4, i_1, i_2)
    g_vovv = replaceinds(g_vovv, inds(g_vovv) => (a_1, i_3, a_3, a_4))
    INTpp_vvoo = replaceinds(INTpp_vvoo, inds(INTpp_vvoo) => (a_3, a_4, i_1, i_2))
    I_vooo += g_vovv * INTpp_vvoo
    INTpp_vvoo = nothing
    g_vovv = nothing
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_2, i_3)
    I_vooo = replaceinds(I_vooo, inds(I_vooo) => (a_1, i_3, i_1, i_2))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_2, i_3))
    I_vvoo += -1 * I_vooo * T1_vo
    T1_vo = nothing
    I_vooo = nothing
    I_oooo = ITensor(zeros(Float64, no, no, no, no), i_3, i_4, i_1, i_2)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    INTpp_vvoo = ITensor(deserialize("INTpp_vvoo.jlbin"), a_3, a_4, i_1, i_2)
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_3, i_4, a_3, a_4))
    INTpp_vvoo = replaceinds(INTpp_vvoo, inds(INTpp_vvoo) => (a_3, a_4, i_1, i_2))
    I_oooo += g_oovv * INTpp_vvoo
    INTpp_vvoo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_2, i_3, i_4)
    I_oooo = replaceinds(I_oooo, inds(I_oooo) => (i_3, i_4, i_1, i_2))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_2, i_3, i_4))
    I_vvoo += 1 / 2 * I_oooo * T2_vvoo
    T2_vvoo = nothing
    I_oooo = nothing
    I_vooo = ITensor(zeros(Float64, nv, no, no, no), a_1, i_4, i_1, i_2)
    I_oooo = ITensor(zeros(Float64, no, no, no, no), i_3, i_4, i_1, i_2)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    INTpp_vvoo = ITensor(deserialize("INTpp_vvoo.jlbin"), a_3, a_4, i_1, i_2)
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_3, i_4, a_3, a_4))
    INTpp_vvoo = replaceinds(INTpp_vvoo, inds(INTpp_vvoo) => (a_3, a_4, i_1, i_2))
    I_oooo += g_oovv * INTpp_vvoo
    INTpp_vvoo = nothing
    g_oovv = nothing
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_1, i_3)
    I_oooo = replaceinds(I_oooo, inds(I_oooo) => (i_3, i_4, i_1, i_2))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_1, i_3))
    I_vooo += I_oooo * T1_vo
    T1_vo = nothing
    I_oooo = nothing
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_2, i_4)
    I_vooo = replaceinds(I_vooo, inds(I_vooo) => (a_1, i_4, i_1, i_2))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_2, i_4))
    I_vvoo += 1 / 2 * I_vooo * T1_vo
    T1_vo = nothing
    I_vooo = nothing
    I_oo = ITensor(zeros(Float64, no, no), i_4, i_2)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    INTpp_vvoo = ITensor(deserialize("INTpp_vvoo.jlbin"), a_3, a_4, i_2, i_3)
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_3, i_4, a_3, a_4))
    INTpp_vvoo = replaceinds(INTpp_vvoo, inds(INTpp_vvoo) => (a_3, a_4, i_2, i_3))
    I_oo += g_oovv * INTpp_vvoo
    INTpp_vvoo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_2, i_1, i_4)
    I_oo = replaceinds(I_oo, inds(I_oo) => (i_4, i_2))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_2, i_1, i_4))
    I_vvoo += I_oo * T2_vvoo
    T2_vvoo = nothing
    I_oo = nothing
    I_vv = ITensor(zeros(Float64, nv, nv), a_2, a_4)
    g_vovv = ITensor(deserialize("g_vovv.jlbin"), a_2, i_3, a_3, a_4)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_3, i_3)
    g_vovv = replaceinds(g_vovv, inds(g_vovv) => (a_2, i_3, a_3, a_4))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_3, i_3))
    I_vv += g_vovv * T1_vo
    T1_vo = nothing
    g_vovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_4, i_1, i_2)
    I_vv = replaceinds(I_vv, inds(I_vv) => (a_2, a_4))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_4, i_1, i_2))
    I_vvoo += -1 * I_vv * T2_vvoo
    T2_vvoo = nothing
    I_vv = nothing
    I_vv = ITensor(zeros(Float64, nv, nv), a_2, a_3)
    g_vovv = ITensor(deserialize("g_vovv.jlbin"), a_2, i_3, a_3, a_4)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_4, i_3)
    g_vovv = replaceinds(g_vovv, inds(g_vovv) => (a_2, i_3, a_3, a_4))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_4, i_3))
    I_vv += g_vovv * T1_vo
    T1_vo = nothing
    g_vovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_3, i_1, i_2)
    I_vv = replaceinds(I_vv, inds(I_vv) => (a_2, a_3))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_3, i_1, i_2))
    I_vvoo += 2 * I_vv * T2_vvoo
    T2_vvoo = nothing
    I_vv = nothing
    I_oo = ITensor(zeros(Float64, no, no), i_3, i_2)
    f_ov = ITensor(deserialize("f_ov.jlbin"), i_3, a_3)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_3, i_2)
    f_ov = replaceinds(f_ov, inds(f_ov) => (i_3, a_3))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_3, i_2))
    I_oo += f_ov * T1_vo
    T1_vo = nothing
    f_ov = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_2, i_1, i_3)
    I_oo = replaceinds(I_oo, inds(I_oo) => (i_3, i_2))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_2, i_1, i_3))
    I_vvoo += -1 * I_oo * T2_vvoo
    T2_vvoo = nothing
    I_oo = nothing
    I_vovo = ITensor(zeros(Float64, nv, no, nv, no), a_1, i_4, a_3, i_2)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_4, i_3, i_2)
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_3, i_4, a_3, a_4))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_4, i_3, i_2))
    I_vovo += g_oovv * T2_vvoo
    T2_vvoo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_2, a_3, i_4, i_1)
    I_vovo = replaceinds(I_vovo, inds(I_vovo) => (a_1, i_4, a_3, i_2))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_2, a_3, i_4, i_1))
    I_vvoo += 1 / 2 * I_vovo * T2_vvoo
    T2_vvoo = nothing
    I_vovo = nothing
    I_vooo = ITensor(zeros(Float64, nv, no, no, no), a_1, i_3, i_1, i_2)
    f_ov = ITensor(deserialize("f_ov.jlbin"), i_3, a_3)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_3, i_1, i_2)
    f_ov = replaceinds(f_ov, inds(f_ov) => (i_3, a_3))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_3, i_1, i_2))
    I_vooo += f_ov * T2_vvoo
    T2_vvoo = nothing
    f_ov = nothing
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_2, i_3)
    I_vooo = replaceinds(I_vooo, inds(I_vooo) => (a_1, i_3, i_1, i_2))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_2, i_3))
    I_vvoo += -1 * I_vooo * T1_vo
    T1_vo = nothing
    I_vooo = nothing
    I_vooo = ITensor(zeros(Float64, nv, no, no, no), a_2, i_4, i_1, i_2)
    I_oovo = ITensor(zeros(Float64, no, no, nv, no), i_3, i_4, a_4, i_1)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_3, i_1)
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_3, i_4, a_3, a_4))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_3, i_1))
    I_oovo += g_oovv * T1_vo
    T1_vo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_2, a_4, i_2, i_3)
    I_oovo = replaceinds(I_oovo, inds(I_oovo) => (i_3, i_4, a_4, i_1))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_2, a_4, i_2, i_3))
    I_vooo += I_oovo * T2_vvoo
    T2_vvoo = nothing
    I_oovo = nothing
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_1, i_4)
    I_vooo = replaceinds(I_vooo, inds(I_vooo) => (a_2, i_4, i_1, i_2))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_1, i_4))
    I_vvoo += I_vooo * T1_vo
    T1_vo = nothing
    I_vooo = nothing
    I_vooo = ITensor(zeros(Float64, nv, no, no, no), a_2, i_3, i_1, i_2)
    I_oovo = ITensor(zeros(Float64, no, no, nv, no), i_3, i_4, a_4, i_1)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_3, i_1)
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_3, i_4, a_3, a_4))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_3, i_1))
    I_oovo += g_oovv * T1_vo
    T1_vo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_2, a_4, i_2, i_4)
    I_oovo = replaceinds(I_oovo, inds(I_oovo) => (i_3, i_4, a_4, i_1))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_2, a_4, i_2, i_4))
    I_vooo += I_oovo * T2_vvoo
    T2_vvoo = nothing
    I_oovo = nothing
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_1, i_3)
    I_vooo = replaceinds(I_vooo, inds(I_vooo) => (a_2, i_3, i_1, i_2))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_1, i_3))
    I_vvoo += -2 * I_vooo * T1_vo
    T1_vo = nothing
    I_vooo = nothing
    I_vooo = ITensor(zeros(Float64, nv, no, no, no), a_1, i_4, i_1, i_2)
    I_oovo = ITensor(zeros(Float64, no, no, nv, no), i_3, i_4, a_4, i_1)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_3, i_1)
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_3, i_4, a_3, a_4))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_3, i_1))
    I_oovo += g_oovv * T1_vo
    T1_vo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_4, i_3, i_2)
    I_oovo = replaceinds(I_oovo, inds(I_oovo) => (i_3, i_4, a_4, i_1))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_4, i_3, i_2))
    I_vooo += I_oovo * T2_vvoo
    T2_vvoo = nothing
    I_oovo = nothing
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_2, i_4)
    I_vooo = replaceinds(I_vooo, inds(I_vooo) => (a_1, i_4, i_1, i_2))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_2, i_4))
    I_vvoo += I_vooo * T1_vo
    T1_vo = nothing
    I_vooo = nothing
    I_vooo = ITensor(zeros(Float64, nv, no, no, no), a_2, i_3, i_1, i_2)
    I_oovo = ITensor(zeros(Float64, no, no, nv, no), i_3, i_4, a_4, i_1)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_3, i_1)
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_3, i_4, a_3, a_4))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_3, i_1))
    I_oovo += g_oovv * T1_vo
    T1_vo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_2, a_4, i_4, i_2)
    I_oovo = replaceinds(I_oovo, inds(I_oovo) => (i_3, i_4, a_4, i_1))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_2, a_4, i_4, i_2))
    I_vooo += I_oovo * T2_vvoo
    T2_vvoo = nothing
    I_oovo = nothing
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_1, i_3)
    I_vooo = replaceinds(I_vooo, inds(I_vooo) => (a_2, i_3, i_1, i_2))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_1, i_3))
    I_vvoo += I_vooo * T1_vo
    T1_vo = nothing
    I_vooo = nothing
    return I_vvoo + swapinds(I_vvoo, (a_1, i_1), (a_2, i_2))
end


function calc_R1()
    nv = deserialize("nv.jlbin")
    no = deserialize("no.jlbin")
    a_1 = Index(nv, "a_1")
    a_2 = Index(nv, "a_2")
    a_3 = Index(nv, "a_3")
    i_1 = Index(no, "i_1")
    i_2 = Index(no, "i_2")
    i_3 = Index(no, "i_3")


    I_vo = ITensor(zeros(Float64, nv, no), a_1, i_1)
    f_ov = ITensor(deserialize("f_ov.jlbin"), i_2, a_2)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_2, i_2, i_1)
    f_ov = replaceinds(f_ov, inds(f_ov) => (i_2, a_2))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_2, i_2, i_1))
    I_vo += -1 * f_ov * T2_vvoo
    T2_vvoo = nothing
    f_ov = nothing
    f_vo = ITensor(deserialize("f_vo.jlbin"), a_1, i_1)
    f_vo = replaceinds(f_vo, inds(f_vo) => (a_1, i_1))
    I_vo += f_vo
    f_vo = nothing
    g_voov = ITensor(deserialize("g_voov.jlbin"), a_1, i_2, i_1, a_2)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_2, i_2)
    g_voov = replaceinds(g_voov, inds(g_voov) => (a_1, i_2, i_1, a_2))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_2, i_2))
    I_vo += 2 * g_voov * T1_vo
    T1_vo = nothing
    g_voov = nothing
    g_vovo = ITensor(deserialize("g_vovo.jlbin"), a_1, i_2, a_2, i_1)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_2, i_2)
    g_vovo = replaceinds(g_vovo, inds(g_vovo) => (a_1, i_2, a_2, i_1))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_2, i_2))
    I_vo += -1 * g_vovo * T1_vo
    T1_vo = nothing
    g_vovo = nothing
    g_oovo = ITensor(deserialize("g_oovo.jlbin"), i_2, i_3, a_2, i_1)
    INTpp_vvoo = ITensor(deserialize("INTpp_vvoo.jlbin"), a_1, a_2, i_2, i_3)
    g_oovo = replaceinds(g_oovo, inds(g_oovo) => (i_2, i_3, a_2, i_1))
    INTpp_vvoo = replaceinds(INTpp_vvoo, inds(INTpp_vvoo) => (a_1, a_2, i_2, i_3))
    I_vo += g_oovo * INTpp_vvoo
    INTpp_vvoo = nothing
    g_oovo = nothing
    g_vovv = ITensor(deserialize("g_vovv.jlbin"), a_1, i_2, a_2, a_3)
    INTpp_vvoo = ITensor(deserialize("INTpp_vvoo.jlbin"), a_3, a_2, i_1, i_2)
    g_vovv = replaceinds(g_vovv, inds(g_vovv) => (a_1, i_2, a_2, a_3))
    INTpp_vvoo = replaceinds(INTpp_vvoo, inds(INTpp_vvoo) => (a_3, a_2, i_1, i_2))
    I_vo += -1 * g_vovv * INTpp_vvoo
    INTpp_vvoo = nothing
    g_vovv = nothing
    g_oovo = ITensor(deserialize("g_oovo.jlbin"), i_2, i_3, a_2, i_1)
    INTpp_vvoo = ITensor(deserialize("INTpp_vvoo.jlbin"), a_1, a_2, i_3, i_2)
    g_oovo = replaceinds(g_oovo, inds(g_oovo) => (i_2, i_3, a_2, i_1))
    INTpp_vvoo = replaceinds(INTpp_vvoo, inds(INTpp_vvoo) => (a_1, a_2, i_3, i_2))
    I_vo += -2 * g_oovo * INTpp_vvoo
    INTpp_vvoo = nothing
    g_oovo = nothing
    f_vv = ITensor(deserialize("f_vv.jlbin"), a_1, a_2)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_2, i_1)
    f_vv = replaceinds(f_vv, inds(f_vv) => (a_1, a_2))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_2, i_1))
    I_vo += f_vv * T1_vo
    T1_vo = nothing
    f_vv = nothing
    f_oo = ITensor(deserialize("f_oo.jlbin"), i_2, i_1)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_1, i_2)
    f_oo = replaceinds(f_oo, inds(f_oo) => (i_2, i_1))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_1, i_2))
    I_vo += -1 * f_oo * T1_vo
    T1_vo = nothing
    f_oo = nothing
    g_vovv = ITensor(deserialize("g_vovv.jlbin"), a_1, i_2, a_2, a_3)
    INTpp_vvoo = ITensor(deserialize("INTpp_vvoo.jlbin"), a_2, a_3, i_1, i_2)
    g_vovv = replaceinds(g_vovv, inds(g_vovv) => (a_1, i_2, a_2, a_3))
    INTpp_vvoo = replaceinds(INTpp_vvoo, inds(INTpp_vvoo) => (a_2, a_3, i_1, i_2))
    I_vo += 2 * g_vovv * INTpp_vvoo
    INTpp_vvoo = nothing
    g_vovv = nothing
    f_ov = ITensor(deserialize("f_ov.jlbin"), i_2, a_2)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_2, i_1, i_2)
    f_ov = replaceinds(f_ov, inds(f_ov) => (i_2, a_2))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_2, i_1, i_2))
    I_vo += 2 * f_ov * T2_vvoo
    T2_vvoo = nothing
    f_ov = nothing
    I_ov = ITensor(zeros(Float64, no, nv), i_2, a_2)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_2, i_3, a_2, a_3)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_3, i_3)
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_2, i_3, a_2, a_3))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_3, i_3))
    I_ov += g_oovv * T1_vo
    T1_vo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_2, i_2, i_1)
    I_ov = replaceinds(I_ov, inds(I_ov) => (i_2, a_2))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_2, i_2, i_1))
    I_vo += -2 * I_ov * T2_vvoo
    T2_vvoo = nothing
    I_ov = nothing
    I_ov = ITensor(zeros(Float64, no, nv), i_2, a_2)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_2, i_3, a_2, a_3)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_3, i_3)
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_2, i_3, a_2, a_3))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_3, i_3))
    I_ov += g_oovv * T1_vo
    T1_vo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_2, i_1, i_2)
    I_ov = replaceinds(I_ov, inds(I_ov) => (i_2, a_2))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_2, i_1, i_2))
    I_vo += 4 * I_ov * T2_vvoo
    T2_vvoo = nothing
    I_ov = nothing
    I_oo = ITensor(zeros(Float64, no, no), i_2, i_1)
    f_ov = ITensor(deserialize("f_ov.jlbin"), i_2, a_2)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_2, i_1)
    f_ov = replaceinds(f_ov, inds(f_ov) => (i_2, a_2))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_2, i_1))
    I_oo += f_ov * T1_vo
    T1_vo = nothing
    f_ov = nothing
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_1, i_2)
    I_oo = replaceinds(I_oo, inds(I_oo) => (i_2, i_1))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_1, i_2))
    I_vo += -1 * I_oo * T1_vo
    T1_vo = nothing
    I_oo = nothing
    I_oovo = ITensor(zeros(Float64, no, no, nv, no), i_2, i_3, a_2, i_1)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_2, i_3, a_2, a_3)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_3, i_1)
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_2, i_3, a_2, a_3))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_3, i_1))
    I_oovo += g_oovv * T1_vo
    T1_vo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_2, i_2, i_3)
    I_oovo = replaceinds(I_oovo, inds(I_oovo) => (i_2, i_3, a_2, i_1))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_2, i_2, i_3))
    I_vo += I_oovo * T2_vvoo
    T2_vvoo = nothing
    I_oovo = nothing
    I_oovo = ITensor(zeros(Float64, no, no, nv, no), i_2, i_3, a_2, i_1)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_2, i_3, a_2, a_3)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_3, i_1)
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_2, i_3, a_2, a_3))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_3, i_1))
    I_oovo += g_oovv * T1_vo
    T1_vo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_2, i_3, i_2)
    I_oovo = replaceinds(I_oovo, inds(I_oovo) => (i_2, i_3, a_2, i_1))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_2, i_3, i_2))
    I_vo += -2 * I_oovo * T2_vvoo
    T2_vvoo = nothing
    I_oovo = nothing
    I_oo = ITensor(zeros(Float64, no, no), i_3, i_1)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_2, i_3, a_2, a_3)
    INTpp_vvoo = ITensor(deserialize("INTpp_vvoo.jlbin"), a_2, a_3, i_1, i_2)
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_2, i_3, a_2, a_3))
    INTpp_vvoo = replaceinds(INTpp_vvoo, inds(INTpp_vvoo) => (a_2, a_3, i_1, i_2))
    I_oo += g_oovv * INTpp_vvoo
    INTpp_vvoo = nothing
    g_oovv = nothing
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_1, i_3)
    I_oo = replaceinds(I_oo, inds(I_oo) => (i_3, i_1))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_1, i_3))
    I_vo += I_oo * T1_vo
    T1_vo = nothing
    I_oo = nothing
    I_oo = ITensor(zeros(Float64, no, no), i_3, i_1)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_2, i_3, a_2, a_3)
    INTpp_vvoo = ITensor(deserialize("INTpp_vvoo.jlbin"), a_3, a_2, i_1, i_2)
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_2, i_3, a_2, a_3))
    INTpp_vvoo = replaceinds(INTpp_vvoo, inds(INTpp_vvoo) => (a_3, a_2, i_1, i_2))
    I_oo += g_oovv * INTpp_vvoo
    INTpp_vvoo = nothing
    g_oovv = nothing
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_1, i_3)
    I_oo = replaceinds(I_oo, inds(I_oo) => (i_3, i_1))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_1, i_3))
    I_vo += -2 * I_oo * T1_vo
    T1_vo = nothing
    I_oo = nothing
    I_ov = ITensor(zeros(Float64, no, nv), i_2, a_3)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_2, i_3, a_2, a_3)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_2, i_3)
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_2, i_3, a_2, a_3))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_2, i_3))
    I_ov += g_oovv * T1_vo
    T1_vo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_3, i_2, i_1)
    I_ov = replaceinds(I_ov, inds(I_ov) => (i_2, a_3))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_3, i_2, i_1))
    I_vo += I_ov * T2_vvoo
    T2_vvoo = nothing
    I_ov = nothing
    I_ov = ITensor(zeros(Float64, no, nv), i_2, a_3)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_2, i_3, a_2, a_3)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_2, i_3)
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_2, i_3, a_2, a_3))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_2, i_3))
    I_ov += g_oovv * T1_vo
    T1_vo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_3, i_1, i_2)
    I_ov = replaceinds(I_ov, inds(I_ov) => (i_2, a_3))
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_3, i_1, i_2))
    I_vo += -2 * I_ov * T2_vvoo
    T2_vvoo = nothing
    I_ov = nothing
    return I_vo

end



function calculate_ECCSD()
    nv = deserialize("nv.jlbin")
    no = deserialize("no.jlbin")
    a_1 = Index(nv, "a_1")
    a_2 = Index(nv, "a_2")
    i_1 = Index(no, "i_1")
    i_2 = Index(no, "i_2")



    tmpvar = 0.0
    Z = ITensor(tmpvar)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_1, i_2, a_1, a_2)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_2, i_1, i_2)
    Z += 2 * g_oovv * T2_vvoo
    T2_vvoo = nothing
    g_oovv = nothing
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_1, i_2, a_1, a_2)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_2, i_2, i_1)
    Z += -1 * g_oovv * T2_vvoo
    T2_vvoo = nothing
    g_oovv = nothing
    f_ov = ITensor(deserialize("f_ov.jlbin"), i_1, a_1)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_1, i_1)
    Z += 2 * f_ov * T1_vo
    T1_vo = nothing
    f_ov = nothing
    I_ov = ITensor(zeros(Float64, no, nv), i_1, a_2)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_1, i_2, a_1, a_2)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_1, i_2)
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_1, i_2, a_1, a_2))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_1, i_2))
    I_ov += g_oovv * T1_vo
    T1_vo = nothing
    g_oovv = nothing
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_2, i_1)
    Z += -1 * I_ov * T1_vo
    T1_vo = nothing
    I_ov = nothing
    I_ov = ITensor(zeros(Float64, no, nv), i_2, a_2)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_1, i_2, a_1, a_2)
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_1, i_1)
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_1, i_2, a_1, a_2))
    T1_vo = replaceinds(T1_vo, inds(T1_vo) => (a_1, i_1))
    I_ov += g_oovv * T1_vo
    T1_vo = nothing
    g_oovv = nothing
    T1_vo = ITensor(deserialize("T1_vo.jlbin"), a_2, i_2)
    Z += 2 * I_ov * T1_vo
    T1_vo = nothing
    I_ov = nothing
    return Z

end

function calculate_INTpp_vvoo!(T2,T1,result)
    a_1, a_2, i_1, i_2 = inds(T2)
    result = 0.5 * (T2 + swapinds(T2, (a_1, i_1), (a_2, i_2)))
    result += 0.5 * T1 * replaceinds(T1, inds(T1) => (a_2, i_2))
    result += 0.5 * replaceinds(T1, inds(T1) => (a_2, i_2)) * T1
    return result
end




































