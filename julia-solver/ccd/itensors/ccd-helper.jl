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
    fvv = zeros(nv, nv)
    foo = zeros(no, no)
    for a in 1:nv, b in 1:nv
        fvv[a, b] = f[no+a, no+b]

        for i in 1:no, j in 1:no
            g_vvoo[a, b, i, j] = g[no+a, i, no+b, j]
            g_voov[a, i, j, b] = g[no+a, j, i, no+b]
            g_vovo[a, i, b, j] = g[no+a, no+b, i, j]
            g_oovv[i, j, a, b] = g[i, no+a, j, no+b]
            foo[i, j] = f[i, j]
        end
    end
    for i in 1:no, j in 1:no, k in 1:no, l in 1:no
        g_oooo[i, j, k, l] = g[i, k, j, l]
    end
    for a in 1:nv, b in 1:nv, c in 1:nv, d in 1:nv
        g_vvvv[a, b, c, d] = g[no+a, no+c, no+b, no+d]
    end
    serialize("g_vvoo.jlbin", g_vvoo)
    serialize("g_voov.jlbin", g_voov)
    serialize("g_vovo.jlbin", g_vovo)
    serialize("g_oovv.jlbin", g_oovv)
    serialize("g_oooo.jlbin", g_oooo)
    serialize("g_vvvv.jlbin", g_vvvv)
    serialize("K.jlbin", K)
    serialize("f_vv.jlbin", fvv)
    serialize("f_oo.jlbin", foo)
end






function calculate_residual(T2, R2u, R2)
    g_vvoo, g_voov, g_vovo, g_oovv, g_oooo, g_vvvv = deserialize("g_vvoo.jlbin"), deserialize("g_voov.jlbin"), deserialize("g_vovo.jlbin"), deserialize("g_oovv.jlbin"), deserialize("g_oooo.jlbin"), deserialize("g_vvvv.jlbin")
    fvv, foo = deserialize("f_vv.jlbin"), deserialize("f_oo.jlbin")
    @tensor begin
        R2u[a_1, a_2, i_1, i_2] = 0.5 * g_vvoo[a_1, a_2, i_1, i_2] - g_voov[a_1, i_3, i_1, a_3] * T2[a_2, a_3, i_3, i_2] - g_vovo[a_2, i_3, a_3, i_2] * T2[a_1, a_3, i_1, i_3] - g_vovo[a_2, i_3, a_3, i_1] * T2[a_1, a_3, i_3, i_2] + 2 * g_voov[a_1, i_3, i_1, a_3] * T2[a_2, a_3, i_2, i_3] + 0.5 * g_oooo[i_3, i_4, i_1, i_2] * T2[a_1, a_2, i_3, i_4] + fvv[a_2, a_3] * T2[a_1, a_3, i_1, i_2] + 0.5 * g_vvvv[a_1, a_2, a_3, a_4] * T2[a_3, a_4, i_1, i_2] - foo[i_3, i_2] * T2[a_1, a_2, i_1, i_3] + 2 * (g_oovv[i_3, i_4, a_3, a_4] * T2[a_1, a_3, i_1, i_3]) * T2[a_2, a_4, i_2, i_4] - 2 * (g_oovv[i_3, i_4, a_3, a_4] * T2[a_1, a_3, i_1, i_3]) * T2[a_2, a_4, i_4, i_2] + 0.5 * (g_oovv[i_3, i_4, a_3, a_4] * T2[a_1, a_3, i_3, i_1]) * T2[a_2, a_4, i_4, i_2] - 2 * (g_oovv[i_3, i_4, a_3, a_4] * T2[a_3, a_4, i_3, i_2]) * T2[a_1, a_2, i_1, i_4] - 1 * (g_oovv[i_3, i_4, a_3, a_4] * T2[a_1, a_4, i_1, i_3]) * T2[a_2, a_3, i_2, i_4] + (g_oovv[i_3, i_4, a_3, a_4] * T2[a_1, a_4, i_1, i_3]) * T2[a_2, a_3, i_4, i_2] + (g_oovv[i_3, i_4, a_3, a_4] * T2[a_2, a_4, i_4, i_3]) * T2[a_1, a_3, i_1, i_2] + (g_oovv[i_3, i_4, a_3, a_4] * T2[a_3, a_4, i_2, i_3]) * T2[a_1, a_2, i_1, i_4] - 2 * (g_oovv[i_3, i_4, a_3, a_4] * T2[a_2, a_3, i_4, i_3]) * T2[a_1, a_4, i_1, i_2] + 0.5 * (g_oovv[i_3, i_4, a_3, a_4] * T2[a_1, a_4, i_3, i_2]) * T2[a_2, a_3, i_4, i_1] + 0.5 * (g_oovv[i_3, i_4, a_3, a_4] * T2[a_3, a_4, i_1, i_2]) * T2[a_1, a_2, i_3, i_4]
        R2[a, b, i, j] = R2u[a, b, i, j] + R2u[b, a, j, i]
    end
    return R2
end

function calculate_residual_memeff_gc(T2)
    g_vvoo, g_voov, g_vovo, g_oovv, g_oooo, g_vvvv = deserialize("g_vvoo.jlbin"), deserialize("g_voov.jlbin"), deserialize("g_vovo.jlbin"), deserialize("g_oovv.jlbin"), deserialize("g_oooo.jlbin"), deserialize("g_vvvv.jlbin")
    fvv, foo = deserialize("f_vv.jlbin"), deserialize("f_oo.jlbin")
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
        Trm7[a_1, a_2, i_1, i_2] := fvv[a_2, a_3] * T2[a_1, a_3, i_1, i_2]
        R2u[a_1, a_2, i_1, i_2] += Trm7[a_1, a_2, i_1, i_2]
        @notensor Trm7 = nothing
        @notensor fvv = nothing
        GC.gc()
        Trm8[a_1, a_2, i_1, i_2] := +0.5 * g_vvvv[a_1, a_2, a_3, a_4] * T2[a_3, a_4, i_1, i_2]
        R2u[a_1, a_2, i_1, i_2] += Trm8[a_1, a_2, i_1, i_2]
        @notensor Trm8 = nothing
        @notensor g_vvvv = nothing
        GC.gc()
        Trm9[a_1, a_2, i_1, i_2] := -foo[i_3, i_2] * T2[a_1, a_2, i_1, i_3]
        R2u[a_1, a_2, i_1, i_2] += Trm9[a_1, a_2, i_1, i_2]
        @notensor Trm9 = nothing
        @notensor foo = nothing
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

function update_T2(T2::Array{Float64,4}, R2::Array{Float64,4}, Scaled_R2::Array{Float64,4})
    fvv = deserialize("f_vv.jlbin")
    foo = deserialize("f_oo.jlbin")
    nv = deserialize("nv.jlbin")
    no = deserialize("no.jlbin")
    shiftp = 0.20
    for a in 1:nv, b in 1:nv, i in 1:no, j in 1:no
        Scaled_R2[a, b, i, j] = R2[a, b, i, j] / (fvv[a, a] + fvv[b, b] - foo[i, i] - foo[j, j] + shiftp)
    end
    for a in 1:nv, b in 1:nv, i in 1:no, j in 1:no
        T2[a, b, i, j] = T2[a, b, i, j] - Scaled_R2[a, b, i, j]
    end
    return T2
end

function update_T2(T2::ITensor, R2::ITensor, Scaled_R2::ITensor)
    a_1, a_2, i_1, i_2 = inds(T2)
    fvv = ITensor(deserialize("f_vv.jlbin"), a_1, a_2)
    foo = ITensor(deserialize("f_oo.jlbin"), i_1, i_2)
    # fvv = deserialize("f_vv.jlbin")
    # foo = deserialize("f_oo.jlbin")
    nv = deserialize("nv.jlbin")::Int64
    no = deserialize("no.jlbin")::Int64
    # T2 = Array(T2,a_1,a_2,i_1,i_2)
    # Scaled_R2 = Array(Scaled_R2,a_1,a_2,i_1,i_2)
    # R2 = Array(R2,a_1,a_2,i_1,i_2)
    shiftp = 0.20
    for a in 1:nv, b in 1:nv, i in 1:no, j in 1:no
        Scaled_R2[a, b, i, j] = R2[a, b, i, j] / (fvv[a, a] + fvv[b, b] - foo[i, i] - foo[j, j] + shiftp)
    end
    for a in 1:nv, b in 1:nv, i in 1:no, j in 1:no
        T2[a, b, i, j] = T2[a, b, i, j] - Scaled_R2[a, b, i, j]
    end
    return T2
end


function check_convergence(R2, normtol, e_old, e_new, etol)
    r2norm = sqrt(dot(R2, R2))
    if (r2norm < normtol && abs(e_new[1] - e_old[1]) < etol)
        return true, r2norm
    else
        return false, r2norm
    end

end

function calculate_R_iter(R_iter, R2)
    nv = deserialize("nv.jlbin")
    no = deserialize("no.jlbin")
    fvv = deserialize("f_vv.jlbin")
    foo = deserialize("f_oo.jlbin")
    shiftp = 0.20
    for a in 1:nv, b in 1:nv, i in 1:no, j in 1:no
        R_iter[a, b, i, j] = R2[a, b, i, j] / (fvv[a, a] + fvv[b, b] - foo[i, i] - foo[j, j] + shiftp)
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

function PerformDiis(R_iter_storage, p, T2_storage, R2_storage)
    B = zeros(p + 1, p + 1)
    Bpp = dot(R2_storage[p], R2_storage[p])
    a_1, a_2, i_1, i_2 = inds(T2_storage[1])
    for i in 1:p
        R2_storage[i] = replaceinds(R2_storage[i], inds(R2_storage[i]) => (a_1, a_2, i_1, i_2))
    end
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
    a_1, a_2, i_1, i_2 = inds(T2_storage[1])
    t = ITensor(zeros(size(R_iter_storage[1])), a_1, a_2, i_1, i_2)
    for i in 1:p
        t = t + C[i] .* T2_storage[i]
    end
    # display(B)
    # display(C)
    return (t)
end

function calculate_residual_memeff(T2::Array{Float64,4})
    # g_vvoo,g_voov,g_vovo,g_oovv,g_oooo,g_vvvv= deserialize("g_vvoo.jlbin"),deserialize("g_voov.jlbin"),deserialize("g_vovo.jlbin"),deserialize("g_oovv.jlbin"),deserialize("g_oooo.jlbin"),deserialize("g_vvvv.jlbin")
    g_vvoo::Array{Float64,4} = deserialize("g_vvoo.jlbin")
    g_voov::Array{Float64,4} = deserialize("g_voov.jlbin")
    g_vovo::Array{Float64,4} = deserialize("g_vovo.jlbin")
    g_oovv::Array{Float64,4} = deserialize("g_oovv.jlbin")
    g_oooo::Array{Float64,4} = deserialize("g_oooo.jlbin")
    g_vvvv::Array{Float64,4} = deserialize("g_vvvv.jlbin")
    fvv::Array{Float64,2}, foo::Array{Float64,2} = deserialize("f_vv.jlbin"), deserialize("f_oo.jlbin")
    nv::Int64 = deserialize("nv.jlbin")
    no::Int64 = deserialize("no.jlbin")
    R2u::Array{Float64,4} = zeros(Float64, nv, nv, no, no)
    R2::Array{Float64,4} = zeros(Float64, nv, nv, no, no)
    @tensor begin
        Trm1[a_1, a_2, i_1, i_2] := 0.5 * g_vvoo[a_1, a_2, i_1, i_2]
        R2u[a_1, a_2, i_1, i_2] += Trm1[a_1, a_2, i_1, i_2]
        # @notensor Trm1 = nothing
        # @notensor g_vvoo = nothing
        Trm2[a_1, a_2, i_1, i_2] := -g_voov[a_1, i_3, i_1, a_3] * T2[a_2, a_3, i_3, i_2]
        R2u[a_1, a_2, i_1, i_2] += Trm2[a_1, a_2, i_1, i_2]
        # @notensor Trm2 = nothing
        Trm3[a_1, a_2, i_1, i_2] := -g_vovo[a_2, i_3, a_3, i_2] * T2[a_1, a_3, i_1, i_3]
        R2u[a_1, a_2, i_1, i_2] += Trm3[a_1, a_2, i_1, i_2]
        # @notensor Trm3 = nothing
        Trm4[a_1, a_2, i_1, i_2] := -g_vovo[a_2, i_3, a_3, i_1] * T2[a_1, a_3, i_3, i_2]
        R2u[a_1, a_2, i_1, i_2] += Trm4[a_1, a_2, i_1, i_2]
        # @notensor Trm4 = nothing
        # @notensor g_vovo = nothing
        Trm5[a_1, a_2, i_1, i_2] := 2 * g_voov[a_1, i_3, i_1, a_3] * T2[a_2, a_3, i_2, i_3]
        R2u[a_1, a_2, i_1, i_2] += Trm5[a_1, a_2, i_1, i_2]
        # @notensor Trm5 = nothing
        # @notensor g_voov = nothing
        Trm6[a_1, a_2, i_1, i_2] := 0.5 * g_oooo[i_3, i_4, i_1, i_2] * T2[a_1, a_2, i_3, i_4]
        R2u[a_1, a_2, i_1, i_2] += Trm6[a_1, a_2, i_1, i_2]
        # @notensor Trm6 = nothing
        # @notensor g_oooo = nothing
        Trm7[a_1, a_2, i_1, i_2] := fvv[a_2, a_3] * T2[a_1, a_3, i_1, i_2]
        R2u[a_1, a_2, i_1, i_2] += Trm7[a_1, a_2, i_1, i_2]
        # @notensor Trm7 = nothing
        # @notensor fvv = nothing
        Trm8[a_1, a_2, i_1, i_2] := +0.5 * g_vvvv[a_1, a_2, a_3, a_4] * T2[a_3, a_4, i_1, i_2]
        R2u[a_1, a_2, i_1, i_2] += Trm8[a_1, a_2, i_1, i_2]
        # @notensor Trm8 = nothing
        # @notensor g_vvvv = nothing
        Trm9[a_1, a_2, i_1, i_2] := -foo[i_3, i_2] * T2[a_1, a_2, i_1, i_3]
        R2u[a_1, a_2, i_1, i_2] += Trm9[a_1, a_2, i_1, i_2]
        # @notensor Trm9 = nothing
        # @notensor foo = nothing
        B1[i_4, a_4, a_1, i_1] := 2 * (g_oovv[i_3, i_4, a_3, a_4] * T2[a_1, a_3, i_1, i_3])
        R2u[a_1, a_2, i_1, i_2] += B1[i_4, a_4, a_1, i_1] * T2[a_2, a_4, i_2, i_4] - B1[i_4, a_4, a_1, i_1] * T2[a_2, a_4, i_4, i_2]
        # @notensor B1 = nothing
        B2[i_4, a_4, a_1, i_1] := 0.5 * (g_oovv[i_3, i_4, a_3, a_4] * T2[a_1, a_3, i_3, i_1])
        R2u[a_1, a_2, i_1, i_2] += B2[i_4, a_4, a_1, i_1] * T2[a_2, a_4, i_4, i_2]
        # @notensor B2 = nothing
        B3[i_4, i_2] := 2 * (g_oovv[i_3, i_4, a_3, a_4] * T2[a_3, a_4, i_3, i_2])
        R2u[a_1, a_2, i_1, i_2] += -B3[i_4, i_2] * T2[a_1, a_2, i_1, i_4]
        # @notensor B3 = nothing
        B4[i_4, a_3, a_1, i_1] := (g_oovv[i_3, i_4, a_3, a_4] * T2[a_1, a_4, i_1, i_3])
        R2u[a_1, a_2, i_1, i_2] += -B4[i_4, a_3, a_1, i_1] * T2[a_2, a_3, i_2, i_4] + B4[i_4, a_3, a_1, i_1] * T2[a_2, a_3, i_4, i_2]
        # @notensor B4 = nothing
        B5[a_3, a_2] := (g_oovv[i_3, i_4, a_3, a_4] * T2[a_2, a_4, i_4, i_3])
        R2u[a_1, a_2, i_1, i_2] += B5[a_3, a_2] * T2[a_1, a_3, i_1, i_2]
        # @notensor B5 = nothing
        B6[i_4, i_2] := (g_oovv[i_3, i_4, a_3, a_4] * T2[a_3, a_4, i_2, i_3])
        R2u[a_1, a_2, i_1, i_2] += B6[i_4, i_2] * T2[a_1, a_2, i_1, i_4]
        # @notensor B6 = nothing
        B7[a_4, a_2] := 2 * (g_oovv[i_3, i_4, a_3, a_4] * T2[a_2, a_3, i_4, i_3])
        R2u[a_1, a_2, i_1, i_2] += -B7[a_4, a_2] * T2[a_1, a_4, i_1, i_2]
        # @notensor B7 = nothing
        B8[i_4, a_3, a_1, i_2] := 0.5 * (g_oovv[i_3, i_4, a_3, a_4] * T2[a_1, a_4, i_3, i_2])
        R2u[a_1, a_2, i_1, i_2] += +B8[i_4, a_3, a_1, i_2] * T2[a_2, a_3, i_4, i_1]
        # @notensor B8 = nothing
        B9[i_3, i_4, i_1, i_2] := 0.5 * (g_oovv[i_3, i_4, a_3, a_4] * T2[a_3, a_4, i_1, i_2])
        R2u[a_1, a_2, i_1, i_2] += +B9[i_3, i_4, i_1, i_2] * T2[a_1, a_2, i_3, i_4]
        # @notensor B9 = nothing
        # @notensor g_oovv = nothing
        R2[a, b, i, j] := R2u[a, b, i, j] + R2u[b, a, j, i]
        # @notensor R2u = nothing
    end
    return R2
end


function calcresnew()
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
    g_voov = ITensor(deserialize("g_voov.jlbin"), a_2, i_3, i_2, a_3)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_3, i_3, i_1)
    #lhs tensor = I_vvoo rhs tensor = g_voov
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_2, i_3, i_2, a_3
    g_voov = replaceinds(g_voov, inds(g_voov) => (a_2, i_3, i_2, a_3))
    #lhs tensor = I_vvoo rhs tensor = T2_vvoo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_1, a_3, i_3, i_1
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_3, i_3, i_1))
    I_vvoo += -1 * g_voov * T2_vvoo
    T2_vvoo = nothing
    g_voov = nothing
    g_vvoo = ITensor(deserialize("g_vvoo.jlbin"), a_1, a_2, i_1, i_2)
    #lhs tensor = I_vvoo rhs tensor = g_vvoo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_1, a_2, i_1, i_2
    g_vvoo = replaceinds(g_vvoo, inds(g_vvoo) => (a_1, a_2, i_1, i_2))
    I_vvoo += 1 / 2 * g_vvoo
    g_vvoo = nothing
    g_vovo = ITensor(deserialize("g_vovo.jlbin"), a_2, i_3, a_3, i_1)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_3, i_3, i_2)
    #lhs tensor = I_vvoo rhs tensor = g_vovo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_2, i_3, a_3, i_1
    g_vovo = replaceinds(g_vovo, inds(g_vovo) => (a_2, i_3, a_3, i_1))
    #lhs tensor = I_vvoo rhs tensor = T2_vvoo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_1, a_3, i_3, i_2
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_3, i_3, i_2))
    I_vvoo += -1 * g_vovo * T2_vvoo
    T2_vvoo = nothing
    g_vovo = nothing
    g_voov = ITensor(deserialize("g_voov.jlbin"), a_2, i_3, i_2, a_3)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_3, i_1, i_3)
    #lhs tensor = I_vvoo rhs tensor = g_voov
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_2, i_3, i_2, a_3
    g_voov = replaceinds(g_voov, inds(g_voov) => (a_2, i_3, i_2, a_3))
    #lhs tensor = I_vvoo rhs tensor = T2_vvoo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_1, a_3, i_1, i_3
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_3, i_1, i_3))
    I_vvoo += 2 * g_voov * T2_vvoo
    T2_vvoo = nothing
    g_voov = nothing
    g_vovo = ITensor(deserialize("g_vovo.jlbin"), a_1, i_3, a_3, i_1)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_2, a_3, i_2, i_3)
    #lhs tensor = I_vvoo rhs tensor = g_vovo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_1, i_3, a_3, i_1
    g_vovo = replaceinds(g_vovo, inds(g_vovo) => (a_1, i_3, a_3, i_1))
    #lhs tensor = I_vvoo rhs tensor = T2_vvoo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_2, a_3, i_2, i_3
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_2, a_3, i_2, i_3))
    I_vvoo += -1 * g_vovo * T2_vvoo
    T2_vvoo = nothing
    g_vovo = nothing
    g_oooo = ITensor(deserialize("g_oooo.jlbin"), i_3, i_4, i_1, i_2)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_2, i_3, i_4)
    #lhs tensor = I_vvoo rhs tensor = g_oooo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = i_3, i_4, i_1, i_2
    g_oooo = replaceinds(g_oooo, inds(g_oooo) => (i_3, i_4, i_1, i_2))
    #lhs tensor = I_vvoo rhs tensor = T2_vvoo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_1, a_2, i_3, i_4
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_2, i_3, i_4))
    I_vvoo += 1 / 2 * g_oooo * T2_vvoo
    T2_vvoo = nothing
    g_oooo = nothing
    f_vv = ITensor(deserialize("f_vv.jlbin"), a_2, a_3)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_3, i_1, i_2)
    #lhs tensor = I_vvoo rhs tensor = f_vv
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_2, a_3
    f_vv = replaceinds(f_vv, inds(f_vv) => (a_2, a_3))
    #lhs tensor = I_vvoo rhs tensor = T2_vvoo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_1, a_3, i_1, i_2
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_3, i_1, i_2))
    I_vvoo += f_vv * T2_vvoo
    T2_vvoo = nothing
    f_vv = nothing
    g_vvvv = ITensor(deserialize("g_vvvv.jlbin"), a_1, a_2, a_3, a_4)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_3, a_4, i_1, i_2)
    #lhs tensor = I_vvoo rhs tensor = g_vvvv
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_1, a_2, a_3, a_4
    g_vvvv = replaceinds(g_vvvv, inds(g_vvvv) => (a_1, a_2, a_3, a_4))
    #lhs tensor = I_vvoo rhs tensor = T2_vvoo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_3, a_4, i_1, i_2
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_3, a_4, i_1, i_2))
    I_vvoo += 1 / 2 * g_vvvv * T2_vvoo
    T2_vvoo = nothing
    g_vvvv = nothing
    f_oo = ITensor(deserialize("f_oo.jlbin"), i_3, i_2)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_2, i_1, i_3)
    #lhs tensor = I_vvoo rhs tensor = f_oo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = i_3, i_2
    f_oo = replaceinds(f_oo, inds(f_oo) => (i_3, i_2))
    #lhs tensor = I_vvoo rhs tensor = T2_vvoo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_1, a_2, i_1, i_3
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_2, i_1, i_3))
    I_vvoo += -1 * f_oo * T2_vvoo
    T2_vvoo = nothing
    f_oo = nothing
    I_vv = ITensor(zeros(Float64, nv, nv), a_2, a_3)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_2, a_4, i_3, i_4)
    #lhs tensor = I_vv rhs tensor = g_oovv
    #lhs indices = a_2, a_3 rhs indices = i_3, i_4, a_3, a_4
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_3, i_4, a_3, a_4))
    #lhs tensor = I_vv rhs tensor = T2_vvoo
    #lhs indices = a_2, a_3 rhs indices = a_2, a_4, i_3, i_4
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_2, a_4, i_3, i_4))
    I_vv += g_oovv * T2_vvoo
    T2_vvoo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_3, i_1, i_2)
    #lhs tensor = I_vvoo rhs tensor = I_vv
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_2, a_3
    I_vv = replaceinds(I_vv, inds(I_vv) => (a_2, a_3))
    #lhs tensor = I_vvoo rhs tensor = T2_vvoo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_1, a_3, i_1, i_2
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_3, i_1, i_2))
    I_vvoo += -2 * I_vv * T2_vvoo
    T2_vvoo = nothing
    I_vv = nothing
    I_vovo = ITensor(zeros(Float64, nv, no, nv, no), a_1, i_4, a_4, i_1)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_3, i_1, i_3)
    #lhs tensor = I_vovo rhs tensor = g_oovv
    #lhs indices = a_1, i_4, a_4, i_1 rhs indices = i_3, i_4, a_3, a_4
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_3, i_4, a_3, a_4))
    #lhs tensor = I_vovo rhs tensor = T2_vvoo
    #lhs indices = a_1, i_4, a_4, i_1 rhs indices = a_1, a_3, i_1, i_3
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_3, i_1, i_3))
    I_vovo += g_oovv * T2_vvoo
    T2_vvoo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_2, a_4, i_2, i_4)
    #lhs tensor = I_vvoo rhs tensor = I_vovo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_1, i_4, a_4, i_1
    I_vovo = replaceinds(I_vovo, inds(I_vovo) => (a_1, i_4, a_4, i_1))
    #lhs tensor = I_vvoo rhs tensor = T2_vvoo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_2, a_4, i_2, i_4
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_2, a_4, i_2, i_4))
    I_vvoo += 2 * I_vovo * T2_vvoo
    T2_vvoo = nothing
    I_vovo = nothing
    I_vovo = ITensor(zeros(Float64, nv, no, nv, no), a_1, i_3, a_4, i_1)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_3, i_1, i_4)
    #lhs tensor = I_vovo rhs tensor = g_oovv
    #lhs indices = a_1, i_3, a_4, i_1 rhs indices = i_3, i_4, a_3, a_4
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_3, i_4, a_3, a_4))
    #lhs tensor = I_vovo rhs tensor = T2_vvoo
    #lhs indices = a_1, i_3, a_4, i_1 rhs indices = a_1, a_3, i_1, i_4
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_3, i_1, i_4))
    I_vovo += g_oovv * T2_vvoo
    T2_vvoo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_2, a_4, i_3, i_2)
    #lhs tensor = I_vvoo rhs tensor = I_vovo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_1, i_3, a_4, i_1
    I_vovo = replaceinds(I_vovo, inds(I_vovo) => (a_1, i_3, a_4, i_1))
    #lhs tensor = I_vvoo rhs tensor = T2_vvoo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_2, a_4, i_3, i_2
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_2, a_4, i_3, i_2))
    I_vvoo += I_vovo * T2_vvoo
    T2_vvoo = nothing
    I_vovo = nothing
    I_vv = ITensor(zeros(Float64, nv, nv), a_2, a_4)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_2, a_3, i_3, i_4)
    #lhs tensor = I_vv rhs tensor = g_oovv
    #lhs indices = a_2, a_4 rhs indices = i_3, i_4, a_3, a_4
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_3, i_4, a_3, a_4))
    #lhs tensor = I_vv rhs tensor = T2_vvoo
    #lhs indices = a_2, a_4 rhs indices = a_2, a_3, i_3, i_4
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_2, a_3, i_3, i_4))
    I_vv += g_oovv * T2_vvoo
    T2_vvoo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_4, i_1, i_2)
    #lhs tensor = I_vvoo rhs tensor = I_vv
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_2, a_4
    I_vv = replaceinds(I_vv, inds(I_vv) => (a_2, a_4))
    #lhs tensor = I_vvoo rhs tensor = T2_vvoo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_1, a_4, i_1, i_2
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_4, i_1, i_2))
    I_vvoo += I_vv * T2_vvoo
    T2_vvoo = nothing
    I_vv = nothing
    I_oo = ITensor(zeros(Float64, no, no), i_3, i_2)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_3, a_4, i_4, i_2)
    #lhs tensor = I_oo rhs tensor = g_oovv
    #lhs indices = i_3, i_2 rhs indices = i_3, i_4, a_3, a_4
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_3, i_4, a_3, a_4))
    #lhs tensor = I_oo rhs tensor = T2_vvoo
    #lhs indices = i_3, i_2 rhs indices = a_3, a_4, i_4, i_2
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_3, a_4, i_4, i_2))
    I_oo += g_oovv * T2_vvoo
    T2_vvoo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_2, i_1, i_3)
    #lhs tensor = I_vvoo rhs tensor = I_oo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = i_3, i_2
    I_oo = replaceinds(I_oo, inds(I_oo) => (i_3, i_2))
    #lhs tensor = I_vvoo rhs tensor = T2_vvoo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_1, a_2, i_1, i_3
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_2, i_1, i_3))
    I_vvoo += I_oo * T2_vvoo
    T2_vvoo = nothing
    I_oo = nothing
    I_vovo = ITensor(zeros(Float64, nv, no, nv, no), a_1, i_4, a_4, i_1)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_3, i_3, i_1)
    #lhs tensor = I_vovo rhs tensor = g_oovv
    #lhs indices = a_1, i_4, a_4, i_1 rhs indices = i_3, i_4, a_3, a_4
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_3, i_4, a_3, a_4))
    #lhs tensor = I_vovo rhs tensor = T2_vvoo
    #lhs indices = a_1, i_4, a_4, i_1 rhs indices = a_1, a_3, i_3, i_1
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_3, i_3, i_1))
    I_vovo += g_oovv * T2_vvoo
    T2_vvoo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_2, a_4, i_4, i_2)
    #lhs tensor = I_vvoo rhs tensor = I_vovo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_1, i_4, a_4, i_1
    I_vovo = replaceinds(I_vovo, inds(I_vovo) => (a_1, i_4, a_4, i_1))
    #lhs tensor = I_vvoo rhs tensor = T2_vvoo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_2, a_4, i_4, i_2
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_2, a_4, i_4, i_2))
    I_vvoo += 1 / 2 * I_vovo * T2_vvoo
    T2_vvoo = nothing
    I_vovo = nothing
    I_vovo = ITensor(zeros(Float64, nv, no, nv, no), a_1, i_3, a_3, i_1)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_4, i_1, i_4)
    #lhs tensor = I_vovo rhs tensor = g_oovv
    #lhs indices = a_1, i_3, a_3, i_1 rhs indices = i_3, i_4, a_3, a_4
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_3, i_4, a_3, a_4))
    #lhs tensor = I_vovo rhs tensor = T2_vvoo
    #lhs indices = a_1, i_3, a_3, i_1 rhs indices = a_1, a_4, i_1, i_4
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_4, i_1, i_4))
    I_vovo += g_oovv * T2_vvoo
    T2_vvoo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_2, a_3, i_3, i_2)
    #lhs tensor = I_vvoo rhs tensor = I_vovo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_1, i_3, a_3, i_1
    I_vovo = replaceinds(I_vovo, inds(I_vovo) => (a_1, i_3, a_3, i_1))
    #lhs tensor = I_vvoo rhs tensor = T2_vvoo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_2, a_3, i_3, i_2
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_2, a_3, i_3, i_2))
    I_vvoo += -2 * I_vovo * T2_vvoo
    T2_vvoo = nothing
    I_vovo = nothing
    I_vovo = ITensor(zeros(Float64, nv, no, nv, no), a_1, i_4, a_3, i_1)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_4, i_1, i_3)
    #lhs tensor = I_vovo rhs tensor = g_oovv
    #lhs indices = a_1, i_4, a_3, i_1 rhs indices = i_3, i_4, a_3, a_4
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_3, i_4, a_3, a_4))
    #lhs tensor = I_vovo rhs tensor = T2_vvoo
    #lhs indices = a_1, i_4, a_3, i_1 rhs indices = a_1, a_4, i_1, i_3
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_4, i_1, i_3))
    I_vovo += g_oovv * T2_vvoo
    T2_vvoo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_2, a_3, i_2, i_4)
    #lhs tensor = I_vvoo rhs tensor = I_vovo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_1, i_4, a_3, i_1
    I_vovo = replaceinds(I_vovo, inds(I_vovo) => (a_1, i_4, a_3, i_1))
    #lhs tensor = I_vvoo rhs tensor = T2_vvoo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_2, a_3, i_2, i_4
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_2, a_3, i_2, i_4))
    I_vvoo += -1 * I_vovo * T2_vvoo
    T2_vvoo = nothing
    I_vovo = nothing
    I_oo = ITensor(zeros(Float64, no, no), i_3, i_2)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_3, a_4, i_2, i_4)
    #lhs tensor = I_oo rhs tensor = g_oovv
    #lhs indices = i_3, i_2 rhs indices = i_3, i_4, a_3, a_4
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_3, i_4, a_3, a_4))
    #lhs tensor = I_oo rhs tensor = T2_vvoo
    #lhs indices = i_3, i_2 rhs indices = a_3, a_4, i_2, i_4
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_3, a_4, i_2, i_4))
    I_oo += g_oovv * T2_vvoo
    T2_vvoo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_2, i_1, i_3)
    #lhs tensor = I_vvoo rhs tensor = I_oo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = i_3, i_2
    I_oo = replaceinds(I_oo, inds(I_oo) => (i_3, i_2))
    #lhs tensor = I_vvoo rhs tensor = T2_vvoo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_1, a_2, i_1, i_3
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_2, i_1, i_3))
    I_vvoo += -2 * I_oo * T2_vvoo
    T2_vvoo = nothing
    I_oo = nothing
    I_vovo = ITensor(zeros(Float64, nv, no, nv, no), a_1, i_4, a_3, i_2)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_4, i_3, i_2)
    #lhs tensor = I_vovo rhs tensor = g_oovv
    #lhs indices = a_1, i_4, a_3, i_2 rhs indices = i_3, i_4, a_3, a_4
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_3, i_4, a_3, a_4))
    #lhs tensor = I_vovo rhs tensor = T2_vvoo
    #lhs indices = a_1, i_4, a_3, i_2 rhs indices = a_1, a_4, i_3, i_2
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_4, i_3, i_2))
    I_vovo += g_oovv * T2_vvoo
    T2_vvoo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_2, a_3, i_4, i_1)
    #lhs tensor = I_vvoo rhs tensor = I_vovo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_1, i_4, a_3, i_2
    I_vovo = replaceinds(I_vovo, inds(I_vovo) => (a_1, i_4, a_3, i_2))
    #lhs tensor = I_vvoo rhs tensor = T2_vvoo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_2, a_3, i_4, i_1
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_2, a_3, i_4, i_1))
    I_vvoo += 1 / 2 * I_vovo * T2_vvoo
    T2_vvoo = nothing
    I_vovo = nothing
    I_oooo = ITensor(zeros(Float64, no, no, no, no), i_3, i_4, i_1, i_2)
    g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_3, a_4, i_1, i_2)
    #lhs tensor = I_oooo rhs tensor = g_oovv
    #lhs indices = i_3, i_4, i_1, i_2 rhs indices = i_3, i_4, a_3, a_4
    g_oovv = replaceinds(g_oovv, inds(g_oovv) => (i_3, i_4, a_3, a_4))
    #lhs tensor = I_oooo rhs tensor = T2_vvoo
    #lhs indices = i_3, i_4, i_1, i_2 rhs indices = a_3, a_4, i_1, i_2
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_3, a_4, i_1, i_2))
    I_oooo += g_oovv * T2_vvoo
    T2_vvoo = nothing
    g_oovv = nothing
    T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"), a_1, a_2, i_3, i_4)
    #lhs tensor = I_vvoo rhs tensor = I_oooo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = i_3, i_4, i_1, i_2
    I_oooo = replaceinds(I_oooo, inds(I_oooo) => (i_3, i_4, i_1, i_2))
    #lhs tensor = I_vvoo rhs tensor = T2_vvoo
    #lhs indices = a_1, a_2, i_1, i_2 rhs indices = a_1, a_2, i_3, i_4
    T2_vvoo = replaceinds(T2_vvoo, inds(T2_vvoo) => (a_1, a_2, i_3, i_4))
    I_vvoo += 1 / 2 * I_oooo * T2_vvoo
    T2_vvoo = nothing
    I_oooo = nothing
    return I_vvoo + swapinds(I_vvoo, (a_1, i_1), (a_2, i_2)) #handwritten line
end
















































# ########################## Mem Saving Code June 15 2024 ###########################
# # This is written in such a way that the big g_xxxx TensorOperations
# # are only read when the subsequent terms need them
# # The g_xxxx and the binary contraction intermediates are thrown away
# # after the subsequent terms are calculated

# function addres_gvvoo(R2u::Array{Float64,4},nv,no)
#     g_vvoo = deserialize("g_vvoo.jlbin")
#     @tensor Trm1[a_1,a_2,i_1,i_2] := 0.5* g_vvoo[a_1,a_2,i_1,i_2]
#     @tensor R2u[a_1,a_2,i_1,i_2] += Trm1[a_1,a_2,i_1,i_2]
#     return R2u
# end

# function addres_gvvoo(R2u::ITensor,nv,no)
#     g_vvoo = ITensor(deserialize("g_vvoo.jlbin"),a_1,a_2,i_1,i_2)
#     Trm1 = ITensor(a_1,a_2,i_1,i_2)
#     Trm1 = 0.5 * g_vvoo
#     Trm1 = permute(Trm1, a_1, a_2, i_1, i_2, allow_alias=false)
#     R2u += Trm1
#     return R2u
# end

# function addres_gvoov(R2u::Array{Float64,4},nv,no,T2::Array{Float64,4})
#     g_voov = deserialize("g_voov.jlbin")
#     @tensor Trm2[a_1,a_2,i_1,i_2] := - g_voov[a_1,i_3,i_1,a_3] * T2[a_2,a_3,i_3,i_2]
#     @tensor R2u[a_1,a_2,i_1,i_2] += Trm2[a_1,a_2,i_1,i_2]
#     @tensor Trm5[a_1,a_2,i_1,i_2] := 2*g_voov[a_1,i_3,i_1,a_3] * T2[a_2,a_3,i_2,i_3]
#     @tensor R2u[a_1,a_2,i_1,i_2] += Trm5[a_1,a_2,i_1,i_2]
#     return R2u
# end

# function addres_gvoov(R2u::ITensor,nv,no,T2::ITensor)
#     i_3 = Index(no,"i_3")
#     a_3 = Index(nv,"a_3")
#     g_voov = ITensor(deserialize("g_voov.jlbin"),a_1,i_3,i_1,a_3)
#     Trm2 = -g_voov * replaceinds(T2,inds(T2)=>(a_2,a_3,i_3,i_2))
#     R2u += Trm2
#     Trm5 = 2*g_voov * replaceinds(T2,inds(T2)=>(a_2,a_3,i_2,i_3))
#     R2u += Trm5
#     return R2u
# end

# function addres_gvovo(R2u::Array{Float64,4},nv,no,T2::Array{Float64,4})
#     g_vovo = deserialize("g_vovo.jlbin")
#     @tensor Trm3[a_1,a_2,i_1,i_2] := - g_vovo[a_2,i_3,a_3,i_2] * T2[a_1,a_3,i_1,i_3]
#     @tensor R2u[a_1,a_2,i_1,i_2] += Trm3[a_1,a_2,i_1,i_2]
#     @tensor Trm4[a_1,a_2,i_1,i_2] := - g_vovo[a_2,i_3,a_3,i_1] * T2[a_1,a_3,i_3,i_2]
#     @tensor R2u[a_1,a_2,i_1,i_2] += Trm4[a_1,a_2,i_1,i_2]
#     return R2u
# end

# function addres_gvovo(R2u::ITensor, nv, no, T2::ITensor)
#     i_3 = Index(no, "i_3")
#     a_3 = Index(nv, "a_3")
#     g_vovo = ITensor(deserialize("g_vovo.jlbin"), a_2, i_3, a_3, i_2)
#     T2_perm1 = replaceinds(T2, inds(T2) => (a_1, a_3, i_1, i_3))
#     Trm3 = -g_vovo * T2_perm1
#     Trm3 = permute(Trm3, a_1, a_2, i_1, i_2, allow_alias=false)
#     R2u += Trm3
#     T2_perm2 = replaceinds(T2, inds(T2) => (a_1, a_3, i_3, i_2))
#     g_vovo = replaceinds(g_vovo, inds(g_vovo) => (a_2, i_3, a_3, i_1)) # Corrected: Replace indices of g_vovo for the second term
#     Trm4 = -g_vovo * T2_perm2
#     Trm4 = permute(Trm4, a_1, a_2, i_1, i_2, allow_alias=false)
#     R2u += Trm4
#     return R2u
# end

# function addres_goooo(R2u::Array{Float64,4},nv,no,T2::Array{Float64,4})
#     g_oooo = deserialize("g_oooo.jlbin")
#     @tensor Trm6[a_1,a_2,i_1,i_2] := 0.5*g_oooo[i_3,i_4,i_1,i_2] * T2[a_1,a_2,i_3,i_4]
#     @tensor R2u[a_1,a_2,i_1,i_2] += Trm6[a_1,a_2,i_1,i_2]
#     return R2u
# end

# function addres_goooo(R2u::ITensor, nv, no, T2::ITensor)
#     i_3 = Index(no, "i_3")
#     i_4 = Index(no, "i_4")
#     g_oooo = ITensor(deserialize("g_oooo.jlbin"), i_3, i_4, i_1, i_2)
#     T2 = replaceinds(T2, inds(T2) => (a_1, a_2, i_3, i_4))
#     Trm6 = 0.5 * g_oooo * T2
#     Trm6 = permute(Trm6, a_1, a_2, i_1, i_2, allow_alias=false)
#     R2u += Trm6
#     return R2u
# end


# function addres_gvvvv(R2u::Array{Float64,4},nv,no,T2::Array{Float64,4})
#     g_vvvv = deserialize("g_vvvv.jlbin")
#     @tensor Trm8[a_1,a_2,i_1,i_2] := + 0.5*g_vvvv[a_1,a_2,a_3,a_4] * T2[a_3,a_4,i_1,i_2]
#     @tensor R2u[a_1,a_2,i_1,i_2] += Trm8[a_1,a_2,i_1,i_2]
#     return R2u
# end


# function addres_gvvvv(R2u::ITensor, nv, no, T2::ITensor)
#     a_3 = Index(nv, "a_3")
#     a_4 = Index(nv, "a_4")
#     g_vvvv = ITensor(deserialize("g_vvvv.jlbin"), a_1, a_2, a_3, a_4)
#     T2 = replaceinds(T2, inds(T2) => (a_3, a_4, i_1, i_2))
#     Trm8 = 0.5 * g_vvvv * T2
#     Trm8 = permute(Trm8, a_1, a_2, i_1, i_2, allow_alias=false)
#     R2u += Trm8
#     return R2u
# end


# function addres_fvv(R2u,nv,no,T2)
#     fvv = deserialize("f_vv.jlbin")
#     @tensor Trm7[a_1,a_2,i_1,i_2] := fvv[a_2,a_3] * T2[a_1,a_3,i_1,i_2]
#     @tensor R2u[a_1,a_2,i_1,i_2] += Trm7[a_1,a_2,i_1,i_2]
#     return R2u
# end

# function addres_fvv(R2u::ITensor, nv, no, T2::ITensor)
#     a_3 = Index(nv, "a_3")
#     fvv = ITensor(deserialize("f_vv.jlbin"), a_2, a_3)
#     T2 = replaceinds(T2, inds(T2) => (a_1, a_3, i_1, i_2))
#     Trm7 = fvv * T2
#     Trm7 = permute(Trm7, a_1, a_2, i_1, i_2, allow_alias=false)
#     R2u += Trm7
#     return R2u
# end

# function addres_foo(R2u,nv,no,T2)
#     foo = deserialize("f_oo.jlbin")
#     @tensor Trm9[a_1,a_2,i_1,i_2] := - foo[i_3,i_2] * T2[a_1,a_2,i_1,i_3]
#     @tensor R2u[a_1,a_2,i_1,i_2] += Trm9[a_1,a_2,i_1,i_2]
#     return R2u
# end

# function addres_foo(R2u::ITensor, nv, no, T2::ITensor)
#     i_3 = Index(no, "i_3")
#     foo = ITensor(deserialize("f_oo.jlbin"), i_3, i_2)
#     T2 = replaceinds(T2, inds(T2) => (a_1, a_2, i_1, i_3))
#     Trm9 = -foo * T2
#     Trm9 = permute(Trm9, a_1, a_2, i_1, i_2, allow_alias=false)
#     R2u += Trm9
#     return R2u
# end

# function addres_goovv(R2u,nv,no,T2)
#     g_oovv = deserialize("g_oovv.jlbin")
#     @tensor B1[i_4,a_4,a_1,i_1] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_3,i_1,i_3])
#     @tensor R2u[a_1,a_2,i_1,i_2] +=  B1[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_2,i_4]- B1[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_4,i_2]
#     @tensor B2[i_4,a_4,a_1,i_1] := 0.5*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_3,i_3,i_1])
#     @tensor R2u[a_1,a_2,i_1,i_2] += B2[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_4,i_2]
#     @tensor B3[i_4,i_2] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_3,i_2])
#     @tensor R2u[a_1,a_2,i_1,i_2] +=  -B3[i_4,i_2] * T2[a_1,a_2,i_1,i_4]
#     @tensor B4[i_4,a_3,a_1,i_1] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_4,i_1,i_3])
#     @tensor R2u[a_1,a_2,i_1,i_2] += -B4[i_4,a_3,a_1,i_1] * T2[a_2,a_3,i_2,i_4] + B4[i_4,a_3,a_1,i_1] * T2[a_2,a_3,i_4,i_2]
#     @tensor B5[a_3,a_2] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_2,a_4,i_4,i_3])
#     @tensor R2u[a_1,a_2,i_1,i_2] += B5[a_3,a_2] * T2[a_1,a_3,i_1,i_2]
#     @tensor B6[i_4,i_2] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_2,i_3])
#     @tensor R2u[a_1,a_2,i_1,i_2] += B6[i_4,i_2] * T2[a_1,a_2,i_1,i_4]
#     @tensor B7[a_4,a_2] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_2,a_3,i_4,i_3])
#     @tensor R2u[a_1,a_2,i_1,i_2] += - B7[a_4,a_2] * T2[a_1,a_4,i_1,i_2]
#     @tensor B8[i_4,a_3,a_1,i_2] := 0.5*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_4,i_3,i_2])
#     @tensor R2u[a_1,a_2,i_1,i_2] +=  +B8[i_4,a_3,a_1,i_2] * T2[a_2,a_3,i_4,i_1]
#     @tensor B9[i_3,i_4,i_1,i_2] := 0.5* (g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_1,i_2])
#     @tensor R2u[a_1,a_2,i_1,i_2] += + B9[i_3,i_4,i_1,i_2] * T2[a_1,a_2,i_3,i_4]
# end


# function addres_goovvb1(R2u::Array{Float64},nv,no,T2::Array{Float64})
#     g_oovv = deserialize("g_oovv.jlbin")
#     @tensor B1[i_4,a_4,a_1,i_1] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_3,i_1,i_3])
#     @tensor R2u[a_1,a_2,i_1,i_2] +=  B1[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_2,i_4]- B1[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_4,i_2]
#     return R2u
# end

# function addres_goovvb1(R2u::ITensor,nv,no,T2::ITensor)
#     i_3 = Index(no, "i_3")
#     a_3 = Index(nv, "a_3")
#     i_4 = Index(no, "i_4")
#     a_4 = Index(nv, "a_4")
#     g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
#     B1 = ITensor(i_4, a_4, a_1, i_1)
#     B1 = 2 * g_oovv * replaceinds(T2, inds(T2) => (a_1, a_3, i_1, i_3))
#     tmp1 = replaceinds(B1, inds(B1) => (i_4, a_4, a_1, i_1))*replaceinds(T2, inds(T2) => (a_2, a_4, i_2, i_4)) - replaceinds(B1, inds(B1) => (i_4, a_4, a_1, i_1))*replaceinds(T2, inds(T2) => (a_2, a_4, i_4, i_2))
#     R2u += tmp1
#     return R2u

# end

# function addres_goovvb2(R2u::Array{Float64},nv,no,T2::Array{Float64})
#     g_oovv = deserialize("g_oovv.jlbin")
#     @tensor B2[i_4,a_4,a_1,i_1] := 0.5*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_3,i_3,i_1])
#     @tensor R2u[a_1,a_2,i_1,i_2] += B2[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_4,i_2]
#     return R2u
# end

# function addres_goovvb2(R2u::ITensor, nv, no, T2::ITensor)
#     i_3 = Index(no, "i_3")
#     a_3 = Index(nv, "a_3")
#     i_4 = Index(no, "i_4")
#     a_4 = Index(nv, "a_4")
#     g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
#     B2 = ITensor(i_4, a_4, a_1, i_1)
#     B2 = 0.5 * g_oovv * replaceinds(T2, inds(T2) => (a_1, a_3, i_3, i_1))
#     tmp2 = replaceinds(B2, inds(B2) => (i_4, a_4, a_1, i_1)) * replaceinds(T2, inds(T2) => (a_2, a_4, i_4, i_2))
#     R2u += tmp2
#     return R2u
# end

# function addres_goovvb3(R2u::Array{Float64,4},nv,no,T2::Array{Float64,4})
#     g_oovv = deserialize("g_oovv.jlbin")
#     @tensor B3[i_4,i_2] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_3,i_2])
#     @tensor R2u[a_1,a_2,i_1,i_2] +=  -B3[i_4,i_2] * T2[a_1,a_2,i_1,i_4]
#     return R2u
# end

# function addres_goovvb3(R2u::ITensor, nv, no, T2::ITensor)
#     i_3 = Index(no, "i_3")
#     a_3 = Index(nv, "a_3")
#     i_4 = Index(no, "i_4")
#     a_4 = Index(nv, "a_4")
#     g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
#     B3 = ITensor(i_4, i_2)
#     B3 = 2 * g_oovv * replaceinds(T2, inds(T2) => (a_3, a_4, i_3, i_2))
#     tmp3 = -replaceinds(B3, inds(B3) => (i_4, i_2)) * replaceinds(T2, inds(T2) => (a_1, a_2, i_1, i_4))
#     tmp3 = permute(tmp3, a_1, a_2, i_1, i_2, allow_alias=false)
#     R2u += tmp3
#     return R2u
# end

# function addres_goovvb4(R2u::Array{Float64,4},nv,no,T2::Array{Float64,4})
#     g_oovv = deserialize("g_oovv.jlbin")
#     @tensor B4[i_4,a_3,a_1,i_1] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_4,i_1,i_3])
#     @tensor R2u[a_1,a_2,i_1,i_2] += -B4[i_4,a_3,a_1,i_1] * T2[a_2,a_3,i_2,i_4] + B4[i_4,a_3,a_1,i_1] * T2[a_2,a_3,i_4,i_2]
#     return R2u
# end

# function addres_goovvb4(R2u::ITensor, nv, no, T2::ITensor)
#     i_3 = Index(no, "i_3")
#     a_3 = Index(nv, "a_3")
#     i_4 = Index(no, "i_4")
#     a_4 = Index(nv, "a_4")
#     g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
#     B4 = ITensor(i_4, a_3, a_1, i_1)
#     B4 = g_oovv * replaceinds(T2, inds(T2) => (a_1, a_4, i_1, i_3))
#     tmp4 = -replaceinds(B4, inds(B4) => (i_4, a_3, a_1, i_1)) * replaceinds(T2, inds(T2) => (a_2, a_3, i_2, i_4))
#     tmp4 += replaceinds(B4, inds(B4) => (i_4, a_3, a_1, i_1)) * replaceinds(T2, inds(T2) => (a_2, a_3, i_4, i_2))
#     tmp4 = permute(tmp4, a_1, a_2, i_1, i_2, allow_alias=false)
#     R2u += tmp4
#     return R2u
# end

# function addres_goovvb5(R2u::Array{Float64,4},nv,no,T2::Array{Float64,4})
#     g_oovv = deserialize("g_oovv.jlbin")
#     @tensor B5[a_3,a_2] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_2,a_4,i_4,i_3])
#     @tensor R2u[a_1,a_2,i_1,i_2] += B5[a_3,a_2] * T2[a_1,a_3,i_1,i_2]
#     return R2u
# end

# function addres_goovvb5(R2u::ITensor, nv, no, T2::ITensor)
#     i_3 = Index(no, "i_3")
#     a_3 = Index(nv, "a_3")
#     i_4 = Index(no, "i_4")
#     a_4 = Index(nv, "a_4")
#     g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
#     B5 = g_oovv * replaceinds(T2, inds(T2) => (a_2, a_4, i_4, i_3))
#     B5 = permute(B5, a_3, a_2, allow_alias=false)
#     R2u += B5 * replaceinds(T2, inds(T2) => (a_1, a_3, i_1, i_2))
#     return R2u
# end

# function addres_goovvb6(R2u::Array{Float64,4},nv,no,T2::Array{Float64,4})
#     g_oovv = deserialize("g_oovv.jlbin")
#     @tensor B6[i_4,i_2] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_2,i_3])
#     @tensor R2u[a_1,a_2,i_1,i_2] += B6[i_4,i_2] * T2[a_1,a_2,i_1,i_4]
#     return R2u
# end

# function addres_goovvb6(R2u::ITensor, nv, no, T2::ITensor)
#     i_3 = Index(no, "i_3")
#     a_3 = Index(nv, "a_3")
#     i_4 = Index(no, "i_4")
#     a_4 = Index(nv, "a_4")
#     g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
#     B6 = g_oovv * replaceinds(T2, inds(T2) => (a_3, a_4, i_2, i_3))
#     B6 = permute(B6, i_4, i_2, allow_alias=false)
#     R2u += B6 * replaceinds(T2, inds(T2) => (a_1, a_2, i_1, i_4))
#     return R2u
# end

# function addres_goovvb7(R2u::Array{Float64,4},nv,no,T2::Array{Float64,4})
#     g_oovv = deserialize("g_oovv.jlbin")
#     @tensor B7[a_4,a_2] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_2,a_3,i_4,i_3])
#     @tensor R2u[a_1,a_2,i_1,i_2] += - B7[a_4,a_2] * T2[a_1,a_4,i_1,i_2]
#     return R2u
# end

# function addres_goovvb7(R2u::ITensor, nv, no, T2::ITensor)
#     i_3 = Index(no, "i_3")
#     a_3 = Index(nv, "a_3")
#     i_4 = Index(no, "i_4")
#     a_4 = Index(nv, "a_4")
#     g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
#     B7 = 2 * g_oovv * replaceinds(T2, inds(T2) => (a_2, a_3, i_4, i_3))
#     B7 = permute(B7, a_4, a_2, allow_alias=false)
#     R2u += -B7 * replaceinds(T2, inds(T2) => (a_1, a_4, i_1, i_2))
#     return R2u
# end

# function addres_goovvb8(R2u::Array{Float64,4},nv,no,T2::Array{Float64,4})
#     g_oovv = deserialize("g_oovv.jlbin")
#     @tensor B8[i_4,a_3,a_1,i_2] := 0.5*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_4,i_3,i_2])
#     @tensor R2u[a_1,a_2,i_1,i_2] +=  +B8[i_4,a_3,a_1,i_2] * T2[a_2,a_3,i_4,i_1]
#     return R2u
# end

# function addres_goovvb8(R2u::ITensor, nv, no, T2::ITensor)
#     i_3 = Index(no, "i_3")
#     a_3 = Index(nv, "a_3")
#     i_4 = Index(no, "i_4")
#     a_4 = Index(nv, "a_4")
#     g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
#     B8 = 0.5 * g_oovv * replaceinds(T2, inds(T2) => (a_1, a_4, i_3, i_2))
#     B8 = permute(B8, i_4, a_3, a_1, i_2, allow_alias=false)
#     R2u += B8 * replaceinds(T2, inds(T2) => (a_2, a_3, i_4, i_1))
#     return R2u
# end

# function addres_goovvb9(R2u,nv,no,T2)
#     g_oovv = deserialize("g_oovv.jlbin")
#     @tensor B9[i_3,i_4,i_1,i_2] := 0.5* (g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_1,i_2])
#     @tensor R2u[a_1,a_2,i_1,i_2] += + B9[i_3,i_4,i_1,i_2] * T2[a_1,a_2,i_3,i_4]
#     return R2u
# end

# function addres_goovvb9(R2u::ITensor, nv, no, T2::ITensor)
#     i_3 = Index(no, "i_3")
#     i_4 = Index(no, "i_4")
#     a_3 = Index(nv, "a_3")
#     a_4 = Index(nv, "a_4")
#     g_oovv = ITensor(deserialize("g_oovv.jlbin"), i_3, i_4, a_3, a_4)
#     B9 = 0.5 * g_oovv * replaceinds(T2, inds(T2) => (a_3, a_4, i_1, i_2))
#     B9 = permute(B9, i_3, i_4, i_1, i_2, allow_alias=false)
#     R2u += B9 * replaceinds(T2, inds(T2) => (a_1, a_2, i_3, i_4))
#     return R2u
# end


# function calcresnew(T2::Array{Float64,4})
#     nv = deserialize("nv.jlbin")
#     no = deserialize("no.jlbin")
#     R2u = zeros(Float64,nv,nv,no,no)
#     R2 =  zeros(Float64,nv,nv,no,no)


#     R2u = addres_gvvoo(R2u,nv,no)
#     # display(R2u)
#     R2u = addres_gvoov(R2u,nv,no,T2)
#     # display(R2u)
#     R2u = addres_gvovo(R2u,nv,no,T2)
#     # display(R2u)
#     R2u = addres_goooo(R2u,nv,no,T2)
#     # display(R2u)
#     R2u = addres_gvvvv(R2u,nv,no,T2)
#     # display(R2u)
#     R2u = addres_fvv(R2u,nv,no,T2)
#     R2u = addres_foo(R2u,nv,no,T2)
#     R2u = addres_goovvb1(R2u,nv,no,T2)
#     R2u = addres_goovvb2(R2u,nv,no,T2)
#     R2u = addres_goovvb3(R2u,nv,no,T2)
#     R2u = addres_goovvb4(R2u,nv,no,T2)
#     R2u = addres_goovvb5(R2u,nv,no,T2)
#     R2u = addres_goovvb6(R2u,nv,no,T2)
#     R2u = addres_goovvb7(R2u,nv,no,T2)
#     R2u = addres_goovvb8(R2u,nv,no,T2)
#     R2u = addres_goovvb9(R2u,nv,no,T2)
#     @tensor R2[a,b,i,j] += R2u[a,b,i,j] + R2u[b,a,j,i]
#     return R2
# end

# function calcresnew(T2::ITensor)
#     nv = deserialize("nv.jlbin")
#     no = deserialize("no.jlbin")
#     R2u = ITensor(zeros(Float64,nv,nv,no,no),a_1,a_2,i_1,i_2)
#     R2  = ITensor(zeros(Float64,nv,nv,no,no),a_1,a_2,i_1,i_2)


#     R2u = addres_gvvoo(R2u,nv,no)
#     # @show R2u
#     R2u = addres_gvoov(R2u,nv,no,T2)
#     # @show R2u
#     R2u = addres_gvovo(R2u,nv,no,T2)
#     # @show R2u
#     R2u = addres_goooo(R2u,nv,no,T2)
#     # @show R2u
#     R2u = addres_gvvvv(R2u,nv,no,T2)
#     # @show R2u
#     R2u = addres_fvv(R2u,nv,no,T2)
#     R2u = addres_foo(R2u,nv,no,T2)
#     R2u = addres_goovvb1(R2u,nv,no,T2)
#     R2u = addres_goovvb2(R2u,nv,no,T2)
#     R2u = addres_goovvb3(R2u,nv,no,T2)
#     R2u = addres_goovvb4(R2u,nv,no,T2)
#     R2u = addres_goovvb5(R2u,nv,no,T2)
#     R2u = addres_goovvb6(R2u,nv,no,T2)
#     R2u = addres_goovvb7(R2u,nv,no,T2)
#     R2u = addres_goovvb8(R2u,nv,no,T2)
#     R2u = addres_goovvb9(R2u,nv,no,T2)
#     R2 = R2u + permute(R2u,a_2,a_1,i_2,i_1)
#     return R2
# end
