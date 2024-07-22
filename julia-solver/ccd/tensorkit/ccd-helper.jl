using DelimitedFiles,TensorOperations,LinearAlgebra,MKL,TensorKit, Serialization





function main(pathtofcidump)
    linenum=0
    norb=0
    nelec=0
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
    hnuc::Float64 = data[end,1]
    
    
    data = copy(data[1:end-1, :])
    # h = Array{Union{Missing, Float64}}(missing, norb, norb);
    # g = Array{Union{Missing, Float64}}(missing, norb, norb, norb, norb);
    h::Array{Float64,2} = fill(0.0, norb, norb)
    g::Array{Float64,4} = fill(0.0, norb, norb, norb, norb)
    l::Int64 = length(data[:,1])
    non_redundant_indices = []
    for i in 1:l
        if(data[i,4]== 0 && data[i,5]==0)
            I = round(Int,data[i,2])
            J = round(Int,data[i,3])
            h[I,J] = data[i,1]
        else
            I = round(Int,data[i,2])
            J = round(Int,data[i,3])
            K = round(Int,data[i,4])
            L = round(Int,data[i,5])
            push!(non_redundant_indices, [I, J, K, L])
            g[I,J,K,L] = data[i,1]
        end
    end
    for (I,J,K,L) in non_redundant_indices
        open("non_redundant_indices.txt", "a") do f
            println(f, I,J,K,L,"   ",g[I,J,K,L])
        end
        g[K,L,I,J]=g[I,J,K,L]
        g[J,I,L,K]=g[I,J,K,L]
        g[L,K,J,I]=g[I,J,K,L]
        g[J,I,K,L]=g[I,J,K,L]
        g[L,K,I,J]=g[I,J,K,L]
        g[I,J,L,K]=g[I,J,K,L]
        g[K,L,J,I]=g[I,J,K,L]

    end
    for I in 1:norb
        for J in 1:I
            h[J,I] = h[I,J]
        end
    end
    no = round(Int,nelec/2)
    nv = norb - no
    h = convert(Array{Float64}, h)
    g = convert(Array{Float64}, g)
    K = zeros(nv,nv,no,no)
    for a in 1:nv, b in 1:nv, i in 1:no, j in 1:no
        K[a,b,i,j] = g[no+a,i,no+b,j]
    end
    H1::Float64 = 0.0;
    H2::Float64 = 0.0;
    for i in 1:no
        H1 = H1+  h[i,i]
        for j in 1:no
            H2 = H2 +  2 * g[i,i,j,j] - g[i,j,i,j]
        end
    end
    erhf = 2*H1 + H2 + hnuc
    f = zeros(size(h)) # fock matrix Initialization
    for p in 1:norb , q in 1:norb
        s_pq=0
        for i in 1:no
            s_pq += 2*g[p,q,i,i]-g[p,i,i,q]
        end
        f[p,q] = h[p,q] + s_pq
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
    nv::Int64 = deserialize("nv.jlbin")
    no::Int64 = deserialize("no.jlbin")
    norb = nv + no
################## Amplitude Initialization ######################################
    # K = deserialize("K.jlbin")
    # f = deserialize("f.jlbin")
    K= TensorMap(deserialize("K.jlbin"),ℝ^nv ⊗ ℝ^nv,ℝ^no ⊗ ℝ^no)
    f= TensorMap(deserialize("f.jlbin"),ℝ^norb,ℝ^norb)
    T2 = - copy(K) 
    for a in 1:nv, b in 1:nv, i in 1:no, j in 1:no
        T2[a,b,i,j] = T2[a,b,i,j]/(f[no+a,no+a] + f[no+b,no+b] - f[i,i] - f[j,j])
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
    g_vvoo::Array{Float64,4} = zeros(nv,nv,no,no)
    g_voov::Array{Float64,4} = zeros(nv,no,no,nv)
    g_vovo::Array{Float64,4} = zeros(nv,no,nv,no)
    g_oovv::Array{Float64,4} = zeros(no,no,nv,nv)
    g_oooo::Array{Float64,4} = zeros(no,no,no,no)
    g_vvvv::Array{Float64,4} = zeros(nv,nv,nv,nv)
    fvv = zeros(Float64,nv,nv)
    foo = zeros(Float64,no,no)
    for a in 1:nv, b in 1:nv
        fvv[a,b] = f[no+a,no+b]
        
        for i in 1:no, j in 1:no
            g_vvoo[a,b,i,j] = g[no+a,i,no+b,j]
            g_voov[a,i,j,b] = g[no+a,j,i,no+b]
            g_vovo[a,i,b,j] = g[no+a,no+b,i,j]
            g_oovv[i,j,a,b] = g[i,no+a,j,no+b]
            foo[i,j] = f[i,j]
        end
    end
    for i in 1:no, j in 1:no, k in 1:no, l in 1:no
        g_oooo[i,j,k,l] = g[i,k,j,l]
    end
    for a in 1:nv, b in 1:nv, c in 1:nv, d in 1:nv
        g_vvvv[a,b,c,d] = g[no+a,no+c,no+b,no+d]
    end
    serialize("g_vvoo.jlbin", g_vvoo)
    serialize("g_voov.jlbin", g_voov)
    serialize("g_vovo.jlbin", g_vovo)
    serialize("g_oovv.jlbin", g_oovv)
    serialize("g_oooo.jlbin", g_oooo)
    serialize("g_vvvv.jlbin", g_vvvv)
    serialize("K.jlbin", K)
    serialize("fvv.jlbin", fvv)
    serialize("foo.jlbin", foo)
end






function calculate_residual(T2,R2u,R2)
    nv = deserialize("nv.jlbin")
    no = deserialize("no.jlbin")
    # g_vvoo,g_voov,g_vovo,g_oovv,g_oooo,g_vvvv = deserialize("g_vvoo.jlbin"),deserialize("g_voov.jlbin"),deserialize("g_vovo.jlbin"),deserialize("g_oovv.jlbin"),deserialize("g_oooo.jlbin"),deserialize("g_vvvv.jlbin")
    # fvv , foo = deserialize("fvv.jlbin"),deserialize("foo.jlbin")
    fvv = TensorMap(deserialize("fvv.jlbin"),ℝ^nv,ℝ^nv)
    foo = TensorMap(deserialize("foo.jlbin"),ℝ^no,ℝ^no)
    g_vvoo = TensorMap(deserialize("g_vvoo.jlbin"),ℝ^nv ⊗ ℝ^nv,ℝ^no ⊗ ℝ^no)
    g_voov = TensorMap(deserialize("g_voov.jlbin"),ℝ^nv ⊗ ℝ^no,ℝ^no ⊗ ℝ^nv) 
    g_vovo = TensorMap(deserialize("g_vovo.jlbin"),ℝ^nv ⊗ ℝ^no,ℝ^nv ⊗ ℝ^no)
    g_oovv = TensorMap(deserialize("g_oovv.jlbin"),ℝ^no ⊗ ℝ^no,ℝ^nv ⊗ ℝ^nv)
    g_oooo = TensorMap(deserialize("g_oooo.jlbin"),ℝ^no ⊗ ℝ^no,ℝ^no ⊗ ℝ^no)
    g_vvvv = TensorMap(deserialize("g_vvvv.jlbin"),ℝ^nv ⊗ ℝ^nv,ℝ^nv ⊗ ℝ^nv) 
    @tensor begin
        R2u[a_1,a_2,i_1,i_2]= 0.5* g_vvoo[a_1,a_2,i_1,i_2] - g_voov[a_1,i_3,i_1,a_3] * T2[a_2,a_3,i_3,i_2] - g_vovo[a_2,i_3,a_3,i_2] * T2[a_1,a_3,i_1,i_3]- g_vovo[a_2,i_3,a_3,i_1] * T2[a_1,a_3,i_3,i_2]+ 2*g_voov[a_1,i_3,i_1,a_3] * T2[a_2,a_3,i_2,i_3]+ 0.5*g_oooo[i_3,i_4,i_1,i_2] * T2[a_1,a_2,i_3,i_4]+ fvv[a_2,a_3] * T2[a_1,a_3,i_1,i_2]+ 0.5*g_vvvv[a_1,a_2,a_3,a_4] * T2[a_3,a_4,i_1,i_2]- foo[i_3,i_2] * T2[a_1,a_2,i_1,i_3]+ 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_3,i_1,i_3]) * T2[a_2,a_4,i_2,i_4]- 2 *(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_3,i_1,i_3]) * T2[a_2,a_4,i_4,i_2]+ 0.5*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_3,i_3,i_1]) * T2[a_2,a_4,i_4,i_2]- 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_3,i_2]) * T2[a_1,a_2,i_1,i_4]- 1*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_4,i_1,i_3]) * T2[a_2,a_3,i_2,i_4]+ (g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_4,i_1,i_3]) * T2[a_2,a_3,i_4,i_2]+ (g_oovv[i_3,i_4,a_3,a_4] * T2[a_2,a_4,i_4,i_3]) * T2[a_1,a_3,i_1,i_2]+ (g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_2,i_3]) * T2[a_1,a_2,i_1,i_4]- 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_2,a_3,i_4,i_3]) * T2[a_1,a_4,i_1,i_2]+ 0.5*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_4,i_3,i_2]) * T2[a_2,a_3,i_4,i_1]+ 0.5* (g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_1,i_2]) * T2[a_1,a_2,i_3,i_4]
        R2[a,b,i,j] = R2u[a,b,i,j] + R2u[b,a,j,i]
     end
     return R2
end

function calculate_residual_memeff_gc(T2)
    nv = deserialize("nv.jlbin")
    no = deserialize("no.jlbin")
    fvv = TensorMap(deserialize("fvv.jlbin"),ℝ^nv,ℝ^nv)
    foo = TensorMap(deserialize("foo.jlbin"),ℝ^no,ℝ^no)
    g_vvoo = TensorMap(deserialize("g_vvoo.jlbin"),ℝ^nv ⊗ ℝ^nv,ℝ^no ⊗ ℝ^no)
    g_voov = TensorMap(deserialize("g_voov.jlbin"),ℝ^nv ⊗ ℝ^no,ℝ^no ⊗ ℝ^nv) 
    g_vovo = TensorMap(deserialize("g_vovo.jlbin"),ℝ^nv ⊗ ℝ^no,ℝ^nv ⊗ ℝ^no)
    g_oovv = TensorMap(deserialize("g_oovv.jlbin"),ℝ^no ⊗ ℝ^no,ℝ^nv ⊗ ℝ^nv)
    g_oooo = TensorMap(deserialize("g_oooo.jlbin"),ℝ^no ⊗ ℝ^no,ℝ^no ⊗ ℝ^no)
    g_vvvv = TensorMap(deserialize("g_vvvv.jlbin"),ℝ^nv ⊗ ℝ^nv,ℝ^nv ⊗ ℝ^nv)
    R2u = TensorMap(zeros(Float64,nv,nv,no,no),ℝ^nv ⊗ ℝ^nv,ℝ^no ⊗ ℝ^no)
    R2 = TensorMap(zeros(Float64,nv,nv,no,no),ℝ^nv ⊗ ℝ^nv,ℝ^no ⊗ ℝ^no)
    @tensor begin
        Trm1[a_1,a_2,i_1,i_2] := 0.5* g_vvoo[a_1,a_2,i_1,i_2]
        R2u[a_1,a_2,i_1,i_2] += Trm1[a_1,a_2,i_1,i_2]
        @notensor Trm1 = nothing
        @notensor g_vvoo = nothing
        GC.gc()
        Trm2[a_1,a_2,i_1,i_2] := - g_voov[a_1,i_3,i_1,a_3] * T2[a_2,a_3,i_3,i_2]
        R2u[a_1,a_2,i_1,i_2] += Trm2[a_1,a_2,i_1,i_2]
        @notensor Trm2 = nothing
        GC.gc()
        Trm3[a_1,a_2,i_1,i_2] := - g_vovo[a_2,i_3,a_3,i_2] * T2[a_1,a_3,i_1,i_3]
        R2u[a_1,a_2,i_1,i_2] += Trm3[a_1,a_2,i_1,i_2]
        @notensor Trm3 = nothing
        GC.gc()
        Trm4[a_1,a_2,i_1,i_2] := - g_vovo[a_2,i_3,a_3,i_1] * T2[a_1,a_3,i_3,i_2]
        R2u[a_1,a_2,i_1,i_2] += Trm4[a_1,a_2,i_1,i_2]
        @notensor Trm4 = nothing
        @notensor g_vovo = nothing
        GC.gc()
        Trm5[a_1,a_2,i_1,i_2] := 2*g_voov[a_1,i_3,i_1,a_3] * T2[a_2,a_3,i_2,i_3]
        R2u[a_1,a_2,i_1,i_2] += Trm5[a_1,a_2,i_1,i_2]
        @notensor Trm5 = nothing
        @notensor g_voov = nothing
        GC.gc()
        Trm6[a_1,a_2,i_1,i_2] := 0.5*g_oooo[i_3,i_4,i_1,i_2] * T2[a_1,a_2,i_3,i_4]
        R2u[a_1,a_2,i_1,i_2] += Trm6[a_1,a_2,i_1,i_2]
        @notensor Trm6 = nothing
        @notensor g_oooo = nothing
        GC.gc()
        Trm7[a_1,a_2,i_1,i_2] := fvv[a_2,a_3] * T2[a_1,a_3,i_1,i_2]
        R2u[a_1,a_2,i_1,i_2] += Trm7[a_1,a_2,i_1,i_2]
        @notensor Trm7 = nothing
        @notensor fvv = nothing
        GC.gc()
        Trm8[a_1,a_2,i_1,i_2] := + 0.5*g_vvvv[a_1,a_2,a_3,a_4] * T2[a_3,a_4,i_1,i_2]
        R2u[a_1,a_2,i_1,i_2] += Trm8[a_1,a_2,i_1,i_2]
        @notensor Trm8 = nothing
        @notensor g_vvvv = nothing
        GC.gc()
        Trm9[a_1,a_2,i_1,i_2] := - foo[i_3,i_2] * T2[a_1,a_2,i_1,i_3]
        R2u[a_1,a_2,i_1,i_2] += Trm9[a_1,a_2,i_1,i_2]
        @notensor Trm9 = nothing
        @notensor foo = nothing
        GC.gc()
        B1[i_4,a_4,a_1,i_1] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_3,i_1,i_3])
        R2u[a_1,a_2,i_1,i_2] +=  B1[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_2,i_4]- B1[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_4,i_2]
        @notensor B1 = nothing
        GC.gc()
        B2[i_4,a_4,a_1,i_1] := 0.5*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_3,i_3,i_1])
        R2u[a_1,a_2,i_1,i_2] += B2[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_4,i_2]
        @notensor B2 = nothing
        GC.gc()
        B3[i_4,i_2] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_3,i_2])
        R2u[a_1,a_2,i_1,i_2] +=  -B3[i_4,i_2] * T2[a_1,a_2,i_1,i_4]
        @notensor B3 = nothing
        GC.gc()
        B4[i_4,a_3,a_1,i_1] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_4,i_1,i_3])
        R2u[a_1,a_2,i_1,i_2] += -B4[i_4,a_3,a_1,i_1] * T2[a_2,a_3,i_2,i_4] + B4[i_4,a_3,a_1,i_1] * T2[a_2,a_3,i_4,i_2]
        @notensor B4 = nothing
        GC.gc()
        B5[a_3,a_2] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_2,a_4,i_4,i_3])
        R2u[a_1,a_2,i_1,i_2] += B5[a_3,a_2] * T2[a_1,a_3,i_1,i_2]
        @notensor B5 = nothing
        GC.gc()
        B6[i_4,i_2] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_2,i_3])
        R2u[a_1,a_2,i_1,i_2] += B6[i_4,i_2] * T2[a_1,a_2,i_1,i_4]
        @notensor B6 = nothing
        GC.gc()
        B7[a_4,a_2] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_2,a_3,i_4,i_3])
        R2u[a_1,a_2,i_1,i_2] += - B7[a_4,a_2] * T2[a_1,a_4,i_1,i_2]
        @notensor B7 = nothing
        GC.gc()
        B8[i_4,a_3,a_1,i_2] := 0.5*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_4,i_3,i_2])
        R2u[a_1,a_2,i_1,i_2] +=  +B8[i_4,a_3,a_1,i_2] * T2[a_2,a_3,i_4,i_1]
        @notensor B8 = nothing
        GC.gc()
        B9[i_3,i_4,i_1,i_2] := 0.5* (g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_1,i_2])
        R2u[a_1,a_2,i_1,i_2] += + B9[i_3,i_4,i_1,i_2] * T2[a_1,a_2,i_3,i_4]
        @notensor B9 = nothing
        @notensor g_oovv = nothing
        GC.gc()
        R2[a,b,i,j] += R2u[a,b,i,j] + R2u[b,a,j,i]
        @notensor R2u = nothing
        GC.gc()
    end
    return R2
end

# function calculate_ECCD()
#     nv = deserialize("nv.jlbin")
#     no = deserialize("no.jlbin")
#     T2 = TensorMap(deserialize("T2_vvoo.jlbin"),ℝ^nv ⊗ ℝ^nv,ℝ^no ⊗ ℝ^no)
#     g_oovv = TensorMap(deserialize("g_oovv.jlbin"),ℝ^no ⊗ ℝ^no,ℝ^nv ⊗ ℝ^nv)
#     ECCSD::Float64 = 0.0
#     @tensor begin
#         ECCSD = 2*g_oovv[i_1,i_2,a_1,a_2] * T2[a_1,a_2,i_1,i_2] - g_oovv[i_1,i_2,a_1,a_2] * T2[a_1,a_2,i_2,i_1]
#     end
#     return ECCSD
# end

function calculate_ECCD()
    nv = deserialize("nv.jlbin")
    no = deserialize("no.jlbin")
    Z::Float64 = 0.0
    g_oovv = TensorMap(deserialize("g_oovv.jlbin"),ℝ^no ⊗ ℝ^no,ℝ^nv ⊗ ℝ^nv)
    T2_vvoo = TensorMap(deserialize("T2_vvoo.jlbin"),ℝ^nv ⊗ ℝ^nv,ℝ^no ⊗ ℝ^no)
    @tensor Z += 2*g_oovv[i_1, i_2, a_1, a_2]*T2_vvoo[a_1, a_2, i_1, i_2]
    T2_vvoo = nothing
    g_oovv = nothing
    g_oovv = TensorMap(deserialize("g_oovv.jlbin"),ℝ^no ⊗ ℝ^no,ℝ^nv ⊗ ℝ^nv)
    T2_vvoo = TensorMap(deserialize("T2_vvoo.jlbin"),ℝ^nv ⊗ ℝ^nv,ℝ^no ⊗ ℝ^no)
    @tensor Z += -1*g_oovv[i_1, i_2, a_1, a_2]*T2_vvoo[a_1, a_2, i_2, i_1]
    T2_vvoo = nothing
    g_oovv = nothing
    return Z
end

function update_T2(T2,R2,Scaled_R2)
    nv = deserialize("nv.jlbin")
    no = deserialize("no.jlbin")
    fvv = TensorMap(deserialize("fvv.jlbin"),ℝ^nv,ℝ^nv)
    foo = TensorMap(deserialize("foo.jlbin"),ℝ^no,ℝ^no)
    shiftp = 0.20
    for a in 1:nv, b in 1:nv, i in 1:no, j in 1:no
        Scaled_R2[a,b,i,j] = R2[a,b,i,j]/(fvv[a,a] + fvv[b,b]-foo[i,i] - foo[j,j]+shiftp)
    end
    for a in 1:nv, b in 1:nv, i in 1:no, j in 1:no
        T2[a,b,i,j] = T2[a,b,i,j] - Scaled_R2[a,b,i,j]
    end
    return T2
end
function check_convergence(R2,normtol,e_old,e_new,etol)
    r2norm = sqrt(abs(dot(R2,R2)))
    if(r2norm < normtol && abs(e_new[1]-e_old[1])<etol)
        return true,r2norm
    else
        return false,r2norm
    end
        
end

function calculate_R_iter(R_iter,R2)
    nv = deserialize("nv.jlbin")
    no = deserialize("no.jlbin")
    fvv = deserialize("fvv.jlbin")
    foo = deserialize("foo.jlbin")
    shiftp = 0.20
    for a in 1:nv, b in 1:nv, i in 1:no, j in 1:no
        R_iter[a,b,i,j] = R2[a,b,i,j]/(fvv[a,a] + fvv[b,b]-foo[i,i] - foo[j,j]+shiftp)
    end
    return R_iter
end


function just_show_Bmatrix(R_iter_storage,p,T2_storage,R2_storage)
    nv = deserialize("nv.jlbin")
    no = deserialize("no.jlbin")
    B = zeros(p+1,p+1)
    Bpp = dot(R2_storage[p],R2_storage[p])
    for i in 1:p
        for j in 1:i
            # B[i,j] = dot(R_iter_storage[i],R_iter_storage[j])/Bpp
            B[i,j] = dot(R2_storage[i],R2_storage[j])
            B[j,i] = B[i,j]
        end
    end
    # display(@views B[1:p,1:p])
    B[p+1, 1:p] .= -1
    B[1:p, p+1] .= -1
    B[p+1, p+1] = 0
    Id = zeros(p+1)
    Id[p+1]= -1
    C = B\Id
    pop!(C) # As C[p+1] is the Lagrange multiplier
    # t = zeros(size(R_iter_storage[1]))
    t = TensorMap(zeros(Float64,nv,nv,no,no),ℝ^nv ⊗ ℝ^nv , ℝ^no ⊗ ℝ^no)
    for i in 1:p
        t = t + C[i]*T2_storage[i]
    end
    # display(B)
    # display(C)
    # return (t)
end

function PerformDiis(R_iter_storage,p,T2_storage,R2_storage)
    nv = deserialize("nv.jlbin")
    no = deserialize("no.jlbin")
    B = zeros(p+1,p+1)
    Bpp = dot(R2_storage[p],R2_storage[p])
    for i in 1:p
        for j in 1:i
            # B[i,j] = dot(R_iter_storage[i],R_iter_storage[j])/Bpp
            B[i,j] = dot(R2_storage[i],R2_storage[j])
            B[j,i] = B[i,j]
        end
    end
    # display(@views B[1:p,1:p])
    B[p+1, 1:p] .= -1
    B[1:p, p+1] .= -1
    B[p+1, p+1] = 0
    Id = zeros(p+1)
    Id[p+1]= -1
    C = B\Id
    pop!(C) # As C[p+1] is the Lagrange multiplier
    # t = zeros(size(R_iter_storage[1]))
    t = TensorMap(zeros(Float64,nv,nv,no,no),ℝ^nv ⊗ ℝ^nv , ℝ^no ⊗ ℝ^no)
    for i in 1:p
        t = t + C[i]*T2_storage[i]
    end
    # display(B)
    # display(C)
    return (t)
end

function calculate_residual_memeff(T2)
    nv = deserialize("nv.jlbin")
    no = deserialize("no.jlbin")
    fvv = TensorMap(deserialize("fvv.jlbin"),ℝ^nv,ℝ^nv)
    foo = TensorMap(deserialize("foo.jlbin"),ℝ^no,ℝ^no)
    g_vvoo = TensorMap(deserialize("g_vvoo.jlbin"),ℝ^nv ⊗ ℝ^nv,ℝ^no ⊗ ℝ^no)
    g_voov = TensorMap(deserialize("g_voov.jlbin"),ℝ^nv ⊗ ℝ^no,ℝ^no ⊗ ℝ^nv) 
    g_vovo = TensorMap(deserialize("g_vovo.jlbin"),ℝ^nv ⊗ ℝ^no,ℝ^nv ⊗ ℝ^no)
    g_oovv = TensorMap(deserialize("g_oovv.jlbin"),ℝ^no ⊗ ℝ^no,ℝ^nv ⊗ ℝ^nv)
    g_oooo = TensorMap(deserialize("g_oooo.jlbin"),ℝ^no ⊗ ℝ^no,ℝ^no ⊗ ℝ^no)
    g_vvvv = TensorMap(deserialize("g_vvvv.jlbin"),ℝ^nv ⊗ ℝ^nv,ℝ^nv ⊗ ℝ^nv)
    R2u = TensorMap(zeros(Float64,nv,nv,no,no),ℝ^nv ⊗ ℝ^nv,ℝ^no ⊗ ℝ^no)
    R2 = TensorMap(zeros(Float64,nv,nv,no,no),ℝ^nv ⊗ ℝ^nv,ℝ^no ⊗ ℝ^no)
    @tensor begin
        Trm1[a_1,a_2,i_1,i_2] := 0.5* g_vvoo[a_1,a_2,i_1,i_2]
        R2u[a_1,a_2,i_1,i_2] += Trm1[a_1,a_2,i_1,i_2]
        @notensor Trm1 = nothing
        @notensor g_vvoo = nothing
        Trm2[a_1,a_2,i_1,i_2] := - g_voov[a_1,i_3,i_1,a_3] * T2[a_2,a_3,i_3,i_2]
        R2u[a_1,a_2,i_1,i_2] += Trm2[a_1,a_2,i_1,i_2]
        @notensor Trm2 = nothing
        Trm3[a_1,a_2,i_1,i_2] := - g_vovo[a_2,i_3,a_3,i_2] * T2[a_1,a_3,i_1,i_3]
        R2u[a_1,a_2,i_1,i_2] += Trm3[a_1,a_2,i_1,i_2]
        @notensor Trm3 = nothing
        Trm4[a_1,a_2,i_1,i_2] := - g_vovo[a_2,i_3,a_3,i_1] * T2[a_1,a_3,i_3,i_2]
        R2u[a_1,a_2,i_1,i_2] += Trm4[a_1,a_2,i_1,i_2]
        @notensor Trm4 = nothing
        @notensor g_vovo = nothing
        Trm5[a_1,a_2,i_1,i_2] := 2*g_voov[a_1,i_3,i_1,a_3] * T2[a_2,a_3,i_2,i_3]
        R2u[a_1,a_2,i_1,i_2] += Trm5[a_1,a_2,i_1,i_2]
        @notensor Trm5 = nothing
        @notensor g_voov = nothing
        Trm6[a_1,a_2,i_1,i_2] := 0.5*g_oooo[i_3,i_4,i_1,i_2] * T2[a_1,a_2,i_3,i_4]
        R2u[a_1,a_2,i_1,i_2] += Trm6[a_1,a_2,i_1,i_2]
        @notensor Trm6 = nothing
        @notensor g_oooo = nothing
        Trm7[a_1,a_2,i_1,i_2] := fvv[a_2,a_3] * T2[a_1,a_3,i_1,i_2]
        R2u[a_1,a_2,i_1,i_2] += Trm7[a_1,a_2,i_1,i_2]
        @notensor Trm7 = nothing
        @notensor fvv = nothing
        Trm8[a_1,a_2,i_1,i_2] := + 0.5*g_vvvv[a_1,a_2,a_3,a_4] * T2[a_3,a_4,i_1,i_2]
        R2u[a_1,a_2,i_1,i_2] += Trm8[a_1,a_2,i_1,i_2]
        @notensor Trm8 = nothing
        @notensor g_vvvv = nothing
        Trm9[a_1,a_2,i_1,i_2] := - foo[i_3,i_2] * T2[a_1,a_2,i_1,i_3]
        R2u[a_1,a_2,i_1,i_2] += Trm9[a_1,a_2,i_1,i_2]
        @notensor Trm9 = nothing
        @notensor foo = nothing
        B1[i_4,a_4,a_1,i_1] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_3,i_1,i_3])
        R2u[a_1,a_2,i_1,i_2] +=  B1[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_2,i_4]- B1[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_4,i_2]
        @notensor B1 = nothing
        B2[i_4,a_4,a_1,i_1] := 0.5*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_3,i_3,i_1])
        R2u[a_1,a_2,i_1,i_2] += B2[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_4,i_2]
        @notensor B2 = nothing
        B3[i_4,i_2] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_3,i_2])
        R2u[a_1,a_2,i_1,i_2] +=  -B3[i_4,i_2] * T2[a_1,a_2,i_1,i_4]
        @notensor B3 = nothing
        B4[i_4,a_3,a_1,i_1] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_4,i_1,i_3])
        R2u[a_1,a_2,i_1,i_2] += -B4[i_4,a_3,a_1,i_1] * T2[a_2,a_3,i_2,i_4] + B4[i_4,a_3,a_1,i_1] * T2[a_2,a_3,i_4,i_2]
        @notensor B4 = nothing
        B5[a_3,a_2] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_2,a_4,i_4,i_3])
        R2u[a_1,a_2,i_1,i_2] += B5[a_3,a_2] * T2[a_1,a_3,i_1,i_2]
        @notensor B5 = nothing
        B6[i_4,i_2] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_2,i_3])
        R2u[a_1,a_2,i_1,i_2] += B6[i_4,i_2] * T2[a_1,a_2,i_1,i_4]
        @notensor B6 = nothing
        B7[a_4,a_2] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_2,a_3,i_4,i_3])
        R2u[a_1,a_2,i_1,i_2] += - B7[a_4,a_2] * T2[a_1,a_4,i_1,i_2]
        @notensor B7 = nothing
        B8[i_4,a_3,a_1,i_2] := 0.5*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_4,i_3,i_2])
        R2u[a_1,a_2,i_1,i_2] +=  +B8[i_4,a_3,a_1,i_2] * T2[a_2,a_3,i_4,i_1]
        @notensor B8 = nothing
        B9[i_3,i_4,i_1,i_2] := 0.5* (g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_1,i_2])
        R2u[a_1,a_2,i_1,i_2] += + B9[i_3,i_4,i_1,i_2] * T2[a_1,a_2,i_3,i_4]
        @notensor B9 = nothing
        @notensor g_oovv = nothing
        R2[a,b,i,j] += R2u[a,b,i,j] + R2u[b,a,j,i]
        @notensor R2u = nothing
    end
    return R2
end



########################## Mem Saving Code June 15 2024 ###########################
# This is written in such a way that the big g_xxxx TensorOperations
# are only read when the subsequent terms need them
# The g_xxxx and the binary contraction intermediates are thrown away
# after the subsequent terms are calculated

function addres_gvvoo(R2u,nv,no)
    g_vvoo = TensorMap(deserialize("g_vvoo.jlbin")::Array{Float64,4},ℝ^nv ⊗ ℝ^nv,ℝ^no ⊗ ℝ^no)
    @tensor Trm1[a_1,a_2,i_1,i_2] := 0.5* g_vvoo[a_1,a_2,i_1,i_2]
    @tensor R2u[a_1,a_2,i_1,i_2] += Trm1[a_1,a_2,i_1,i_2]
    return R2u
end

function addres_gvoov(R2u,nv,no,T2)
    g_voov = TensorMap(deserialize("g_voov.jlbin")::Array{Float64,4},ℝ^nv ⊗ ℝ^no,ℝ^no ⊗ ℝ^nv)
    @tensor Trm2[a_1,a_2,i_1,i_2] := - g_voov[a_1,i_3,i_1,a_3] * T2[a_2,a_3,i_3,i_2]
    @tensor R2u[a_1,a_2,i_1,i_2] += Trm2[a_1,a_2,i_1,i_2]
    @tensor Trm5[a_1,a_2,i_1,i_2] := 2*g_voov[a_1,i_3,i_1,a_3] * T2[a_2,a_3,i_2,i_3]
    @tensor R2u[a_1,a_2,i_1,i_2] += Trm5[a_1,a_2,i_1,i_2]
    return R2u
end

function addres_gvovo(R2u,nv,no,T2)
    g_vovo = TensorMap(deserialize("g_vovo.jlbin")::Array{Float64,4},ℝ^nv ⊗ ℝ^no,ℝ^nv ⊗ ℝ^no)
    @tensor Trm3[a_1,a_2,i_1,i_2] := - g_vovo[a_2,i_3,a_3,i_2] * T2[a_1,a_3,i_1,i_3]
    @tensor R2u[a_1,a_2,i_1,i_2] += Trm3[a_1,a_2,i_1,i_2]
    @tensor Trm4[a_1,a_2,i_1,i_2] := - g_vovo[a_2,i_3,a_3,i_1] * T2[a_1,a_3,i_3,i_2]
    @tensor R2u[a_1,a_2,i_1,i_2] += Trm4[a_1,a_2,i_1,i_2]
    return R2u
end

function addres_goooo(R2u,nv,no,T2)
    g_oooo = TensorMap(deserialize("g_oooo.jlbin")::Array{Float64,4},ℝ^no ⊗ ℝ^no,ℝ^no ⊗ ℝ^no)
    @tensor Trm6[a_1,a_2,i_1,i_2] := 0.5*g_oooo[i_3,i_4,i_1,i_2] * T2[a_1,a_2,i_3,i_4]
    @tensor R2u[a_1,a_2,i_1,i_2] += Trm6[a_1,a_2,i_1,i_2]
    return R2u
end

function addres_gvvvv(R2u,nv,no,T2)
    g_vvvv = TensorMap(deserialize("g_vvvv.jlbin")::Array{Float64,4},ℝ^nv ⊗ ℝ^nv,ℝ^nv ⊗ ℝ^nv)
    @tensor Trm8[a_1,a_2,i_1,i_2] := + 0.5*g_vvvv[a_1,a_2,a_3,a_4] * T2[a_3,a_4,i_1,i_2]
    @tensor R2u[a_1,a_2,i_1,i_2] += Trm8[a_1,a_2,i_1,i_2]
    return R2u
end

function addres_fvv(R2u,nv,no,T2)
    fvv = TensorMap(deserialize("fvv.jlbin")::Array{Float64,2},ℝ^nv,ℝ^nv)
    @tensor Trm7[a_1,a_2,i_1,i_2] := fvv[a_2,a_3] * T2[a_1,a_3,i_1,i_2]
    @tensor R2u[a_1,a_2,i_1,i_2] += Trm7[a_1,a_2,i_1,i_2]
    return R2u
end

function addres_foo(R2u,nv,no,T2)
    foo = TensorMap(deserialize("foo.jlbin")::Array{Float64,2},ℝ^no,ℝ^no)
    @tensor Trm9[a_1,a_2,i_1,i_2] := - foo[i_3,i_2] * T2[a_1,a_2,i_1,i_3]
    @tensor R2u[a_1,a_2,i_1,i_2] += Trm9[a_1,a_2,i_1,i_2]
    return R2u
end

function addres_goovv(R2u,nv,no,T2)
    g_oovv = TensorMap(deserialize("g_oovv.jlbin")::Array{Float64,4},ℝ^no ⊗ ℝ^no,ℝ^nv ⊗ ℝ^nv)
    @tensor B1[i_4,a_4,a_1,i_1] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_3,i_1,i_3])
    @tensor R2u[a_1,a_2,i_1,i_2] +=  B1[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_2,i_4]- B1[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_4,i_2]
    @tensor B2[i_4,a_4,a_1,i_1] := 0.5*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_3,i_3,i_1])
    @tensor R2u[a_1,a_2,i_1,i_2] += B2[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_4,i_2]
    @tensor B3[i_4,i_2] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_3,i_2])
    @tensor R2u[a_1,a_2,i_1,i_2] +=  -B3[i_4,i_2] * T2[a_1,a_2,i_1,i_4]
    @tensor B4[i_4,a_3,a_1,i_1] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_4,i_1,i_3])
    @tensor R2u[a_1,a_2,i_1,i_2] += -B4[i_4,a_3,a_1,i_1] * T2[a_2,a_3,i_2,i_4] + B4[i_4,a_3,a_1,i_1] * T2[a_2,a_3,i_4,i_2]
    @tensor B5[a_3,a_2] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_2,a_4,i_4,i_3])
    @tensor R2u[a_1,a_2,i_1,i_2] += B5[a_3,a_2] * T2[a_1,a_3,i_1,i_2]
    @tensor B6[i_4,i_2] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_2,i_3])
    @tensor R2u[a_1,a_2,i_1,i_2] += B6[i_4,i_2] * T2[a_1,a_2,i_1,i_4]
    @tensor B7[a_4,a_2] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_2,a_3,i_4,i_3])
    @tensor R2u[a_1,a_2,i_1,i_2] += - B7[a_4,a_2] * T2[a_1,a_4,i_1,i_2]
    @tensor B8[i_4,a_3,a_1,i_2] := 0.5*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_4,i_3,i_2])
    @tensor R2u[a_1,a_2,i_1,i_2] +=  +B8[i_4,a_3,a_1,i_2] * T2[a_2,a_3,i_4,i_1]
    @tensor B9[i_3,i_4,i_1,i_2] := 0.5* (g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_1,i_2])
    @tensor R2u[a_1,a_2,i_1,i_2] += + B9[i_3,i_4,i_1,i_2] * T2[a_1,a_2,i_3,i_4]
end

function addres_goovvb1(R2u,nv,no,T2)
    g_oovv = TensorMap(deserialize("g_oovv.jlbin")::Array{Float64,4},ℝ^no ⊗ ℝ^no,ℝ^nv ⊗ ℝ^nv)
    @tensor B1[i_4,a_4,a_1,i_1] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_3,i_1,i_3])
    @tensor R2u[a_1,a_2,i_1,i_2] +=  B1[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_2,i_4]- B1[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_4,i_2]
    return R2u
end

function addres_goovvb2(R2u,nv,no,T2)
    g_oovv = TensorMap(deserialize("g_oovv.jlbin")::Array{Float64,4},ℝ^no ⊗ ℝ^no,ℝ^nv ⊗ ℝ^nv)
    @tensor B2[i_4,a_4,a_1,i_1] := 0.5*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_3,i_3,i_1])
    @tensor R2u[a_1,a_2,i_1,i_2] += B2[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_4,i_2]
    return R2u
end

function addres_goovvb3(R2u,nv,no,T2)
    g_oovv = TensorMap(deserialize("g_oovv.jlbin")::Array{Float64,4},ℝ^no ⊗ ℝ^no,ℝ^nv ⊗ ℝ^nv)
    @tensor B3[i_4,i_2] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_3,i_2])
    @tensor R2u[a_1,a_2,i_1,i_2] +=  -B3[i_4,i_2] * T2[a_1,a_2,i_1,i_4]
    return R2u
end

function addres_goovvb4(R2u,nv,no,T2)
    g_oovv = TensorMap(deserialize("g_oovv.jlbin")::Array{Float64,4},ℝ^no ⊗ ℝ^no,ℝ^nv ⊗ ℝ^nv)
    @tensor B4[i_4,a_3,a_1,i_1] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_4,i_1,i_3])
    @tensor R2u[a_1,a_2,i_1,i_2] += -B4[i_4,a_3,a_1,i_1] * T2[a_2,a_3,i_2,i_4] + B4[i_4,a_3,a_1,i_1] * T2[a_2,a_3,i_4,i_2]
    return R2u
end

function addres_goovvb5(R2u,nv,no,T2)
    g_oovv = TensorMap(deserialize("g_oovv.jlbin")::Array{Float64,4},ℝ^no ⊗ ℝ^no,ℝ^nv ⊗ ℝ^nv)
    @tensor B5[a_3,a_2] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_2,a_4,i_4,i_3])
    @tensor R2u[a_1,a_2,i_1,i_2] += B5[a_3,a_2] * T2[a_1,a_3,i_1,i_2]
    return R2u
end

function addres_goovvb6(R2u,nv,no,T2)
    g_oovv = TensorMap(deserialize("g_oovv.jlbin")::Array{Float64,4},ℝ^no ⊗ ℝ^no,ℝ^nv ⊗ ℝ^nv)
    @tensor B6[i_4,i_2] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_2,i_3])
    @tensor R2u[a_1,a_2,i_1,i_2] += B6[i_4,i_2] * T2[a_1,a_2,i_1,i_4]
    return R2u
end

function addres_goovvb7(R2u,nv,no,T2)
    g_oovv = TensorMap(deserialize("g_oovv.jlbin")::Array{Float64,4},ℝ^no ⊗ ℝ^no,ℝ^nv ⊗ ℝ^nv)
    @tensor B7[a_4,a_2] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_2,a_3,i_4,i_3])
    @tensor R2u[a_1,a_2,i_1,i_2] += - B7[a_4,a_2] * T2[a_1,a_4,i_1,i_2]
    return R2u
end

function addres_goovvb8(R2u,nv,no,T2)
    g_oovv = TensorMap(deserialize("g_oovv.jlbin")::Array{Float64,4},ℝ^no ⊗ ℝ^no,ℝ^nv ⊗ ℝ^nv)
    @tensor B8[i_4,a_3,a_1,i_2] := 0.5*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_4,i_3,i_2])
    @tensor R2u[a_1,a_2,i_1,i_2] +=  +B8[i_4,a_3,a_1,i_2] * T2[a_2,a_3,i_4,i_1]
    return R2u
end

function addres_goovvb9(R2u,nv,no,T2)
    g_oovv = TensorMap(deserialize("g_oovv.jlbin")::Array{Float64,4},ℝ^no ⊗ ℝ^no,ℝ^nv ⊗ ℝ^nv)
    @tensor B9[i_3,i_4,i_1,i_2] := 0.5* (g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_1,i_2])
    @tensor R2u[a_1,a_2,i_1,i_2] += + B9[i_3,i_4,i_1,i_2] * T2[a_1,a_2,i_3,i_4]
    return R2u
end

function calcresnew()
    # T2 = deserialize("T2_vvoo.jlbin")
    nv = deserialize("nv.jlbin")::Int64
    no = deserialize("no.jlbin")::Int64
    T2 = TensorMap(deserialize("T2_vvoo.jlbin"),ℝ^nv ⊗ ℝ^nv,ℝ^no ⊗ ℝ^no)
    R2u = TensorMap(zeros(Float64,nv,nv,no,no),ℝ^nv ⊗ ℝ^nv,ℝ^no ⊗ ℝ^no)
    R2 = TensorMap(zeros(Float64,nv,nv,no,no),ℝ^nv ⊗ ℝ^nv,ℝ^no ⊗ ℝ^no)
    R2u = addres_gvvoo(R2u,nv,no)
    R2u = addres_gvoov(R2u,nv,no,T2)
    R2u = addres_gvovo(R2u,nv,no,T2)
    R2u = addres_goooo(R2u,nv,no,T2)
    R2u = addres_gvvvv(R2u,nv,no,T2)
    R2u = addres_fvv(R2u,nv,no,T2)
    R2u = addres_foo(R2u,nv,no,T2)
    # R2u = addres_goovv(R2u,nv,no,T2)
    R2u = addres_goovvb1(R2u,nv,no,T2)
    R2u = addres_goovvb2(R2u,nv,no,T2)
    R2u = addres_goovvb3(R2u,nv,no,T2)
    R2u = addres_goovvb4(R2u,nv,no,T2)
    R2u = addres_goovvb5(R2u,nv,no,T2)
    R2u = addres_goovvb6(R2u,nv,no,T2)
    R2u = addres_goovvb7(R2u,nv,no,T2)
    R2u = addres_goovvb8(R2u,nv,no,T2)
    R2u = addres_goovvb9(R2u,nv,no,T2)
    @tensor R2[a,b,i,j] += R2u[a,b,i,j] + R2u[b,a,j,i]
    return R2
end