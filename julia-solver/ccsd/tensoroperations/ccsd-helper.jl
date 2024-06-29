using DelimitedFiles,TensorOperations,LinearAlgebra,MKL, TensorKit, Serialization


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
    hnuc = data[end,1]
    
    
    data = copy(data[1:end-1, :])
    h = Array{Union{Missing, Float64}}(missing, norb, norb);
    g = Array{Union{Missing, Float64}}(missing, norb, norb, norb, norb);
    l = length(data[:,1])
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
    nocc = round(Int,nelec/2)
    nv = norb - nocc
    h = convert(Array{Float64}, h)
    g = convert(Array{Float64}, g)
    K = zeros(nv,nv,nocc,nocc)
    for a in 1:nv, b in 1:nv, i in 1:nocc, j in 1:nocc
        K[a,b,i,j] = g[nocc+a,i,nocc+b,j]
    end
    H1 = 0.0;
    H2 = 0.0;
    for i in 1:nocc
        H1 = H1+  h[i,i]
        for j in 1:nocc
            H2 = H2 +  2 * g[i,i,j,j] - g[i,j,i,j]
        end
    end
    erhf = 2*H1 + H2 + hnuc
    f = zeros(size(h)) # fock matrix Initialization
    for p in 1:norb , q in 1:norb
        s_pq=0
        for i in 1:nocc
            s_pq += 2*g[p,q,i,i]-g[p,i,i,q]
        end
        f[p,q] = h[p,q] + s_pq
    end
    serialize("h.jlbin", h)
    serialize("g.jlbin", g)
    serialize("hnuc.jlbin", hnuc)
    serialize("norbs.jlbin", norb)
    serialize("nelec.jlbin", nelec)
    serialize("nocc.jlbin", nocc)
    serialize("nv.jlbin", nv)
    serialize("K.jlbin", K)
    serialize("erhf.jlbin", erhf)
    serialize("f.jlbin", f)
end

function initialize_t2_only()
    nv = deserialize("nv.jlbin")
    nocc = deserialize("nocc.jlbin")
################## Amplitude Initialization ######################################
    K = deserialize("K.jlbin")
    f = deserialize("f.jlbin")
    T2::Array{Float64,4} = - K 
    for a in 1:nv, b in 1:nv, i in 1:nocc, j in 1:nocc
        T2[a,b,i,j] = T2[a,b,i,j]/(f[nocc+a,nocc+a] + f[nocc+b,nocc+b] - f[i,i] - f[j,j])
        # if abs(T2[a,b,i,j]) < 10e-8
        #     T2[a,b,i,j] = 0.0
        # end
    end

##################################################################################
    return T2
end

function initialize_t1_only()
    nv = deserialize("nv.jlbin")
    nocc = deserialize("nocc.jlbin")
    f_vo = deserialize("f_vo.jlbin")
    f_vv = deserialize("f_vv.jlbin")
    f_oo = deserialize("f_oo.jlbin")
    T1::Array{Float64,2} = zeros(nv,nocc)
    for a in 1:nv, i in 1:nocc
        T1[a,i] = f_vo[a,i]/(f_vv[a,a] - f_oo[i,i])
    end
    return T1
end



function initialize_cc_variables()
    main("water_dump.fcidump")
    K = deserialize("K.jlbin")
    f = deserialize("f.jlbin")
    g = deserialize("g.jlbin")
    nocc = deserialize("nocc.jlbin")
    nv = deserialize("nv.jlbin")
    g_vvoo = zeros(nv,nv,nocc,nocc)
    g_voov = zeros(nv,nocc,nocc,nv)
    g_vovo = zeros(nv,nocc,nv,nocc)
    g_oovv = zeros(nocc,nocc,nv,nv)
    g_oooo = zeros(nocc,nocc,nocc,nocc)
    g_vvvv = zeros(nv,nv,nv,nv)
    f_vv = zeros(nv,nv)
    f_oo = zeros(nocc,nocc)
    f_vo = zeros(nv,nocc)
    g_eccc = zeros(nv,nocc,nocc,nocc)
    for a in 1:nv, b in 1:nv
        f_vv[a,b] = f[nocc+a,nocc+b]
        
        for i in 1:nocc, j in 1:nocc
            g_vvoo[a,b,i,j] = g[nocc+a,i,nocc+b,j]
            g_voov[a,i,j,b] = g[nocc+a,j,i,nocc+b]
            g_vovo[a,i,b,j] = g[nocc+a,nocc+b,i,j]
            g_oovv[i,j,a,b] = g[i,nocc+a,j,nocc+b]
            f_oo[i,j] = f[i,j]
        end
    end
    for i in 1:nocc, j in 1:nocc, k in 1:nocc, l in 1:nocc
        g_oooo[i,j,k,l] = g[i,k,j,l]
    end
    for a in 1:nv, b in 1:nv, c in 1:nv, d in 1:nv
        g_vvvv[a,b,c,d] = g[nocc+a,nocc+c,nocc+b,nocc+d]
    end
    for a in 1:nv , i in 1:nocc, j in 1:nocc, k in 1:nocc
        g_eccc[a,i,j,k] = g[nocc+a,i,j,k]
    end
    serialize("g_vvoo.jlbin", g_vvoo)
    serialize("g_voov.jlbin", g_voov)
    serialize("g_vovo.jlbin", g_vovo)
    serialize("g_oovv.jlbin", g_oovv)
    serialize("g_oooo.jlbin", g_oooo)
    serialize("g_vvvv.jlbin", g_vvvv)
    serialize("K.jlbin", K)
    serialize("f_vv.jlbin", f_vv)
    serialize("f_oo.jlbin", f_oo)
    serialize("f_vo.jlbin", f_vo)
    serialize("g_eccc.jlbin", g_eccc)
end








function calculate_ECCD(T2)
    g_oovv = deserialize("g_oovv.jlbin")
    ECCSD::Float64 = 0.0
    @tensor begin
        ECCSD = 2*g_oovv[i_1,i_2,a_1,a_2] * T2[a_1,a_2,i_1,i_2] - g_oovv[i_1,i_2,a_1,a_2] * T2[a_1,a_2,i_2,i_1]
    end
    return ECCSD
end

function update_amplitudes(T2,R2,Scaled_R2)
    ## Update Doubles
    f_vv = deserialize("f_vv.jlbin")
    f_oo = deserialize("f_oo.jlbin")
    nv = deserialize("nv.jlbin")
    nocc = deserialize("nocc.jlbin")
    shiftp = 0.20
    for a in 1:nv, b in 1:nv, i in 1:nocc, j in 1:nocc
        Scaled_R2[a,b,i,j] = R2[a,b,i,j]/(f_vv[a,a] + f_vv[b,b]-f_oo[i,i] - f_oo[j,j]+shiftp)
    end
    for a in 1:nv, b in 1:nv, i in 1:nocc, j in 1:nocc
        T2[a,b,i,j] = T2[a,b,i,j] - Scaled_R2[a,b,i,j]
    end
    #Update Singles
    # f_vo = deserialize("f_vo.jlbin")
    # for a in 1:nv, i in 1:nocc
    #     Scaled_R1 = R2[a,i]/(f_vv[a,a] - f_oo[i,i]+shiftp)
    # end
    # T1 .-= Scaled_R1
    
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
    nocc = deserialize("nocc.jlbin")
    f_vv = deserialize("f_vv.jlbin")
    f_oo = deserialize("f_oo.jlbin")
    shiftp = 0.20
    for a in 1:nv, b in 1:nv, i in 1:nocc, j in 1:nocc
        R_iter[a,b,i,j] = R2[a,b,i,j]/(f_vv[a,a] + f_vv[b,b]-f_oo[i,i] - f_oo[j,j]+shiftp)
    end
    return R_iter
end


function just_show_Bmatrix(R_iter_storage,p,T2_storage,R2_storage)
    B = zeros(p+1,p+1)
    Bpp = dot(R_iter_storage[p],R_iter_storage[p])
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
    flag = 0
    s = zeros(size(R_iter_storage[1]))
    rs = zeros(size(R_iter_storage[1]))
    r = zeros(size(R_iter_storage[1]))
    for i in 1:p
        s = s + C[i].*(R_iter_storage[i]+T2_storage[i])   # t⁽ᵏ⁺¹⁾=∑cₖ(t⁽ᵏ⁾+Δt⁽ᵏ⁾)
        rs = rs + C[i].*R_iter_storage[i]
        r = r + C[i].*R2_storage[i]
    end
    errnorm = sqrt(abs(dot(rs,rs)))
    println("Current Norms of Scaled Residuals: ",[sqrt(abs(dot(R_iter_storage[i],R_iter_storage[i]))) for i in 1:p])
    println("Current error vector norm  = $(errnorm)")
    # display(B)
    # display(C)
end

function PerformDiis(R_iter_storage,p,T2_storage,R2_storage)
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
    t = zeros(size(R_iter_storage[1]))
    for i in 1:p
        t = t + C[i].*T2_storage[i]
    end
    # display(B)
    # display(C)
    return (t)
end
function calc_R2()
    nv = deserialize("nv.jlbin")::Int64
    nocc = deserialize("nocc.jlbin")::Int64
    # From here everything is autogenerated
    I_vvoo = zeros(Float64, nv, nv, nocc, nocc)
    g_voov = deserialize("g_voov.jlbin")
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor I_vvoo[a_1, a_2, i_1, i_2] += -1*g_voov[a_1, i_3, i_1, a_3]*T2_vvoo[a_2, a_3, i_3, i_2]
    #Unload T2_vvoo[a_2, a_3, i_3, i_2]
    #Unload g_voov[a_1, i_3, i_1, a_3]
    g_vvoo = deserialize("g_vvoo.jlbin")
    @tensor I_vvoo[a_1, a_2, i_1, i_2] += 1/2*g_vvoo[a_1, a_2, i_1, i_2]
    #Unload g_vvoo[a_1, a_2, i_1, i_2]
    g_vovo = deserialize("g_vovo.jlbin")
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor I_vvoo[a_1, a_2, i_1, i_2] += -1*g_vovo[a_2, i_3, a_3, i_2]*T2_vvoo[a_1, a_3, i_1, i_3]
    #Unload T2_vvoo[a_1, a_3, i_1, i_3]
    #Unload g_vovo[a_2, i_3, a_3, i_2]
    g_vovo = deserialize("g_vovo.jlbin")
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor I_vvoo[a_1, a_2, i_1, i_2] += -1*g_vovo[a_2, i_3, a_3, i_1]*T2_vvoo[a_1, a_3, i_3, i_2]
    #Unload T2_vvoo[a_1, a_3, i_3, i_2]
    #Unload g_vovo[a_2, i_3, a_3, i_1]
    g_voov = deserialize("g_voov.jlbin")
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor I_vvoo[a_1, a_2, i_1, i_2] += 2*g_voov[a_1, i_3, i_1, a_3]*T2_vvoo[a_2, a_3, i_2, i_3]
    #Unload T2_vvoo[a_2, a_3, i_2, i_3]
    #Unload g_voov[a_1, i_3, i_1, a_3]
    g_oooo = deserialize("g_oooo.jlbin")
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor I_vvoo[a_1, a_2, i_1, i_2] += 1/2*g_oooo[i_3, i_4, i_1, i_2]*T2_vvoo[a_1, a_2, i_3, i_4]
    #Unload T2_vvoo[a_1, a_2, i_3, i_4]
    #Unload g_oooo[i_3, i_4, i_1, i_2]
    f_vv = deserialize("f_vv.jlbin")
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor I_vvoo[a_1, a_2, i_1, i_2] += f_vv[a_2, a_3]*T2_vvoo[a_1, a_3, i_1, i_2]
    #Unload T2_vvoo[a_1, a_3, i_1, i_2]
    #Unload f_vv[a_2, a_3]
    g_vvvv = deserialize("g_vvvv.jlbin")
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor I_vvoo[a_1, a_2, i_1, i_2] += 1/2*g_vvvv[a_1, a_2, a_3, a_4]*T2_vvoo[a_3, a_4, i_1, i_2]
    #Unload T2_vvoo[a_3, a_4, i_1, i_2]
    #Unload g_vvvv[a_1, a_2, a_3, a_4]
    f_oo = deserialize("f_oo.jlbin")
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor I_vvoo[a_1, a_2, i_1, i_2] += -1*f_oo[i_3, i_2]*T2_vvoo[a_1, a_2, i_1, i_3]
    #Unload T2_vvoo[a_1, a_2, i_1, i_3]
    #Unload f_oo[i_3, i_2]
    I_ovov = zeros(Float64, nocc, nv, nocc, nv)
    g_oovv = deserialize("g_oovv.jlbin")
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor I_ovov[i_4, a_1, i_1, a_4] += g_oovv[i_3, i_4, a_3, a_4]*T2_vvoo[a_1, a_3, i_1, i_3]
    #Unload T2_vvoo[a_1, a_3, i_1, i_3]
    #Unload g_oovv[i_3, i_4, a_3, a_4]
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor I_vvoo[a_1, a_2, i_1, i_2] += 2*I_ovov[i_4, a_1, i_1, a_4]*T2_vvoo[a_2, a_4, i_2, i_4]
    #Unload T2_vvoo[a_2, a_4, i_2, i_4]
    #Unload I_ovov[i_4, a_1, i_1, a_4]
    I_ovov = zeros(Float64, nocc, nv, nocc, nv)#Load and set to zero julia equivalent

    g_oovv = deserialize("g_oovv.jlbin")
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor I_ovov[i_4, a_1, i_1, a_4] += g_oovv[i_3, i_4, a_3, a_4]*T2_vvoo[a_1, a_3, i_1, i_3]
    #Unload T2_vvoo[a_1, a_3, i_1, i_3]
    #Unload g_oovv[i_3, i_4, a_3, a_4]
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor I_vvoo[a_1, a_2, i_1, i_2] += -2*I_ovov[i_4, a_1, i_1, a_4]*T2_vvoo[a_2, a_4, i_4, i_2]
    #Unload T2_vvoo[a_2, a_4, i_4, i_2]
    #Unload I_ovov[i_4, a_1, i_1, a_4]
    I_ovov = zeros(Float64, nocc, nv, nocc, nv)#Load and set to zero julia equivalent

    g_oovv = deserialize("g_oovv.jlbin")
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor I_ovov[i_4, a_1, i_1, a_4] += g_oovv[i_3, i_4, a_3, a_4]*T2_vvoo[a_1, a_3, i_3, i_1]
    #Unload T2_vvoo[a_1, a_3, i_3, i_1]
    #Unload g_oovv[i_3, i_4, a_3, a_4]
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor I_vvoo[a_1, a_2, i_1, i_2] += 1/2*I_ovov[i_4, a_1, i_1, a_4]*T2_vvoo[a_2, a_4, i_4, i_2]
    #Unload T2_vvoo[a_2, a_4, i_4, i_2]
    #Unload I_ovov[i_4, a_1, i_1, a_4]
    I_oo = zeros(Float64, nocc, nocc)
    g_oovv = deserialize("g_oovv.jlbin")
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor I_oo[i_4, i_2] += g_oovv[i_3, i_4, a_3, a_4]*T2_vvoo[a_3, a_4, i_3, i_2]
    #Unload T2_vvoo[a_3, a_4, i_3, i_2]
    #Unload g_oovv[i_3, i_4, a_3, a_4]
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor I_vvoo[a_1, a_2, i_1, i_2] += -2*I_oo[i_4, i_2]*T2_vvoo[a_1, a_2, i_1, i_4]
    #Unload T2_vvoo[a_1, a_2, i_1, i_4]
    #Unload I_oo[i_4, i_2]
    I_ovov = zeros(Float64, nocc, nv, nocc, nv)#Load and set to zero julia equivalent

    g_oovv = deserialize("g_oovv.jlbin")
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor I_ovov[i_4, a_1, i_1, a_3] += g_oovv[i_3, i_4, a_3, a_4]*T2_vvoo[a_1, a_4, i_1, i_3]
    #Unload T2_vvoo[a_1, a_4, i_1, i_3]
    #Unload g_oovv[i_3, i_4, a_3, a_4]
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor I_vvoo[a_1, a_2, i_1, i_2] += -1*I_ovov[i_4, a_1, i_1, a_3]*T2_vvoo[a_2, a_3, i_2, i_4]
    #Unload T2_vvoo[a_2, a_3, i_2, i_4]
    #Unload I_ovov[i_4, a_1, i_1, a_3]
    I_ovov = zeros(Float64, nocc, nv, nocc, nv)#Load and set to zero julia equivalent

    g_oovv = deserialize("g_oovv.jlbin")
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor I_ovov[i_4, a_1, i_1, a_3] += g_oovv[i_3, i_4, a_3, a_4]*T2_vvoo[a_1, a_4, i_1, i_3]
    #Unload T2_vvoo[a_1, a_4, i_1, i_3]
    #Unload g_oovv[i_3, i_4, a_3, a_4]
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor I_vvoo[a_1, a_2, i_1, i_2] += I_ovov[i_4, a_1, i_1, a_3]*T2_vvoo[a_2, a_3, i_4, i_2]
    #Unload T2_vvoo[a_2, a_3, i_4, i_2]
    #Unload I_ovov[i_4, a_1, i_1, a_3]
    I_vv = zeros(Float64, nv, nv)
    g_oovv = deserialize("g_oovv.jlbin")
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor I_vv[a_2, a_3] += g_oovv[i_3, i_4, a_3, a_4]*T2_vvoo[a_2, a_4, i_4, i_3]
    #Unload T2_vvoo[a_2, a_4, i_4, i_3]
    #Unload g_oovv[i_3, i_4, a_3, a_4]
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor I_vvoo[a_1, a_2, i_1, i_2] += I_vv[a_2, a_3]*T2_vvoo[a_1, a_3, i_1, i_2]
    #Unload T2_vvoo[a_1, a_3, i_1, i_2]
    #Unload I_vv[a_2, a_3]
    I_oo = zeros(Float64, nocc, nocc)#Load and set to zero julia equivalent

    g_oovv = deserialize("g_oovv.jlbin")
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor I_oo[i_4, i_2] += g_oovv[i_3, i_4, a_3, a_4]*T2_vvoo[a_3, a_4, i_2, i_3]
    #Unload T2_vvoo[a_3, a_4, i_2, i_3]
    #Unload g_oovv[i_3, i_4, a_3, a_4]
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor I_vvoo[a_1, a_2, i_1, i_2] += I_oo[i_4, i_2]*T2_vvoo[a_1, a_2, i_1, i_4]
    #Unload T2_vvoo[a_1, a_2, i_1, i_4]
    #Unload I_oo[i_4, i_2]
    I_vv = zeros(Float64, nv, nv)#Load and set to zero julia equivalent

    g_oovv = deserialize("g_oovv.jlbin")
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor I_vv[a_2, a_4] += g_oovv[i_3, i_4, a_3, a_4]*T2_vvoo[a_2, a_3, i_4, i_3]
    #Unload T2_vvoo[a_2, a_3, i_4, i_3]
    #Unload g_oovv[i_3, i_4, a_3, a_4]
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor I_vvoo[a_1, a_2, i_1, i_2] += -2*I_vv[a_2, a_4]*T2_vvoo[a_1, a_4, i_1, i_2]
    #Unload T2_vvoo[a_1, a_4, i_1, i_2]
    #Unload I_vv[a_2, a_4]
    I_ovov = zeros(Float64, nocc, nv, nocc, nv)#Load and set to zero julia equivalent

    g_oovv = deserialize("g_oovv.jlbin")
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor I_ovov[i_4, a_1, i_2, a_3] += g_oovv[i_3, i_4, a_3, a_4]*T2_vvoo[a_1, a_4, i_3, i_2]
    #Unload T2_vvoo[a_1, a_4, i_3, i_2]
    #Unload g_oovv[i_3, i_4, a_3, a_4]
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor I_vvoo[a_1, a_2, i_1, i_2] += 1/2*I_ovov[i_4, a_1, i_2, a_3]*T2_vvoo[a_2, a_3, i_4, i_1]
    #Unload T2_vvoo[a_2, a_3, i_4, i_1]
    #Unload I_ovov[i_4, a_1, i_2, a_3]
    I_oooo = zeros(Float64, nocc, nocc, nocc, nocc)
    g_oovv = deserialize("g_oovv.jlbin")
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor I_oooo[i_3, i_4, i_1, i_2] += g_oovv[i_3, i_4, a_3, a_4]*T2_vvoo[a_3, a_4, i_1, i_2]
    #Unload T2_vvoo[a_3, a_4, i_1, i_2]
    #Unload g_oovv[i_3, i_4, a_3, a_4]
    T2_vvoo = deserialize("T2_vvoo.jlbin")
    @tensor I_vvoo[a_1, a_2, i_1, i_2] += 1/2*I_oooo[i_3, i_4, i_1, i_2]*T2_vvoo[a_1, a_2, i_3, i_4]
    #Unload T2_vvoo[a_1, a_2, i_3, i_4]
    #Unload I_oooo[i_3, i_4, i_1, i_2]

    #Symmetrize, this part is hand written
    @tensor R2[a,b,i,j] := I_vvoo[a,b,i,j] + I_vvoo[b,a,j,i]
return R2
end

# function calculate_residual(T2,R2u,R2)
#     g_vvoo,g_voov,g_vovo,g_oovv,g_oooo,g_vvvv = deserialize("g_vvoo.jlbin"),deserialize("g_voov.jlbin"),deserialize("g_vovo.jlbin"),deserialize("g_oovv.jlbin"),deserialize("g_oooo.jlbin"),deserialize("g_vvvv.jlbin")
#     f_vv , f_oo = deserialize("f_vv.jlbin"),deserialize("f_oo.jlbin")
#     @tensor begin
#         R2u[a_1,a_2,i_1,i_2]= 0.5* g_vvoo[a_1,a_2,i_1,i_2] - g_voov[a_1,i_3,i_1,a_3] * T2[a_2,a_3,i_3,i_2] - g_vovo[a_2,i_3,a_3,i_2] * T2[a_1,a_3,i_1,i_3]- g_vovo[a_2,i_3,a_3,i_1] * T2[a_1,a_3,i_3,i_2]+ 2*g_voov[a_1,i_3,i_1,a_3] * T2[a_2,a_3,i_2,i_3]+ 0.5*g_oooo[i_3,i_4,i_1,i_2] * T2[a_1,a_2,i_3,i_4]+ f_vv[a_2,a_3] * T2[a_1,a_3,i_1,i_2]+ 0.5*g_vvvv[a_1,a_2,a_3,a_4] * T2[a_3,a_4,i_1,i_2]- f_oo[i_3,i_2] * T2[a_1,a_2,i_1,i_3]+ 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_3,i_1,i_3]) * T2[a_2,a_4,i_2,i_4]- 2 *(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_3,i_1,i_3]) * T2[a_2,a_4,i_4,i_2]+ 0.5*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_3,i_3,i_1]) * T2[a_2,a_4,i_4,i_2]- 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_3,i_2]) * T2[a_1,a_2,i_1,i_4]- 1*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_4,i_1,i_3]) * T2[a_2,a_3,i_2,i_4]+ (g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_4,i_1,i_3]) * T2[a_2,a_3,i_4,i_2]+ (g_oovv[i_3,i_4,a_3,a_4] * T2[a_2,a_4,i_4,i_3]) * T2[a_1,a_3,i_1,i_2]+ (g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_2,i_3]) * T2[a_1,a_2,i_1,i_4]- 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_2,a_3,i_4,i_3]) * T2[a_1,a_4,i_1,i_2]+ 0.5*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_4,i_3,i_2]) * T2[a_2,a_3,i_4,i_1]+ 0.5* (g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_1,i_2]) * T2[a_1,a_2,i_3,i_4]
#         R2[a,b,i,j] = R2u[a,b,i,j] + R2u[b,a,j,i]
#      end
#      return R2
# end

# function calculate_residual_memeff_gc(T2)
#     g_vvoo,g_voov,g_vovo,g_oovv,g_oooo,g_vvvv = deserialize("g_vvoo.jlbin"),deserialize("g_voov.jlbin"),deserialize("g_vovo.jlbin"),deserialize("g_oovv.jlbin"),deserialize("g_oooo.jlbin"),deserialize("g_vvvv.jlbin")
#     f_vv , f_oo = deserialize("f_vv.jlbin"),deserialize("f_oo.jlbin")
#     nv = deserialize("nv.jlbin")
#     nocc = deserialize("nocc.jlbin")
#     R2u = zeros(Float64,nv,nv,nocc,nocc)
#     R2 = zeros(Float64,nv,nv,nocc,nocc)
#     @tensor begin
#         Trm1[a_1,a_2,i_1,i_2] := 0.5* g_vvoo[a_1,a_2,i_1,i_2]
#         R2u[a_1,a_2,i_1,i_2] += Trm1[a_1,a_2,i_1,i_2]
#         @notensor Trm1 = nothing
#         @notensor g_vvoo = nothing
#         GC.gc()
#         Trm2[a_1,a_2,i_1,i_2] := - g_voov[a_1,i_3,i_1,a_3] * T2[a_2,a_3,i_3,i_2]
#         R2u[a_1,a_2,i_1,i_2] += Trm2[a_1,a_2,i_1,i_2]
#         @notensor Trm2 = nothing
#         GC.gc()
#         Trm3[a_1,a_2,i_1,i_2] := - g_vovo[a_2,i_3,a_3,i_2] * T2[a_1,a_3,i_1,i_3]
#         R2u[a_1,a_2,i_1,i_2] += Trm3[a_1,a_2,i_1,i_2]
#         @notensor Trm3 = nothing
#         GC.gc()
#         Trm4[a_1,a_2,i_1,i_2] := - g_vovo[a_2,i_3,a_3,i_1] * T2[a_1,a_3,i_3,i_2]
#         R2u[a_1,a_2,i_1,i_2] += Trm4[a_1,a_2,i_1,i_2]
#         @notensor Trm4 = nothing
#         @notensor g_vovo = nothing
#         GC.gc()
#         Trm5[a_1,a_2,i_1,i_2] := 2*g_voov[a_1,i_3,i_1,a_3] * T2[a_2,a_3,i_2,i_3]
#         R2u[a_1,a_2,i_1,i_2] += Trm5[a_1,a_2,i_1,i_2]
#         @notensor Trm5 = nothing
#         @notensor g_voov = nothing
#         GC.gc()
#         Trm6[a_1,a_2,i_1,i_2] := 0.5*g_oooo[i_3,i_4,i_1,i_2] * T2[a_1,a_2,i_3,i_4]
#         R2u[a_1,a_2,i_1,i_2] += Trm6[a_1,a_2,i_1,i_2]
#         @notensor Trm6 = nothing
#         @notensor g_oooo = nothing
#         GC.gc()
#         Trm7[a_1,a_2,i_1,i_2] := f_vv[a_2,a_3] * T2[a_1,a_3,i_1,i_2]
#         R2u[a_1,a_2,i_1,i_2] += Trm7[a_1,a_2,i_1,i_2]
#         @notensor Trm7 = nothing
#         @notensor f_vv = nothing
#         GC.gc()
#         Trm8[a_1,a_2,i_1,i_2] := + 0.5*g_vvvv[a_1,a_2,a_3,a_4] * T2[a_3,a_4,i_1,i_2]
#         R2u[a_1,a_2,i_1,i_2] += Trm8[a_1,a_2,i_1,i_2]
#         @notensor Trm8 = nothing
#         @notensor g_vvvv = nothing
#         GC.gc()
#         Trm9[a_1,a_2,i_1,i_2] := - f_oo[i_3,i_2] * T2[a_1,a_2,i_1,i_3]
#         R2u[a_1,a_2,i_1,i_2] += Trm9[a_1,a_2,i_1,i_2]
#         @notensor Trm9 = nothing
#         @notensor f_oo = nothing
#         GC.gc()
#         B1[i_4,a_4,a_1,i_1] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_3,i_1,i_3])
#         R2u[a_1,a_2,i_1,i_2] +=  B1[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_2,i_4]- B1[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_4,i_2]
#         @notensor B1 = nothing
#         GC.gc()
#         B2[i_4,a_4,a_1,i_1] := 0.5*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_3,i_3,i_1])
#         R2u[a_1,a_2,i_1,i_2] += B2[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_4,i_2]
#         @notensor B2 = nothing
#         GC.gc()
#         B3[i_4,i_2] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_3,i_2])
#         R2u[a_1,a_2,i_1,i_2] +=  -B3[i_4,i_2] * T2[a_1,a_2,i_1,i_4]
#         @notensor B3 = nothing
#         GC.gc()
#         B4[i_4,a_3,a_1,i_1] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_4,i_1,i_3])
#         R2u[a_1,a_2,i_1,i_2] += -B4[i_4,a_3,a_1,i_1] * T2[a_2,a_3,i_2,i_4] + B4[i_4,a_3,a_1,i_1] * T2[a_2,a_3,i_4,i_2]
#         @notensor B4 = nothing
#         GC.gc()
#         B5[a_3,a_2] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_2,a_4,i_4,i_3])
#         R2u[a_1,a_2,i_1,i_2] += B5[a_3,a_2] * T2[a_1,a_3,i_1,i_2]
#         @notensor B5 = nothing
#         GC.gc()
#         B6[i_4,i_2] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_2,i_3])
#         R2u[a_1,a_2,i_1,i_2] += B6[i_4,i_2] * T2[a_1,a_2,i_1,i_4]
#         @notensor B6 = nothing
#         GC.gc()
#         B7[a_4,a_2] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_2,a_3,i_4,i_3])
#         R2u[a_1,a_2,i_1,i_2] += - B7[a_4,a_2] * T2[a_1,a_4,i_1,i_2]
#         @notensor B7 = nothing
#         GC.gc()
#         B8[i_4,a_3,a_1,i_2] := 0.5*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_4,i_3,i_2])
#         R2u[a_1,a_2,i_1,i_2] +=  +B8[i_4,a_3,a_1,i_2] * T2[a_2,a_3,i_4,i_1]
#         @notensor B8 = nothing
#         GC.gc()
#         B9[i_3,i_4,i_1,i_2] := 0.5* (g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_1,i_2])
#         R2u[a_1,a_2,i_1,i_2] += + B9[i_3,i_4,i_1,i_2] * T2[a_1,a_2,i_3,i_4]
#         @notensor B9 = nothing
#         @notensor g_oovv = nothing
#         GC.gc()
#         R2[a,b,i,j] := R2u[a,b,i,j] + R2u[b,a,j,i]
#         @notensor R2u = nothing
#         GC.gc()
#     end
#     return R2
# end

# function calculate_residual_memeff(T2::Array{Float64,4})
#     # g_vvoo,g_voov,g_vovo,g_oovv,g_oooo,g_vvvv= deserialize("g_vvoo.jlbin"),deserialize("g_voov.jlbin"),deserialize("g_vovo.jlbin"),deserialize("g_oovv.jlbin"),deserialize("g_oooo.jlbin"),deserialize("g_vvvv.jlbin")
#     g_vvoo::Array{Float64,4} = deserialize("g_vvoo.jlbin")
#     g_voov::Array{Float64,4} = deserialize("g_voov.jlbin")
#     g_vovo::Array{Float64,4} = deserialize("g_vovo.jlbin")
#     g_oovv::Array{Float64,4} = deserialize("g_oovv.jlbin")
#     g_oooo::Array{Float64,4} = deserialize("g_oooo.jlbin")
#     g_vvvv::Array{Float64,4} = deserialize("g_vvvv.jlbin")
#     f_vv::Array{Float64,2} , f_oo::Array{Float64,2} = deserialize("f_vv.jlbin"),deserialize("f_oo.jlbin")
#     nv::Int64 = deserialize("nv.jlbin")
#     nocc::Int64 = deserialize("nocc.jlbin")
#     R2u::Array{Float64,4} = zeros(Float64,nv,nv,nocc,nocc)
#     R2::Array{Float64,4} = zeros(Float64,nv,nv,nocc,nocc)
#     @tensor begin
#         Trm1[a_1,a_2,i_1,i_2] := 0.5* g_vvoo[a_1,a_2,i_1,i_2]
#         R2u[a_1,a_2,i_1,i_2] += Trm1[a_1,a_2,i_1,i_2]
#         # @notensor Trm1 = nothing
#         # @notensor g_vvoo = nothing
#         Trm2[a_1,a_2,i_1,i_2] := - g_voov[a_1,i_3,i_1,a_3] * T2[a_2,a_3,i_3,i_2]
#         R2u[a_1,a_2,i_1,i_2] += Trm2[a_1,a_2,i_1,i_2]
#         # @notensor Trm2 = nothing
#         Trm3[a_1,a_2,i_1,i_2] := - g_vovo[a_2,i_3,a_3,i_2] * T2[a_1,a_3,i_1,i_3]
#         R2u[a_1,a_2,i_1,i_2] += Trm3[a_1,a_2,i_1,i_2]
#         # @notensor Trm3 = nothing
#         Trm4[a_1,a_2,i_1,i_2] := - g_vovo[a_2,i_3,a_3,i_1] * T2[a_1,a_3,i_3,i_2]
#         R2u[a_1,a_2,i_1,i_2] += Trm4[a_1,a_2,i_1,i_2]
#         # @notensor Trm4 = nothing
#         # @notensor g_vovo = nothing
#         Trm5[a_1,a_2,i_1,i_2] := 2*g_voov[a_1,i_3,i_1,a_3] * T2[a_2,a_3,i_2,i_3]
#         R2u[a_1,a_2,i_1,i_2] += Trm5[a_1,a_2,i_1,i_2]
#         # @notensor Trm5 = nothing
#         # @notensor g_voov = nothing
#         Trm6[a_1,a_2,i_1,i_2] := 0.5*g_oooo[i_3,i_4,i_1,i_2] * T2[a_1,a_2,i_3,i_4]
#         R2u[a_1,a_2,i_1,i_2] += Trm6[a_1,a_2,i_1,i_2]
#         # @notensor Trm6 = nothing
#         # @notensor g_oooo = nothing
#         Trm7[a_1,a_2,i_1,i_2] := f_vv[a_2,a_3] * T2[a_1,a_3,i_1,i_2]
#         R2u[a_1,a_2,i_1,i_2] += Trm7[a_1,a_2,i_1,i_2]
#         # @notensor Trm7 = nothing
#         # @notensor f_vv = nothing
#         Trm8[a_1,a_2,i_1,i_2] := + 0.5*g_vvvv[a_1,a_2,a_3,a_4] * T2[a_3,a_4,i_1,i_2]
#         R2u[a_1,a_2,i_1,i_2] += Trm8[a_1,a_2,i_1,i_2]
#         # @notensor Trm8 = nothing
#         # @notensor g_vvvv = nothing
#         Trm9[a_1,a_2,i_1,i_2] := - f_oo[i_3,i_2] * T2[a_1,a_2,i_1,i_3]
#         R2u[a_1,a_2,i_1,i_2] += Trm9[a_1,a_2,i_1,i_2]
#         # @notensor Trm9 = nothing
#         # @notensor f_oo = nothing
#         B1[i_4,a_4,a_1,i_1] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_3,i_1,i_3])
#         R2u[a_1,a_2,i_1,i_2] +=  B1[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_2,i_4]- B1[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_4,i_2]
#         # @notensor B1 = nothing
#         B2[i_4,a_4,a_1,i_1] := 0.5*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_3,i_3,i_1])
#         R2u[a_1,a_2,i_1,i_2] += B2[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_4,i_2]
#         # @notensor B2 = nothing
#         B3[i_4,i_2] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_3,i_2])
#         R2u[a_1,a_2,i_1,i_2] +=  -B3[i_4,i_2] * T2[a_1,a_2,i_1,i_4]
#         # @notensor B3 = nothing
#         B4[i_4,a_3,a_1,i_1] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_4,i_1,i_3])
#         R2u[a_1,a_2,i_1,i_2] += -B4[i_4,a_3,a_1,i_1] * T2[a_2,a_3,i_2,i_4] + B4[i_4,a_3,a_1,i_1] * T2[a_2,a_3,i_4,i_2]
#         # @notensor B4 = nothing
#         B5[a_3,a_2] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_2,a_4,i_4,i_3])
#         R2u[a_1,a_2,i_1,i_2] += B5[a_3,a_2] * T2[a_1,a_3,i_1,i_2]
#         # @notensor B5 = nothing
#         B6[i_4,i_2] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_2,i_3])
#         R2u[a_1,a_2,i_1,i_2] += B6[i_4,i_2] * T2[a_1,a_2,i_1,i_4]
#         # @notensor B6 = nothing
#         B7[a_4,a_2] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_2,a_3,i_4,i_3])
#         R2u[a_1,a_2,i_1,i_2] += - B7[a_4,a_2] * T2[a_1,a_4,i_1,i_2]
#         # @notensor B7 = nothing
#         B8[i_4,a_3,a_1,i_2] := 0.5*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_4,i_3,i_2])
#         R2u[a_1,a_2,i_1,i_2] +=  +B8[i_4,a_3,a_1,i_2] * T2[a_2,a_3,i_4,i_1]
#         # @notensor B8 = nothing
#         B9[i_3,i_4,i_1,i_2] := 0.5* (g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_1,i_2])
#         R2u[a_1,a_2,i_1,i_2] += + B9[i_3,i_4,i_1,i_2] * T2[a_1,a_2,i_3,i_4]
#         # @notensor B9 = nothing
#         # @notensor g_oovv = nothing
#         R2[a,b,i,j] := R2u[a,b,i,j] + R2u[b,a,j,i]
#         # @notensor R2u = nothing
#     end
#     return R2
# end

########################## Mem Saving Code June 15 2024 ###########################
# This is written in such a way that the big g_xxxx TensorOperations
# are only read when the subsequent terms need them
# The g_xxxx and the binary contraction intermediates are thrown away
# after the subsequent terms are calculated

# function addres_gvvoo(R2u,nv,nocc)
#     g_vvoo = deserialize("g_vvoo.jlbin")
#     @tensor Trm1[a_1,a_2,i_1,i_2] := 0.5* g_vvoo[a_1,a_2,i_1,i_2]
#     @tensor R2u[a_1,a_2,i_1,i_2] += Trm1[a_1,a_2,i_1,i_2]
#     return R2u
# end

# function addres_gvoov(R2u,nv,nocc,T2)
#     g_voov = deserialize("g_voov.jlbin")
#     @tensor Trm2[a_1,a_2,i_1,i_2] := - g_voov[a_1,i_3,i_1,a_3] * T2[a_2,a_3,i_3,i_2]
#     @tensor R2u[a_1,a_2,i_1,i_2] += Trm2[a_1,a_2,i_1,i_2]
#     @tensor Trm5[a_1,a_2,i_1,i_2] := 2*g_voov[a_1,i_3,i_1,a_3] * T2[a_2,a_3,i_2,i_3]
#     @tensor R2u[a_1,a_2,i_1,i_2] += Trm5[a_1,a_2,i_1,i_2]
#     return R2u
# end

# function addres_gvovo(R2u,nv,nocc,T2)
#     g_vovo = deserialize("g_vovo.jlbin")
#     @tensor Trm3[a_1,a_2,i_1,i_2] := - g_vovo[a_2,i_3,a_3,i_2] * T2[a_1,a_3,i_1,i_3]
#     @tensor R2u[a_1,a_2,i_1,i_2] += Trm3[a_1,a_2,i_1,i_2]
#     @tensor Trm4[a_1,a_2,i_1,i_2] := - g_vovo[a_2,i_3,a_3,i_1] * T2[a_1,a_3,i_3,i_2]
#     @tensor R2u[a_1,a_2,i_1,i_2] += Trm4[a_1,a_2,i_1,i_2]
#     return R2u
# end

# function addres_goooo(R2u,nv,nocc,T2)
#     g_oooo = deserialize("g_oooo.jlbin")
#     @tensor Trm6[a_1,a_2,i_1,i_2] := 0.5*g_oooo[i_3,i_4,i_1,i_2] * T2[a_1,a_2,i_3,i_4]
#     @tensor R2u[a_1,a_2,i_1,i_2] += Trm6[a_1,a_2,i_1,i_2]
#     return R2u
# end

# function addres_gvvvv(R2u,nv,nocc,T2)
#     g_vvvv = deserialize("g_vvvv.jlbin")
#     @tensor Trm8[a_1,a_2,i_1,i_2] := + 0.5*g_vvvv[a_1,a_2,a_3,a_4] * T2[a_3,a_4,i_1,i_2]
#     @tensor R2u[a_1,a_2,i_1,i_2] += Trm8[a_1,a_2,i_1,i_2]
#     return R2u
# end

# function addres_f_vv(R2u,nv,nocc,T2)
#     f_vv = deserialize("f_vv.jlbin")
#     @tensor Trm7[a_1,a_2,i_1,i_2] := f_vv[a_2,a_3] * T2[a_1,a_3,i_1,i_2]
#     @tensor R2u[a_1,a_2,i_1,i_2] += Trm7[a_1,a_2,i_1,i_2]
#     return R2u
# end

# function addres_f_oo(R2u,nv,nocc,T2)
#     f_oo = deserialize("f_oo.jlbin")
#     @tensor Trm9[a_1,a_2,i_1,i_2] := - f_oo[i_3,i_2] * T2[a_1,a_2,i_1,i_3]
#     @tensor R2u[a_1,a_2,i_1,i_2] += Trm9[a_1,a_2,i_1,i_2]
#     return R2u
# end

# function addres_goovv(R2u,nv,nocc,T2)
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

# function addres_goovvb1(R2u,nv,nocc,T2)
#     g_oovv = deserialize("g_oovv.jlbin")
#     @tensor B1[i_4,a_4,a_1,i_1] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_3,i_1,i_3])
#     @tensor R2u[a_1,a_2,i_1,i_2] +=  B1[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_2,i_4]- B1[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_4,i_2]
#     return R2u
# end

# function addres_goovvb2(R2u,nv,nocc,T2)
#     g_oovv = deserialize("g_oovv.jlbin")
#     @tensor B2[i_4,a_4,a_1,i_1] := 0.5*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_3,i_3,i_1])
#     @tensor R2u[a_1,a_2,i_1,i_2] += B2[i_4,a_4,a_1,i_1] * T2[a_2,a_4,i_4,i_2]
#     return R2u
# end

# function addres_goovvb3(R2u,nv,nocc,T2)
#     g_oovv = deserialize("g_oovv.jlbin")
#     @tensor B3[i_4,i_2] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_3,i_2])
#     @tensor R2u[a_1,a_2,i_1,i_2] +=  -B3[i_4,i_2] * T2[a_1,a_2,i_1,i_4]
#     return R2u
# end

# function addres_goovvb4(R2u,nv,nocc,T2)
#     g_oovv = deserialize("g_oovv.jlbin")
#     @tensor B4[i_4,a_3,a_1,i_1] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_4,i_1,i_3])
#     @tensor R2u[a_1,a_2,i_1,i_2] += -B4[i_4,a_3,a_1,i_1] * T2[a_2,a_3,i_2,i_4] + B4[i_4,a_3,a_1,i_1] * T2[a_2,a_3,i_4,i_2]
#     return R2u
# end

# function addres_goovvb5(R2u,nv,nocc,T2)
#     g_oovv = deserialize("g_oovv.jlbin")
#     @tensor B5[a_3,a_2] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_2,a_4,i_4,i_3])
#     @tensor R2u[a_1,a_2,i_1,i_2] += B5[a_3,a_2] * T2[a_1,a_3,i_1,i_2]
#     return R2u
# end

# function addres_goovvb6(R2u,nv,nocc,T2)
#     g_oovv = deserialize("g_oovv.jlbin")
#     @tensor B6[i_4,i_2] := (g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_2,i_3])
#     @tensor R2u[a_1,a_2,i_1,i_2] += B6[i_4,i_2] * T2[a_1,a_2,i_1,i_4]
#     return R2u
# end

# function addres_goovvb7(R2u,nv,nocc,T2)
#     g_oovv = deserialize("g_oovv.jlbin")
#     @tensor B7[a_4,a_2] := 2*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_2,a_3,i_4,i_3])
#     @tensor R2u[a_1,a_2,i_1,i_2] += - B7[a_4,a_2] * T2[a_1,a_4,i_1,i_2]
#     return R2u
# end

# function addres_goovvb8(R2u,nv,nocc,T2)
#     g_oovv = deserialize("g_oovv.jlbin")
#     @tensor B8[i_4,a_3,a_1,i_2] := 0.5*(g_oovv[i_3,i_4,a_3,a_4] * T2[a_1,a_4,i_3,i_2])
#     @tensor R2u[a_1,a_2,i_1,i_2] +=  +B8[i_4,a_3,a_1,i_2] * T2[a_2,a_3,i_4,i_1]
#     return R2u
# end

# function addres_goovvb9(R2u,nv,nocc,T2)
#     g_oovv = deserialize("g_oovv.jlbin")
#     @tensor B9[i_3,i_4,i_1,i_2] := 0.5* (g_oovv[i_3,i_4,a_3,a_4] * T2[a_3,a_4,i_1,i_2])
#     @tensor R2u[a_1,a_2,i_1,i_2] += + B9[i_3,i_4,i_1,i_2] * T2[a_1,a_2,i_3,i_4]
#     return R2u
# end


# function calcresnew()
#     T2 = deserialize("T2_vvoo.jlbin")
#     nv = deserialize("nv.jlbin")
#     nocc = deserialize("nocc.jlbin")
#     R2u = zeros(Float64,nv,nv,nocc,nocc)
#     R2 =  zeros(Float64,nv,nv,nocc,nocc)
#     R2u = addres_gvvoo(R2u,nv,nocc)
#     R2u = addres_gvoov(R2u,nv,nocc,T2)
#     R2u = addres_gvovo(R2u,nv,nocc,T2)
#     R2u = addres_goooo(R2u,nv,nocc,T2)
#     R2u = addres_gvvvv(R2u,nv,nocc,T2)
#     R2u = addres_f_vv(R2u,nv,nocc,T2)
#     R2u = addres_f_oo(R2u,nv,nocc,T2)
#     # R2u = addres_goovv(R2u,nv,nocc,T2)  #In this all the BX terms are still in memory at once
#     R2u = addres_goovvb1(R2u,nv,nocc,T2)
#     R2u = addres_goovvb2(R2u,nv,nocc,T2)
#     R2u = addres_goovvb3(R2u,nv,nocc,T2)
#     R2u = addres_goovvb4(R2u,nv,nocc,T2)
#     R2u = addres_goovvb5(R2u,nv,nocc,T2)
#     R2u = addres_goovvb6(R2u,nv,nocc,T2)
#     R2u = addres_goovvb7(R2u,nv,nocc,T2)
#     R2u = addres_goovvb8(R2u,nv,nocc,T2)
#     R2u = addres_goovvb9(R2u,nv,nocc,T2)
#     @tensor R2[a,b,i,j] += R2u[a,b,i,j] + R2u[b,a,j,i]
#     return R2
# end
