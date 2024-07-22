using Printf,Plots
include("ccd-helper.jl")
function ccd_by_hand(maxitr)
    initialize_cc_variables()
    nv::Int64   = deserialize("nv.jlbin")
    no::Int64 = deserialize("no.jlbin")
    a_1= Index(nv, "a_1")
    a_2= Index(nv, "a_2")
    i_1= Index(no, "i_1")
    i_2= Index(no, "i_2")
    erhf::Float64 = deserialize("erhf.jlbin")
    R2 = ITensor(zeros(Float64,nv,nv,no,no),a_1,a_2,i_1,i_2)
    R_iter = ITensor(zeros(Float64,nv,nv,no,no),a_1,a_2,i_1,i_2)
    Scaled_R2 = ITensor(zeros(Float64,nv,nv,no,no),a_1,a_2,i_1,i_2)
    T2_old = ITensor(initialize_t2_only(),a_1,a_2,i_1,i_2)
    normtol=1.0e-8
    serialize("T2_vvoo.jlbin",Array(T2_old,a_1,a_2,i_1,i_2))
    e_new = calculate_ECCD()
    e_old = copy(0.0)
    earr = []
    etol = 1.0e-10
    p_max =  6# Maximum number of previous iterations to store for DIIS
    p_min = 2 # DIIS will start after this many iterations
    R_iter_storage = Array{typeof(R2)}(undef, 0)
    T2_storage = Array{typeof(T2_old)}(undef, 0)
    R2_storage = Array{typeof(R2)}(undef, 0)
    push!(earr,e_new[1]+erhf)
    println("Starting CCD Iterations with Convergence Criteria as ||R₂||<$normtol , ΔE<$etol and max iterations=$maxitr")
    println("-----------------------------------------------------------------------------------------------")
    for i in 1:maxitr
        @printf("\n-------------------------\nStarting Iteration: %d  Current Total Energy: %.10f\n", i, e_new[1]+erhf)
        println("Energy before updating = $(e_old[1]+erhf)")
        serialize("T2_vvoo.jlbin",Array(T2_old,a_1,a_2,i_1,i_2))
        R2 = calcresnew()
        T2_old = update_T2(T2_old,R2,Scaled_R2)
        conv,r2norm = check_convergence(R2,normtol,e_old,e_new,etol)
        if conv
            @printf("CCD Converged in %d iterations, no further updates required,current ||R2||=%.10f and ΔE = %.12f\n\n-------------------------\nE_Total = %.10f\n", i,r2norm,abs(e_old[1]-e_new[1]),e_new[1]+erhf)
            break
        end
        R_iter = calculate_R_iter(R_iter,R2)
        @printf("CCD Not Converged in %d iterations, current ||R2|| = %.10f and ΔE = %.12f \n", i,r2norm,abs(e_old[1]-e_new[1]))
        if i >= p_min   #DIIS  starts
            println("DIIS is being implemented in iteration $i")
            if length(R_iter_storage)>=p_max  # Adjusted condition to start popping correctly
                popfirst!(R_iter_storage)
                popfirst!(T2_storage)
                popfirst!(R2_storage)
            end
            push!(R_iter_storage, copy(R_iter))
            push!(T2_storage, copy(T2_old))
            push!(R2_storage, copy(R2))
            #update T2 via DIIS
            p = length(R_iter_storage)
            diis_result = PerformDiis(R_iter_storage,p,T2_storage,R2_storage)
            if diis_result==false
                nothing
            else
                T2_old = diis_result
            end
        elseif i == p_min - 1 #DIIS starts next iteration so add current stuff to memory
            println("DIIS will start in next iteration")
            push!(R_iter_storage, copy(R_iter))
            push!(T2_storage, copy(T2_old))
            push!(R2_storage, copy(R2))
        else
            println("DIIS start is still far away")
        end
        e_old = e_new
        e_new = calculate_ECCD()
        println("Energy after updating = $(e_new[1]+erhf)\n-------------------------\n")
        push!(earr,e_new[1]+erhf)
    end
    
end
maxitr = 100;
@time ccd_by_hand(maxitr);