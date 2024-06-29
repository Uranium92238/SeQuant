using Printf,Plots, TimerOutputs
include("ccd-helper.jl")
function ccd_by_hand(maxitr)
    initialize_cc_variables()
    nv::Int64 = deserialize("nv.jlbin")
    nocc::Int64 = deserialize("nocc.jlbin")
    erhf::Float64 = deserialize("erhf.jlbin")
    R2 = TensorMap(zeros(Float64,nv,nv,nocc,nocc),ℝ^nv ⊗ ℝ^nv , ℝ^nocc ⊗ ℝ^nocc)
    println("R2 is of type $(typeof(R2))")
    R2u = TensorMap(zeros(Float64,nv,nv,nocc,nocc),ℝ^nv ⊗ ℝ^nv , ℝ^nocc ⊗ ℝ^nocc)
    R_iter = copy(R2) 
    Scaled_R2 = copy(R2)
    T2_old::TrivialTensorMap{CartesianSpace, 2, 2, Matrix{Float64}}= initialize_t2_only()
    println("T2 is of type $(typeof(T2_old))")
    normtol=1.0e-8
    e_new::Float64 = calculate_ECCD(T2_old)
    e_old::Float64 = copy(0.0)
    etol = 1.0e-10
    p_max =  6# Maximum number of previous iterations to store for DIIS
    p_min = 2 # DIIS will start after this many iterations
    R_iter_storage = Array{typeof(R2)}(undef, 0)
    T2_storage::Vector{typeof(T2_old)} = Array{typeof(T2_old)}(undef, 0)
    R2_storage = Array{typeof(R2)}(undef, 0)
    println("R2_storage is of type $(typeof(R2_storage))")
    println("Starting CCD Iterations with Convergence Criteria as ||R₂||<$normtol , ΔE<$etol and max iterations=$maxitr")
    println("-----------------------------------------------------------------------------------------------")
    for i in 1:maxitr
        @printf("\n-------------------------\nStarting Iteration: %d  Current Total Energy: %.8f\n", i, e_new+erhf)
        println("Energy before updating = $(e_old+erhf)")
        # R2 = calculate_residual(T2_old,R2u,R2)
        R2 = calcresnew(T2_old)
        println("R2 is of type $(typeof(R2))")
        T2_old = update_T2(T2_old,R2,Scaled_R2)
        println("T2 is of type $(typeof(T2_old))")
        conv::Bool,r2norm::Float64 = check_convergence(R2,normtol,e_old,e_new,etol)
        if conv
            @printf("CCD Converged in %d iterations, no further updates required,current ||R2||=%.10f and ΔE = %.12f\n\n-------------------------\nE_Total = %.8f\n", i,r2norm,abs(e_old-e_new),e_new+erhf)
            break
        end
        R_iter = calculate_R_iter(R_iter,R2)
        @printf("CCD Not Converged in %d iterations, current ||R2|| = %.10f and ΔE = %.12f \n", i,r2norm,abs(e_old-e_new))
        if i >= p_min   #DIIS  starts
            println("DIIS is being implemented in iteration $i")
            # if i >= p_min + p_max - 1
            if length(R_iter_storage)>=p_max  # Adjusted condition to start popping correctly
                popfirst!(R_iter_storage)
                popfirst!(T2_storage)
                popfirst!(R2_storage)
            end
            push!(R_iter_storage, copy(R_iter))
            push!(T2_storage, copy(T2_old))
            push!(R2_storage, copy(R2))
            #update T2 via DIIS
            p::Int64 = length(R_iter_storage)
            diis_result = PerformDiis(R_iter_storage,p,T2_storage,R2_storage)
            if diis_result==false
                # T2_old = update_T2(T2_old,R2,fvv,foo)
                nothing
            else
                T2_old = diis_result
                # println("DIIS has updated T2")
                # display(T2_old)
            end
        elseif i == p_min - 1 #DIIS starts next iteration so add current stuff to memory
            println("DIIS will start in next iteration")
            push!(R_iter_storage, copy(R_iter))
            push!(T2_storage, copy(T2_old))
            push!(R2_storage, copy(R2))
            just_show_Bmatrix(R_iter_storage,length(R_iter_storage),T2_storage,R2_storage)
        else
            println("DIIS start is still far away")
            # No change to memory
        end
        e_old = e_new
        e_new = calculate_ECCD(T2_old)
        println("Energy after updating = $(e_new+erhf)\n-------------------------\n")
    end
    
end
maxitr = 100;
# @code_warntype ccd_by_hand(maxitr);
@time ccd_by_hand(maxitr)