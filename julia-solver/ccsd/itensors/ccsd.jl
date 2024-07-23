using Printf,Plots
include("ccsd-helper.jl")
function CCSD_by_hand(maxitr)
    initialize_cc_variables()
    nv::Int64   = deserialize("nv.jlbin")
    no::Int64 = deserialize("no.jlbin")
    a_1= Index(nv, "a_1")
    a_2= Index(nv, "a_2")
    i_1= Index(no, "i_1")
    i_2= Index(no, "i_2")
    erhf::Float64 = deserialize("erhf.jlbin")


    R2 = ITensor(zeros(Float64,nv,nv,no,no),a_1,a_2,i_1,i_2)
    R1 = ITensor(zeros(Float64,nv,no),a_1,i_1)
    Scaled_R2 = ITensor(zeros(Float64,nv,nv,no,no),a_1,a_2,i_1,i_2)
    Scaled_R1 = ITensor(zeros(Float64,nv,no),a_1,i_1)
    T2 = ITensor(initialize_t2_only(),a_1,a_2,i_1,i_2)
    T1 = copy(R1)


    normtol=1.0e-8
    serialize("T2_vvoo.jlbin",Array(T2,a_1,a_2,i_1,i_2))
    serialize("T1_vo.jlbin",Array(T1,a_1,i_1))
    INTpp_vvoo = copy(R2)
    e_new = Float64(calculate_ECCSD()[])
    e_old = copy(0.0)
    etol = 1.0e-10
    p_max =  6# Maximum number of previous iterations to store for DIIS
    p_min = 2 # DIIS will start after this many iterations
    T2_storage = Array{typeof(T2)}(undef, 0)
    R2_storage = Array{typeof(R2)}(undef, 0)
    T1_storage = Array{typeof(T1)}(undef, 0)
    R1_storage = Array{typeof(R1)}(undef, 0)
    println("Starting CCSD Iterations with Convergence Criteria as ||R₂||<$normtol , ΔE<$etol and max iterations=$maxitr")
    println("-----------------------------------------------------------------------------------------------")
    for i in 1:maxitr
        #Print Iteration Information
        @printf("\n-------------------------\nStarting Iteration: %d  Current Total Energy: %.10f\n", i, e_new+erhf)
        println("Energy before updating = $(e_old[1]+erhf)")

        #Serialize Amplitude related tensors to be used by calc_R1 and calc_R2
        serialize("T2_vvoo.jlbin",Array(T2,a_1,a_2,i_1,i_2))
        serialize("T1_vo.jlbin",Array(T1,a_1,i_1))
        INTpp_vvoo = calculate_INTpp_vvoo!(T2,T1,INTpp_vvoo)
        serialize("INTpp_vvoo.jlbin",Array(INTpp_vvoo,a_1,a_2,i_1,i_2))

        #Calculate Residuals
        R2 = calc_R2()
        R1 = calc_R1()

        #Update Amplitudes
        T2,T1 = update_amplitudes(T2,R2,Scaled_R2,T1,R1,Scaled_R1)

        #Check Convergence
        conv,rnorm = check_convergence(R1,R2,normtol,e_old,e_new,etol)
        if conv
            @printf("CCSD Converged in %d iterations, no further updates required,current ||R||=%.10f and ΔE = %.12f\n\n-------------------------\nE_Total = %.10f\n", i,rnorm,abs(e_old-e_new),e_new+erhf)
            break
        end
        @printf("CCSD Not Converged in %d iterations, current ||R|| = %.10f and ΔE = %.12f \n", i,rnorm,abs(e_old[1]-e_new[1]))

        
        #DIIS
        if i >= p_min   #DIIS  starts
            println("DIIS is being implemented in iteration $i")
            # if i >= p_min + p_max - 1
            if length(R2_storage)>=p_max  # Adjusted condition to start popping correctly
                popfirst!(T2_storage)
                popfirst!(R2_storage)
                popfirst!(T1_storage)
                popfirst!(R1_storage)
            end
            push!(T2_storage, copy(T2))
            push!(R2_storage, copy(R2))
            push!(T1_storage, copy(T1))
            push!(R1_storage, copy(R1))
            p = length(R2_storage)
            diis_result = PerformDiis(p,T2_storage,R2_storage,T1_storage,R1_storage,false)
            if diis_result==false
                nothing
            else
                T2,T1 = diis_result
            end
        elseif i == p_min - 1 #DIIS starts next iteration so add current stuff to memory
            println("DIIS will start in next iteration")
            push!(T2_storage, copy(T2))
            push!(R2_storage, copy(R2))
            push!(T1_storage, copy(T1))
            push!(R1_storage, copy(R1))
            p = length(R2_storage)
            PerformDiis(p,T2_storage,R2_storage,T1_storage,R1_storage,true) #dummy call
        else
            println("DIIS start is still far away")
        end
        e_old = e_new
        e_new = Float64(calculate_ECCSD()[])
        println("Energy after updating = $(e_new+erhf)\n-------------------------\n")
    end
    
end
maxitr = 50;
@time CCSD_by_hand(maxitr);