using BenchmarkTools
include("ccd-helper.jl")
# using BliContractor
T2 = initialize_t2_only();
R2 = calcresnew();
C ::Float64 = 0.0 
@time @tensor C += T2[a,b,i,j]*R2[a,b,i,j]