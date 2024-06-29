include("ccd-helper.jl")
using ITensors
# This is for testing the calcresnew() function of the two different implementations.
a_1 = Index(2,"a_1");
a_2 = Index(2,"a_2");
i_1 = Index(4,"i_1");
i_2 = Index(4,"i_2");

# T2 = rand(Float64,2,2,4,4)
T2 = initialize_t2_only();
Scaled_R2 = zeros(Float64,2,2,4,4)
R2 = zeros(Float64,2,2,4,4)
iT2 = ITensor(T2,a_1,a_2,i_1,i_2)
iScaled_R2 = ITensor(Scaled_R2,a_1,a_2,i_1,i_2)
iR2 = ITensor(R2,a_1,a_2,i_1,i_2)


T2 = update_T2(T2,R2,Scaled_R2);
iT2 = update_T2(iT2,iR2,iScaled_R2);

R2 = calcresnew(T2);
iR2 = calcresnew(iT2);

display(T2)
@show iT2


display(R2)
@show iR2
## Here we have proved that for any random tensor and it's ITensor counterpart