include("ccd-helper.jl")
T2 = initialize_t2_only();
Scaled_R2 = zeros(2,2,4,4);
R2 = zeros(2,2,4,4);

R2 = calcresnew(T2);
T2 = update_T2(T2,R2,Scaled_R2);
display(T2)
display(R2)
