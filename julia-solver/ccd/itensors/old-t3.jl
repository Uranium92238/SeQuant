using ITensors
include("ccd-helper.jl")
nv,no = 2,4
a_1 = Index(nv,"a_1");
a_2 = Index(nv,"a_2");
i_1 = Index(no,"i_1");
i_2 = Index(no,"i_2");
iT2 = ITensor(initialize_t2_only(),a_1,a_2,i_1,i_2);
iR2u = ITensor(zeros(2,2,4,4),a_1,a_2,i_1,i_2);
# iR2 = calcresnew(iT2);
Scaled_iR2 = ITensor(zeros(2,2,4,4),a_1,a_2,i_1,i_2);
# @show iR2
# iT2 = update_T2(iT2,iR2,Scaled_iR2);
# @show iT2
# @show iR2
# norm(iR2)


iR2 = calcresnew(iT2);
iT2 = update_T2(iT2,iR2,Scaled_iR2);
@show iR2
@show iT2