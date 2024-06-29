include("ccd-helper.jl")
using ITensors
a_1 = Index(2,"a_1");
a_2 = Index(2,"a_2");
i_1 = Index(4,"i_1");
i_2 = Index(4,"i_2");
a_3 = Index(2,"a_3");
i_3 = Index(4,"i_3");





T2 = ITensor(initialize_t2_only(),a_1,a_2,i_1,i_2);
g_vvoo = ITensor(deserialize("g_vvoo.jlbin"),a_1,a_2,i_1,i_2);
R2u = ITensor(zeros(Float64,2,2,4,4),a_1,a_2,i_1,i_2);
Trm1 = 0.5* g_vvoo;
R2u += Trm1;
g_voov = ITensor(deserialize("g_voov.jlbin"),a_1,i_3,i_1,a_3);
T2 = ITensor(initialize_t2_only(),a_2,a_3,i_3,i_2);
Trm2 = - g_voov* T2;
R2u += Trm2;
pp = Trm2 + g_vvoo
pp