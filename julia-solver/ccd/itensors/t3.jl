using TensorOperations, Serialization
nv,nocc = 2,4
begin
#Declare index i_1
#Declare index i_2
#Declare index i_3
#Declare index i_4
#Declare index a_1
#Declare index a_2
#Declare index a_3
#Declare index a_4
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
@tensor I_vvoo[a_1, a_2, i_1, i_2] += -1*g_vovo[a_2, i_3, a_3, i_1]*T2_vvoo[a_1, a_3, i_3, i_2]     #fourth update



display(I_vvoo)
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
@tensor I_vvoo[a_1, a_2, i_1, i_2] += 2*I_ovov[i_4, a_1, i_1, a_4]*T2_vvoo[a_2, a_4, i_2, i_4]                  #10th update
# display(I_vvoo)
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
@tensor I_vvoo[a_1, a_2, i_1, i_2] += I_ovov[i_4, a_1, i_1, a_3]*T2_vvoo[a_2, a_3, i_4, i_2]         #15th update








# display(I_vvoo)     #15th update obdi shob same asche
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
@tensor I_vvoo[a_1, a_2, i_1, i_2] += -2*I_vv[a_2, a_4]*T2_vvoo[a_1, a_4, i_1, i_2]       #18th update
# display(I_vvoo)                                                                             #Till here all matches














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
@tensor I_vvoo[a_1, a_2, i_1, i_2] += 1/2*I_oooo[i_3, i_4, i_1, i_2]*T2_vvoo[a_1, a_2, i_3, i_4]   #20th update
@tensor R2[a,b,i,j] := I_vvoo[a,b,i,j] + I_vvoo[b,a,j,i]
end

display(R2)



