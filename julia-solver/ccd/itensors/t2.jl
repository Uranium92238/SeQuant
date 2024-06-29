using ITensors, Serialization
nocc,nv = 4,2
begin
i_1= Index(nocc, "i_1")
i_2= Index(nocc, "i_2")
i_3= Index(nocc, "i_3")
i_4= Index(nocc, "i_4")
a_1= Index(nv, "a_1")
a_2= Index(nv, "a_2")
a_3= Index(nv, "a_3")
a_4= Index(nv, "a_4")
I_vvoo = ITensor(zeros(Float64, nv, nv, nocc, nocc), a_1, a_2, i_1, i_2) #create function generated this line
g_voov = ITensor(deserialize("g_voov.jlbin"),a_1, i_3, i_1, a_3)

T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"),a_2, a_3, i_3, i_2)

I_vvoo += -1*g_voov*T2_vvoo
#Unload T2_vvoo
#Unload g_voov
g_vvoo = ITensor(deserialize("g_vvoo.jlbin"),a_1, a_2, i_1, i_2)

I_vvoo += 1/2*g_vvoo
#Unload g_vvoo
g_vovo = ITensor(deserialize("g_vovo.jlbin"),a_2, i_3, a_3, i_2)

T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"),a_1, a_3, i_1, i_3)

I_vvoo += -1*g_vovo*T2_vvoo
#Unload T2_vvoo
#Unload g_vovo
g_vovo = ITensor(deserialize("g_vovo.jlbin"),a_2, i_3, a_3, i_1)

T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"),a_1, a_3, i_3, i_2)

I_vvoo += -1*g_vovo*T2_vvoo                                                                 #fourth update
@show I_vvoo
#Unload T2_vvoo
#Unload g_vovo
g_voov = ITensor(deserialize("g_voov.jlbin"),a_1, i_3, i_1, a_3)

T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"),a_2, a_3, i_2, i_3)

I_vvoo += 2*g_voov*T2_vvoo
#Unload T2_vvoo
#Unload g_voov
g_oooo = ITensor(deserialize("g_oooo.jlbin"),i_3, i_4, i_1, i_2)

T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"),a_1, a_2, i_3, i_4)

I_vvoo += 1/2*g_oooo*T2_vvoo
#Unload T2_vvoo
#Unload g_oooo
f_vv = ITensor(deserialize("f_vv.jlbin"),a_2, a_3)

T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"),a_1, a_3, i_1, i_2)

I_vvoo += f_vv*T2_vvoo
#Unload T2_vvoo
#Unload f_vv
g_vvvv = ITensor(deserialize("g_vvvv.jlbin"),a_1, a_2, a_3, a_4)

T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"),a_3, a_4, i_1, i_2)

I_vvoo += 1/2*g_vvvv*T2_vvoo
#Unload T2_vvoo
#Unload g_vvvv
f_oo = ITensor(deserialize("f_oo.jlbin"),i_3, i_2)

T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"),a_1, a_2, i_1, i_3)

I_vvoo += -1*f_oo*T2_vvoo
#Unload T2_vvoo
#Unload f_oo
I_ovov = ITensor(zeros(Float64, nocc, nv, nocc, nv), i_4, a_1, i_1, a_4) #create function generated this line
g_oovv = ITensor(deserialize("g_oovv.jlbin"),i_3, i_4, a_3, a_4)

T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"),a_1, a_3, i_1, i_3)

I_ovov += g_oovv*T2_vvoo
#Unload T2_vvoo
#Unload g_oovv
T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"),a_2, a_4, i_2, i_4)

I_vvoo += 2*I_ovov*T2_vvoo                              #10th update
# @show I_vvoo
#Unload T2_vvoo
#Unload I_ovov
I_ovov = ITensor(zeros(Float64, nocc, nv, nocc, nv), i_4, a_1, i_1, a_4)#Load and set to zero julia equivalent

g_oovv = ITensor(deserialize("g_oovv.jlbin"),i_3, i_4, a_3, a_4)

T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"),a_1, a_3, i_1, i_3)

I_ovov += g_oovv*T2_vvoo
#Unload T2_vvoo
#Unload g_oovv
T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"),a_2, a_4, i_4, i_2)

I_vvoo += -2*I_ovov*T2_vvoo
#Unload T2_vvoo
#Unload I_ovov
I_ovov = ITensor(zeros(Float64, nocc, nv, nocc, nv), i_4, a_1, i_1, a_4)#Load and set to zero julia equivalent

g_oovv = ITensor(deserialize("g_oovv.jlbin"),i_3, i_4, a_3, a_4)

T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"),a_1, a_3, i_3, i_1)

I_ovov += g_oovv*T2_vvoo
#Unload T2_vvoo
#Unload g_oovv
T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"),a_2, a_4, i_4, i_2)

I_vvoo += 1/2*I_ovov*T2_vvoo
#Unload T2_vvoo
#Unload I_ovov
I_oo = ITensor(zeros(Float64, nocc, nocc), i_4, i_2) #create function generated this line
g_oovv = ITensor(deserialize("g_oovv.jlbin"),i_3, i_4, a_3, a_4)

T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"),a_3, a_4, i_3, i_2)

I_oo += g_oovv*T2_vvoo
#Unload T2_vvoo
#Unload g_oovv
T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"),a_1, a_2, i_1, i_4)

I_vvoo += -2*I_oo*T2_vvoo
#Unload T2_vvoo
#Unload I_oo
I_ovov = ITensor(zeros(Float64, nocc, nv, nocc, nv), i_4, a_1, i_1, a_3)#Load and set to zero julia equivalent

g_oovv = ITensor(deserialize("g_oovv.jlbin"),i_3, i_4, a_3, a_4)

T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"),a_1, a_4, i_1, i_3)

I_ovov += g_oovv*T2_vvoo
#Unload T2_vvoo
#Unload g_oovv
T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"),a_2, a_3, i_2, i_4)

I_vvoo += -1*I_ovov*T2_vvoo
#Unload T2_vvoo
#Unload I_ovov
I_ovov = ITensor(zeros(Float64, nocc, nv, nocc, nv), i_4, a_1, i_1, a_3)#Load and set to zero julia equivalent

g_oovv = ITensor(deserialize("g_oovv.jlbin"),i_3, i_4, a_3, a_4)

T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"),a_1, a_4, i_1, i_3)

I_ovov += g_oovv*T2_vvoo
#Unload T2_vvoo
#Unload g_oovv
T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"),a_2, a_3, i_4, i_2)

I_vvoo += I_ovov*T2_vvoo                #15th update










# @show I_vvoo   #15th update obdi shob same asche
#Unload T2_vvoo  
#Unload I_ovov
I_vv = ITensor(zeros(Float64, nv, nv), a_2, a_3) #create function generated this line
g_oovv = ITensor(deserialize("g_oovv.jlbin"),i_3, i_4, a_3, a_4)

T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"),a_2, a_4, i_4, i_3)

I_vv += g_oovv*T2_vvoo
#Unload T2_vvoo
#Unload g_oovv
T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"),a_1, a_3, i_1, i_2)

I_vvoo += I_vv*T2_vvoo
#Unload T2_vvoo
#Unload I_vv
I_oo = ITensor(zeros(Float64, nocc, nocc), i_4, i_2)#Load and set to zero julia equivalent

g_oovv = ITensor(deserialize("g_oovv.jlbin"),i_3, i_4, a_3, a_4)

T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"),a_3, a_4, i_2, i_3)

I_oo += g_oovv*T2_vvoo
#Unload T2_vvoo
#Unload g_oovv
T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"),a_1, a_2, i_1, i_4)

I_vvoo += I_oo*T2_vvoo
#Unload T2_vvoo
#Unload I_oo
I_vv = ITensor(zeros(Float64, nv, nv), a_2, a_4)#Load and set to zero julia equivalent

g_oovv = ITensor(deserialize("g_oovv.jlbin"),i_3, i_4, a_3, a_4)

T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"),a_2, a_3, i_4, i_3)

I_vv += g_oovv*T2_vvoo
#Unload T2_vvoo
#Unload g_oovv
T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"),a_1, a_4, i_1, i_2)

I_vvoo += -2*I_vv*T2_vvoo   #18th update
# @show I_vvoo                     #Till here everything matches











#Unload T2_vvoo
#Unload I_vv
I_ovov = ITensor(zeros(Float64, nocc, nv, nocc, nv), i_4, a_1, i_2, a_3)#Load and set to zero julia equivalent

g_oovv = ITensor(deserialize("g_oovv.jlbin"),i_3, i_4, a_3, a_4)

T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"),a_1, a_4, i_3, i_2)

I_ovov += g_oovv*T2_vvoo
#Unload T2_vvoo
#Unload g_oovv
T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"),a_2, a_3, i_4, i_1)

I_vvoo += 1/2*I_ovov*T2_vvoo
#Unload T2_vvoo
#Unload I_ovov
I_oooo = ITensor(zeros(Float64, nocc, nocc, nocc, nocc), i_3, i_4, i_1, i_2) #create function generated this line
g_oovv = ITensor(deserialize("g_oovv.jlbin"),i_3, i_4, a_3, a_4)

T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"),a_3, a_4, i_1, i_2)

I_oooo += g_oovv*T2_vvoo
#Unload T2_vvoo
#Unload g_oovv
T2_vvoo = ITensor(deserialize("T2_vvoo.jlbin"),a_1, a_2, i_3, i_4)
I_vvoo += 1/2*I_oooo*T2_vvoo               #20th update till here all same

# R = I_vvoo + permute(I_vvoo,a_2,a_1,i_2,i_1) #symmetrization here lies the problem
R = I_vvoo + swapinds(I_vvoo,(a_1,i_1),(a_2,i_2)) #correct symmetrization
end


@show R
Array(R,a_1,a_2,i_1,i_2)