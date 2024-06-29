using Serialization, ITensors
using TensorOperations
include("ccd-helper.jl")
initialize_cc_variables()

begin
a_1 = Index(2,"a_1")
a_2 = Index(2,"a_2")
i_1 = Index(4,"i_1")
i_2 = Index(4,"i_2")
end




begin
T2 = initialize_t2_only()
R2 = zeros(Float64,2,2,4,4)
Scaled_R2 = zeros(Float64,2,2,4,4)
end
# serialize("T2_vvoo.jlbin",T2)



begin
iT2 = ITensor(T2,a_1,a_2,i_1,i_2)
# iR2 = ITensor(R2,a_1,a_2,i_1,i_2)
iScaled_R2 = ITensor(Scaled_R2,a_1,a_2,i_1,i_2)
end



for i in 1:25
    # @show R2
    R2 = calculate_residual_memeff(T2);
    T2 = update_T2(T2,R2,Scaled_R2);
    # display(R2)
end
for i in 1:25
    aT2 = Array(iT2,a_1,a_2,i_1,i_2)
    serialize("T2_vvoo.jlbin",aT2)
    iR2 = calculate_residual_memeff()
    iT2 = update_T2(iT2,iR2,iScaled_R2)
    # @show iR2
end

@show iR2
display(R2)



iR2 = calculate_residual_memeff()
Array(iR2,a_1,a_2,i_1,i_2)
SquareNorm(Array(iR2,a_1,a_2,i_1,i_2))
R2 = calculate_residual_memeff(T2)
SquareNorm(R2)
norm(R2)
norm(iR2)
# I_vvoo = rand(Float64,2,2,4,4)
# begin
#     a_1 = Index(2,"a_1")
#     a_2 = Index(2,"a_2")
#     i_1 = Index(4,"i_1")
#     i_2 = Index(4,"i_2")
# end
# iI_vvoo = ITensor(I_vvoo,a_1,a_2,i_1,i_2)
# iR = iI_vvoo + swapinds(iI_vvoo,(a_1,i_1),(a_2,i_2))
# @tensor R[a,b,i,j] := I_vvoo[a,b,i,j] + I_vvoo[b,a,j,i]


# Array(iR,a_1,a_2,i_1,i_2) - R
