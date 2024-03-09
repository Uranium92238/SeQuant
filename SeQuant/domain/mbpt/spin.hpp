//
// Created by Eduard Valeyev on 2019-02-27.
//

#ifndef SEQUANT_SPIN_HPP
#define SEQUANT_SPIN_HPP

#include <SeQuant/core/tensor_network.hpp>
#include "SeQuant/core/tensor.hpp"

#include <unordered_map>

namespace sequant {

/// @brief Applies index replacement rules to an ExprPtr
/// @param expr ExprPtr to transform
/// @param index_replacements index replacement map
/// @param scaling_factor to scale the result
/// @return a substituted and scaled expression pointer
ExprPtr transform_expr(const ExprPtr& expr,
                       const container::map<Index, Index>& index_replacements,
                       Constant::scalar_type scaling_factor = 1);

/// @brief Preserving particle symmetry, swaps bra and ket labels on all tensors
/// in an expression
/// @param expr ExprPtr to transform
/// @return transformed expression
ExprPtr swap_bra_ket(const ExprPtr& expr);

/// @brief Adds spins to indices in an expression with a replacement map
/// @param expr an expression pointer
/// @param index_replacements a map of pairs containing the index and the
/// corresponding replacement
/// @return expr the ExprPtr with substituted indices
ExprPtr append_spin(const ExprPtr& expr,
                    const container::map<Index, Index>& index_replacements);

/// @brief Removes spin label from all indices in an expression
/// @param expr an ExprPtr with spin indices
/// @return expr an ExprPtr with spin labels removed
ExprPtr remove_spin(const ExprPtr& expr);

/// @brief Checks the spin symmetry of a pair of indices corresponding to a
/// particle in tensor notation
/// @param tensor a tensor with indices containing spin labels
/// @return true if spin symmetry matches for all pairs of indices
bool spin_symm_tensor(const Tensor& tensor);

/// @brief Checks if spin labels on a tensor are same
/// @param tensor a tensor with indices containing spin labels
/// @return true if all spin labels on a tensor are the same
bool same_spin_tensor(const Tensor& tensor);

/// @brief Check if the number of alpha spins in bra and ket are equal;
/// beta spins will match if total number of indices is the same
/// @param tensor with spin indices
/// @return true if number of alpha spins match in bra and ket
bool can_expand(const Tensor& tensor);

/// @brief expand an antisymmetric tensor
///
/// @details For spin-indices, the tensor is NOT expanded if all spin-labels
/// are either alpha or beta
/// @param[in] tensor a tensor
/// @param[in] skip_spinsymm is true, will not expand tensors whose indices
///            all have the same spin [default=false]
/// @return an ExprPtr containing the sum of expanded terms, if antisymmetric
ExprPtr expand_antisymm(const Tensor& tensor, bool skip_spinsymm = false);

// TODO: Correct this function
/// @brief expands all antisymmetric tensors in a product
/// @param expr an expression pointer to expand
/// @return an expression pointer with expanded tensors as a sum
ExprPtr expand_antisymm(const ExprPtr& expr, bool skip_spinsymm = false);

/// @brief Check if a tensor with a certain label is present in an expression
/// @param expr input expression
/// @param label tensor label to find in the expression
/// @return true if tensor with given label is found
bool has_tensor(const ExprPtr& expr, std::wstring label);

/// @brief Generates a vector of replacement maps for Antisymmetrizer operator
/// @param A An antisymmetrizer tensor (A) (with > 2 particle indices)
/// @return Vector of replacement maps
container::svector<container::map<Index, Index>> A_maps(const Tensor& A);

/// @brief Removes tensor with a certain label from product
/// @param product A product expression
/// @param label Label of the tensor to remove
/// @return ExprPtr with the tensor removed
ExprPtr remove_tensor(const Product& product, std::wstring label);

/// @brief Removes tensor with a certain label from an expression
/// @param expr An expression pointer
/// @param label Label of the tensor to remove
/// @return ExprPtr with the tensor removed
ExprPtr remove_tensor(const ExprPtr& expr, std::wstring label);

/// @brief Expand a product containing the Antisymmetrization (A) operator
/// @param A product term with/without A operator
/// @return an ExprPtr containing sum of expanded terms if A is present
ExprPtr expand_A_op(const Product& product);

/// @brief Write expression in terms of Symmetrizer (S operator)
/// @param product
/// @return expression pointer with Symmstrizer operator
ExprPtr symmetrize_expr(const Product& product);

/// @brief Expand an expression containing the Antisymmetrization (A) operator
/// @param expr any ExprPtr
/// @return an ExprPtr containing sum of expanded terms if A is present
ExprPtr symmetrize_expr(const ExprPtr& expr);

/// @brief Expand an expression containing the Antisymmetrization (A) operator
/// @param expr any ExprPtr
/// @return an ExprPtr containing sum of expanded terms if A is present
ExprPtr expand_A_op(const ExprPtr& expr);

/// @brief Generates a vector of replacement maps for particle permutation
/// operator
/// @param P A particle permutation operator (with > 2 particle indices)
/// @return Vector of replacement maps
container::svector<container::map<Index, Index>> P_maps(
    const Tensor& P, bool keep_canonical = true, bool pair_wise = false);

/// @brief Expand a product containing the particle permutation (P) operator
/// @param A product term with/without P operator
/// @return an ExprPtr containing sum of expanded terms if P is present
ExprPtr expand_P_op(const Product& product, bool keep_canonical = true,
                    bool pair_wise = true);

/// @brief Expand an expression containing the particle permutation (P) operator
/// @param expr any ExprPtr
/// @return an ExprPtr containing sum of expanded terms if P is present
ExprPtr expand_P_op(const ExprPtr& expr, bool keep_canonical = true,
                    bool pair_wise = true);

container::svector<container::map<Index, Index>> S_replacement_maps(
    const Tensor& S);

/// @brief Expand S operator
ExprPtr S_maps(const ExprPtr& expr);

/// @brief Returns the number of cycles

/// Counts the number of cycles of a permutation represented in a 2-line form
/// by stacking \p v0 and \p v1 on top of each other.
/// @tparam Seq0 (reference to) a container type
/// @tparam Seq1 (reference to) a container type
/// @param v0 first sequence; if passed as an rvalue reference, it is moved from
/// @param[in] v1 second sequence
/// @param \p v0 is a permutation of \p v1
/// @return the number of cycles
template <typename Seq0, typename Seq1>
std::size_t count_cycles(Seq0&& v0, const Seq1& v1) {
  std::remove_reference_t<Seq0> v(std::forward<Seq0>(v0));
  using T = std::decay_t<decltype(v[0])>;
  assert(ranges::is_permutation(v, v1));

  auto make_null = []() -> T {
    if constexpr (std::is_arithmetic_v<T>) {
      return -1;
    } else if constexpr (std::is_same_v<T, Index>) {
      return L"p_50";
    } else  // unreachable
      abort();
  };

  std::size_t n_cycles = 0;
  const auto null = make_null();
  assert(ranges::contains(v, null) == false);
  assert(ranges::contains(v1, null) == false);
  for (auto it = v.begin(); it != v.end(); ++it) {
    if (*it != null) {
      n_cycles++;
      auto idx = std::distance(v.begin(), it);
      auto it0 = it;
      auto it1 = std::find(v1.begin(), v1.end(), *it0);
      auto idx1 = std::distance(v1.begin(), it1);
      do {
        it0 = std::find(v.begin(), v.end(), v[idx1]);
        it1 = std::find(v1.begin(), v1.end(), *it0);
        idx1 = std::distance(v1.begin(), it1);
        *it0 = null;
      } while (idx1 != idx);
    }
  }
  return n_cycles;
};

/// @brief Transforms an expression from spin orbital to spatial orbitals
/// @details This functions is designed for integrating spin out of expression
/// with Coupled Cluster equations in mind.
/// @attention This function may fail on arbitrarily created expressions that
/// lacks proper index attributes.
/// @param expr ExprPtr with spin orbital indices
/// @param ext_index_groups groups of external indices
/// @return an expression with spin integrated/adapted
ExprPtr closed_shell_spintrace(
    const ExprPtr& expression,
    const container::svector<container::svector<Index>>& ext_index_groups = {});

///
/// \brief Given a OpType::A or OpType::S tensor, generates a list of external
///        indices.
/// \note The argument tensor must have a label of 'S' or 'A' do denote that
///       it is OpType::S or OpType::A respectively.
///
container::svector<container::svector<Index>> external_indices(Tensor const&);

///
/// @param nparticles Number of indices in bra of the target tensor. That must
///                   be equal to the same in the ket.
/// @deprecated not CSV-compatible, and mixes bra and ket relative to
/// external_indices(expr) , will be deprecated
//[[deprecated("use external_indices(expr)")]] container::svector<
//    container::svector<Index>> external_indices(size_t nparticles);

/// @brief Transforms Coupled cluster from spin orbital to spatial orbitals
/// @details The external indices are deduced from Antisymmetrization operator
/// @param expr ExprPtr to Sum type with spin orbital indices
/// @return an expression with spin integrated/adapted
ExprPtr closed_shell_CC_spintrace(ExprPtr const& expr);

/// \brief Same as \c closed_shell_CC_spintrace except internally uses
///        \c sequant::spintrace instead of sequant::closed_shell_spintrace.
ExprPtr closed_shell_CC_spintrace_rigorous(ExprPtr const& expr);

/// Collect all indices from an expression
auto index_list(const ExprPtr& expr);

/// @brief Swap spin labels in a tensor
Tensor swap_spin(const Tensor& t);

/// @brief Swap spin labels in an expression
ExprPtr swap_spin(const ExprPtr& expr);

/// @brief Merge operators into a single operator (designed for P operator)
/// @warning Repetition of indices is allowed in a bra or a ket
ExprPtr merge_tensors(const Tensor& O1, const Tensor& O2);

/// @brief Vector of Anti-symmetrizers for spin-traced open-shell expr
std::vector<ExprPtr> open_shell_A_op(const Tensor& A);

/// @brief Generate a vector of permutation operators for partial expansion of
/// antisymmstrizer
/// @details The antisymmetrizer need not be fully expanded to spin-trace for
/// open-shell case. By expanding only the unlike spin terms, the
/// antisymmetrizer is pereserved for same-spin particle indices.
/// @param A Antisymmetrizer tensor produced in Coupled Cluster
/// @return a vector of expression pointers containing permutation operators as
/// a sum
/// @warning This function assumes the antisymmetrizer (A) has a canonical form
std::vector<ExprPtr> open_shell_P_op_vector(const Tensor& A);

/// @brief Generates spin expressions to be used for open-shell coupled cluster
/// @details Every spin combination of external indices will have all spin
/// combinations of internal indices.
/// @param expr ExprPtr with spin orbital indices
/// @param ext_index_groups groups of external indices
/// @param single_spin_case Calculate open-shell expression for a specific spin
/// case
/// @return a vector of expr ptrs with spin expressions
std::vector<ExprPtr> open_shell_spintrace(
    const ExprPtr& expr,
    const container::svector<container::svector<Index>>& ext_index_groups,
    const int single_spin_case = 0);

/// @brief Transforms Coupled cluster from spin orbital to spatial orbitals
/// @param expr ExprPtr to Sum type with spin orbital indices
/// @return a vector of spin expressions for open-shell reference
std::vector<ExprPtr> open_shell_CC_spintrace(const ExprPtr& expr);

/// @brief Transforms an expression from spin orbital to spatial orbitals
/// @details Given an expression, this function extracts all indices and adds a
/// spin attribute to all the indices in the expression. A map is generated with
/// all possible spin permutations and substituted in the expression. Based on
/// spin symmetry of particle indices: the non-zero terms are expanded, the spin
/// labels removed and a sum of all non-zero expressions is returned.
/// @param expr ExprPtr with spin orbital indices
/// @param ext_index_groups groups of external indices
/// @param spinfree_index_spaces if true, will assume that all index spaces are
/// spin-free and will remove spin from all indices of the result before
/// returning
/// @return an expression with spin integrated/adapted
ExprPtr spintrace(
    const ExprPtr& expression,
    container::svector<container::svector<Index>> ext_index_groups = {},
    bool spinfree_index_spaces = true);

/// @brief Factorize S out of terms
/// @details Given an expression, permute indices and check if a given product
/// @param expression Expression pointer
/// @param fast_method use hash maps (memory intensive) for faster evaluation
/// @param ext_index_groups External index groups to geenrate S operator
/// @return ExprPtr with terms with S operator as a factor
ExprPtr factorize_S(const ExprPtr& expression,
                    std::initializer_list<IndexList> ext_index_groups,
                    bool fast_method = true);

ExprPtr biorthogonal_transform(
    const sequant::ExprPtr& expr,
    const container::svector<container::svector<sequant::Index>>&
        ext_index_groups = {},
    double threshold = 1.e-12);

}  // namespace sequant

#endif  // SEQUANT_SPIN_HPP
