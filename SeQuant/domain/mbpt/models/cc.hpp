#ifndef SEQUANT_DOMAIN_MBPT_MODELS_CC_HPP
#define SEQUANT_DOMAIN_MBPT_MODELS_CC_HPP

#include <SeQuant/core/container.hpp>
#include <SeQuant/core/expr_fwd.hpp>
#include <SeQuant/core/index.hpp>
#include <SeQuant/core/timer.hpp>

namespace sequant::mbpt::sr {

/// derives equations of traditional coupled-cluster method
class CC {
  size_t N, P, PMIN;

 public:
  /// @brief constructor for CC class
  /// @param n coupled cluster excitation rank
  /// @param p projector excitation rank
  /// @param pmin minimum projector excitation rank
  CC(size_t n, size_t p = std::numeric_limits<size_t>::max(), size_t pmin = 1);

  /// @brief derives similarity-transformed expressions of mpbt::Operators
  /// @param expr expression to be transformed
  /// @param r order of truncation
  /// @pre expr should be composed of mbpt::Operators
  /// @return transformed expression
  ExprPtr sim_tr(ExprPtr expr, size_t r);

  /// @brief derives t amplitude equations
  /// @return std::vector of t amplitude equations
  std::vector<sequant::ExprPtr> t(bool screen = true, bool use_topology = true,
                                  bool use_connectivity = true,
                                  bool canonical_only = true);
  /// @brief derives λ amplitude equations
  /// @return std::vector of λ amplitude equations
  std::vector<sequant::ExprPtr> λ(bool screen = false, bool use_topology = true,
                                  bool use_connectivity = true,
                                  bool canonical_only = true);

  // clang-format off
  /// @brief derives perturbed t amplitude equations
  /// @param o order of perturbation
  /// @param r rank of perturbation operator. r = 1 means one-body perturbation operator
  /// @pre `r==1 && o==1`, only first order perturbation and one-body perturbation operator is supported now
  /// @return std::vector of perturbed t amplitude equations
  // clang-format on
  std::vector<sequant::ExprPtr> t_pt(size_t o, size_t r);

  // clang-format off
  /// @brief derives perturbed λ amplitude equations
  /// @param o order of perturbation
  /// @param r rank of perturbation operator. r = 1 means one-body perturbation operator
  /// @pre `r==1 && o==1`, only first order perturbation and one-body perturbation operator is supported now
  /// @return std::vector of perturbed λ amplitude equations
  // clang-format on
  std::vector<sequant::ExprPtr> λ_pt(size_t o, size_t r);
};  // class CC

}  // namespace sequant::mbpt::sr

#endif  // SEQUANT_DOMAIN_MBPT_MODELS_CC_HPP
