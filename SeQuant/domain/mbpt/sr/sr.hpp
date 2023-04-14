//
// Created by Eduard Valeyev on 2019-02-19.
//

#ifndef SEQUANT_SRCC_HPP
#define SEQUANT_SRCC_HPP

#include <limits>
#include <utility>

#include "SeQuant/core/expr_fwd.hpp"
#include "SeQuant/core/sequant.hpp"
#include "SeQuant/core/space.hpp"
#include "SeQuant/domain/mbpt/op.hpp"

namespace sequant {
namespace mbpt {
namespace sr {
namespace so {

class make_op {
  std::size_t nbra_, nket_;
  OpType op_;

 public:
  make_op(std::size_t nbra, std::size_t nket, OpType op);

  ExprPtr operator()(IndexSpace::Type unocc, IndexSpace::Type occ) const;

  ExprPtr operator()() const;
};

make_op Op(OpType _Op, std::size_t Nbra,
           std::size_t Nket = std::numeric_limits<std::size_t>::max());

#include "sr_op.impl.hpp"

ExprPtr H1();

ExprPtr H2();

ExprPtr H0mp();
ExprPtr H1mp();

ExprPtr F();
ExprPtr W();

ExprPtr H();

/// computes the vacuum expectation value (VEV)

/// @param[in] expr input expression
/// @param[in] op_connections specifies the connectivity to be ensured
/// @param[in] use_top if true, topological equivalence will be utilized
/// @return the VEV
ExprPtr vac_av(ExprPtr expr,
               std::vector<std::pair<int, int>> op_connections = {},
               bool use_top = true);

}  // namespace so
}  // namespace sr
}  // namespace mbpt
}  // namespace sequant

#endif  // SEQUANT_SRCC_HPP
