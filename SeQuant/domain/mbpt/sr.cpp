//
// Created by Eduard Valeyev on 2019-02-19.
//

#include "SeQuant/domain/mbpt/sr.hpp"

#include "SeQuant/core/expr.hpp"
#include "SeQuant/core/math.hpp"
#include "SeQuant/core/op.hpp"
#include "SeQuant/core/tensor.hpp"
#include "SeQuant/core/wick.hpp"
#include "SeQuant/domain/mbpt/context.hpp"

namespace sequant {
namespace mbpt {
namespace sr {

qninterval_t ncre(qns_t qns, const IndexSpace::Type& s) {
  assert(s == IndexSpace::active_occupied ||
         s == IndexSpace::active_unoccupied);
  return s == IndexSpace::active_occupied ? qns[0] : qns[2];
}

qninterval_t ncre(qns_t qns, const IndexSpace& s) {
  assert((s.type() == IndexSpace::active_occupied ||
          s.type() == IndexSpace::active_unoccupied) &&
         s.qns() == IndexSpace::nullqns);
  return s.type() == IndexSpace::active_occupied ? qns[0] : qns[2];
}

qninterval_t ncre_occ(qns_t qns) {
  return ncre(qns, IndexSpace::active_occupied);
}

qninterval_t ncre_uocc(qns_t qns) {
  return ncre(qns, IndexSpace::active_unoccupied);
}

qninterval_t ncre(qns_t qns) { return qns[0] + qns[2]; }

qninterval_t nann(qns_t qns, const IndexSpace::Type& s) {
  assert(s == IndexSpace::active_occupied ||
         s == IndexSpace::active_unoccupied);
  return s == IndexSpace::active_occupied ? qns[1] : qns[3];
}

qninterval_t nann(qns_t qns, const IndexSpace& s) {
  assert((s.type() == IndexSpace::active_occupied ||
          s.type() == IndexSpace::active_unoccupied) &&
         s.qns() == IndexSpace::nullqns);
  return s.type() == IndexSpace::active_occupied ? qns[1] : qns[3];
}

qninterval_t nann_occ(qns_t qns) {
  return nann(qns, IndexSpace::active_occupied);
}

qninterval_t nann_uocc(qns_t qns) {
  return nann(qns, IndexSpace::active_unoccupied);
}

qninterval_t nann(qns_t qns) { return qns[1] + qns[3]; }

qns_t combine(qns_t a, qns_t b) {
  const auto ncontr_uocc =
      qninterval_t{0, std::min(ncre(b, IndexSpace::active_unoccupied).upper(),
                               nann(a, IndexSpace::active_unoccupied).upper())};
  const auto ncontr_occ =
      qninterval_t{0, std::min(nann(b, IndexSpace::active_occupied).upper(),
                               ncre(a, IndexSpace::active_occupied).upper())};
  const auto nc_occ =
      nonnegative(ncre(a, IndexSpace::active_occupied) +
                  ncre(b, IndexSpace::active_occupied) - ncontr_occ);
  const auto nc_uocc =
      nonnegative(ncre(a, IndexSpace::active_unoccupied) +
                  ncre(b, IndexSpace::active_unoccupied) - ncontr_uocc);
  const auto na_occ =
      nonnegative(nann(a, IndexSpace::active_occupied) +
                  nann(b, IndexSpace::active_occupied) - ncontr_occ);
  const auto na_uocc =
      nonnegative(nann(a, IndexSpace::active_unoccupied) +
                  nann(b, IndexSpace::active_unoccupied) - ncontr_uocc);
  return qns_t{nc_occ, na_occ, nc_uocc, na_uocc};
}

}  // namespace sr
}  // namespace mbpt

mbpt::sr::qns_t adjoint(mbpt::sr::qns_t qns) {
  return mbpt::sr::qns_t{nann(qns, IndexSpace::active_occupied),
                         ncre(qns, IndexSpace::active_occupied),
                         nann(qns, IndexSpace::active_unoccupied),
                         ncre(qns, IndexSpace::active_unoccupied)};
}

namespace mbpt {
namespace sr {

OpMaker::OpMaker(OpType op, std::size_t nbra, std::size_t nket)
    : base_type(op) {
  nket = nket == std::numeric_limits<std::size_t>::max() ? nbra : nket;
  assert(nbra > 0 || nket > 0);

  const auto unocc = IndexSpace::active_unoccupied;
  const auto occ = IndexSpace::active_occupied;
  switch (to_class(op)) {
    case OpClass::ex:
      bra_spaces_ = decltype(bra_spaces_)(nbra, unocc);
      ket_spaces_ = decltype(ket_spaces_)(nket, occ);
      break;
    case OpClass::deex:
      bra_spaces_ = decltype(bra_spaces_)(nbra, occ);
      ket_spaces_ = decltype(ket_spaces_)(nket, unocc);
      break;
    case OpClass::gen:
      bra_spaces_ = decltype(bra_spaces_)(nbra, IndexSpace::complete);
      ket_spaces_ = decltype(ket_spaces_)(nket, IndexSpace::complete);
      break;
  }
}

#include "../mbpt/sr/op.impl.cpp"

ExprPtr H0mp() { return F(); }

ExprPtr H1mp() {
  assert(get_default_context().vacuum() == Vacuum::SingleProduct);
  return H_(2);
}

ExprPtr W() {
  assert(get_default_context().vacuum() == Vacuum::SingleProduct);
  return H1mp();
}

ExprPtr H_(std::size_t k) {
  assert(k > 0 && k <= 2);
  switch (k) {
    case 1:
      switch (get_default_context().vacuum()) {
        case Vacuum::Physical:
          return OpMaker(OpType::h, 1)();
        case Vacuum::SingleProduct:
          return OpMaker(OpType::f, 1)();
        case Vacuum::MultiProduct:
          abort();
        default:
          abort();
      }

    case 2:
      return OpMaker(OpType::g, 2)();

    default:
      abort();
  }
}

ExprPtr H(std::size_t k) {
  assert(k > 0 && k <= 2);
  return k == 1 ? H_(1) : H_(1) + H_(2);
}

ExprPtr F(bool use_f_tensor) {
  if (use_f_tensor) return OpMaker(OpType::f, 1)();

  // add \bar{g}^{\kappa x}_{\lambda y} \gamma^y_x with x,y in occ_space_type
  auto make_g_contribution = [](const auto occ_space_type) {
    return mbpt::OpMaker<Statistics::FermiDirac>::make(
        {IndexSpace::complete}, {IndexSpace::complete},
        [=](auto braidxs, auto ketidxs, Symmetry opsymm) {
          auto m1 = Index::make_tmp_index(
              IndexSpace{occ_space_type, IndexSpace::nullqns});
          auto m2 = Index::make_tmp_index(
              IndexSpace{occ_space_type, IndexSpace::nullqns});
          assert(opsymm == Symmetry::antisymm || opsymm == Symmetry::nonsymm);
          if (opsymm == Symmetry::antisymm) {
            braidxs.push_back(m1);
            ketidxs.push_back(m2);
            return ex<Tensor>(to_wstring(mbpt::OpType::g), braidxs, ketidxs,
                              Symmetry::antisymm) *
                   ex<Tensor>(to_wstring(mbpt::OpType::δ), IndexList{m2},
                              IndexList{m1}, Symmetry::nonsymm);
          } else {  // opsymm == Symmetry::nonsymm
            auto braidx_J = braidxs;
            braidx_J.push_back(m1);
            auto ketidxs_J = ketidxs;
            ketidxs_J.push_back(m2);
            auto braidx_K = braidxs;
            braidx_K.push_back(m1);
            auto ketidxs_K = ketidxs;
            ketidxs_K.emplace(begin(ketidxs_K), m2);
            return (ex<Tensor>(to_wstring(mbpt::OpType::g), braidx_J, ketidxs_J,
                               Symmetry::nonsymm) -
                    ex<Tensor>(to_wstring(mbpt::OpType::g), braidx_K, ketidxs_K,
                               Symmetry::nonsymm)) *
                   ex<Tensor>(to_wstring(mbpt::OpType::δ), IndexList{m2},
                              IndexList{m1}, Symmetry::nonsymm);
          }
        });
  };

  switch (get_default_context().vacuum()) {
    case Vacuum::Physical:
      return OpMaker(OpType::h, 1)() +
             make_g_contribution(IndexSpace::occupied);  // all occupieds
    case Vacuum::SingleProduct:
      return OpMaker(OpType::f, 1)();
    case Vacuum::MultiProduct:
      abort();
    default:
      abort();
  }
}

// Maybe move these to sr/op.impl.cpp
ExprPtr V(std::size_t R) { return OpMaker(OpType::V, R)(); }

ExprPtr pertT1_(std::size_t Nbra, std::size_t Nket) {
  assert(Nbra > 0);
  assert(Nket > 0);
  return OpMaker(OpType::t_1, Nbra, Nket)();
}

ExprPtr pertLambda1_(std::size_t Nbra, std::size_t Nket) {
  assert(Nbra > 0);
  assert(Nket > 0);
  return OpMaker(OpType::λ_1, Nbra, Nket)();
}

ExprPtr vac_av(ExprPtr expr, std::vector<std::pair<int, int>> nop_connections,
               bool use_top) {
  FWickTheorem wick{expr};
  wick.spinfree(false).use_topology(use_top).set_nop_connections(
      nop_connections);
  auto result = wick.compute();
  simplify(result);
  if (Logger::get_instance().wick_stats) {
    std::wcout << "WickTheorem stats: # of contractions attempted = "
               << wick.stats().num_attempted_contractions
               << " # of useful contractions = "
               << wick.stats().num_useful_contractions << std::endl;
  }
  return result;
}

namespace op {

ExprPtr H2_oo_vv() {
  return ex<op_t>(
      []() -> std::wstring_view { return L"g"; },
      [=]() -> ExprPtr {
        using namespace sequant::mbpt::sr;
        return OpMaker(
            OpType::g,
            {IndexSpace::active_occupied, IndexSpace::active_occupied},
            {IndexSpace::active_unoccupied, IndexSpace::active_unoccupied})();
      },
      [=](qnc_t& qns) {
        qns = combine(qnc_t{2, 0, 0, 2}, qns);
      });
}

ExprPtr H2_vv_vv() {
  return ex<op_t>(
      []() -> std::wstring_view { return L"g"; },
      [=]() -> ExprPtr {
        using namespace sequant::mbpt::sr;
        return OpMaker(
            OpType::g,
            {IndexSpace::active_unoccupied, IndexSpace::active_unoccupied},
            {IndexSpace::active_unoccupied, IndexSpace::active_unoccupied})();
      },
      [=](qnc_t& qns) {
        qns = combine(qnc_t{0, 0, 2, 2}, qns);
      });
}

ExprPtr H_(std::size_t k) {
  assert(k > 0 && k <= 2);
  switch (k) {
    case 1:
      return ex<op_t>(
          [vacuum = get_default_context().vacuum()]() -> std::wstring_view {
            switch (vacuum) {
              case Vacuum::Physical:
                return L"h";
              case Vacuum::SingleProduct:
                return L"f";
              case Vacuum::MultiProduct:
                abort();
              default:
                abort();
            }
          },
          [=]() -> ExprPtr {
            using namespace sequant::mbpt::sr;
            return sr::H_(1);
          },
          [=](qnc_t& qns) {
            qns = combine(qnc_t{{0, 1}, {0, 1}, {0, 1}, {0, 1}}, qns);
          });

    case 2:
      return ex<op_t>(
          []() -> std::wstring_view { return L"g"; },
          [=]() -> ExprPtr {
            using namespace sequant::mbpt::sr;
            return sr::H_(2);
          },
          [=](qnc_t& qns) {
            qns = combine(qnc_t{{0, 2}, {0, 2}, {0, 2}, {0, 2}}, qns);
          });

    default:
      abort();
  }
}

ExprPtr H(std::size_t k) {
  assert(k > 0 && k <= 2);
  return k == 1 ? H_(1) : H_(1) + H_(2);
}

ExprPtr T_(std::size_t K) {
  assert(K > 0);
  return ex<op_t>([]() -> std::wstring_view { return L"t"; },
                  [=]() -> ExprPtr {
                    using namespace sequant::mbpt::sr;
                    return sr::T_(K);
                  },
                  [=](qnc_t& qns) {
                    qns = combine(qnc_t{0ul, K, K, 0ul}, qns);
                  });
}

ExprPtr T(std::size_t K) {
  assert(K > 0);

  ExprPtr result;
  for (auto k = 1ul; k <= K; ++k) {
    result = k > 1 ? result + T_(k) : T_(k);
  }
  return result;
}

ExprPtr Lambda_(std::size_t K) {
  assert(K > 0);
  return ex<op_t>([]() -> std::wstring_view { return L"λ"; },
                  [=]() -> ExprPtr {
                    using namespace sequant::mbpt::sr;
                    return sr::Lambda_(K);
                  },
                  [=](qnc_t& qns) {
                    qns = combine(qnc_t{K, 0ul, 0ul, K}, qns);
                  });
}

ExprPtr Lambda(std::size_t K) {
  assert(K > 0);

  ExprPtr result;
  for (auto k = 1ul; k <= K; ++k) {
    result = k > 1 ? result + Lambda_(k) : Lambda_(k);
  }
  return result;
}

ExprPtr A(std::size_t K) {
  assert(K > 0);
  return ex<op_t>([]() -> std::wstring_view { return L"A"; },
                  [=]() -> ExprPtr {
                    using namespace sequant::mbpt::sr;
                    return sr::A(K);
                  },
                  [=](qnc_t& qns) {
                    qns = combine(qnc_t{K, 0ul, 0ul, K}, qns);
                  });
}

ExprPtr V(std::size_t R) {
  return ex<op_t>(
      []() -> std::wstring_view { return L"V"; },
      [=]() -> ExprPtr {
        using namespace sequant::mbpt::sr;
        return sr::V(R);
      },
      [=](qnc_t& qns) {
        qns = combine(qnc_t{{0ul, R}, {0ul, R}, {0ul, R}, {0ul, R}}, qns);
      });
}

ExprPtr pertT1_(std::size_t K) {
  return ex<op_t>([]() -> std::wstring_view { return L"t1"; },
                  [=]() -> ExprPtr {
                    using namespace sequant::mbpt::sr;
                    return sr::pertT1_(K);
                  },
                  [=](qnc_t& qns) {
                    qns = combine(qnc_t{0ul, K, K, 0ul}, qns);
                  });
}

ExprPtr pertT1(std::size_t K) {
  assert(K > 0);

  ExprPtr result;
  for (auto k = 1ul; k <= K; ++k) {
    result = k > 1 ? result + pertT1_(k) : pertT1_(k);
  }
  return result;
}

ExprPtr pertLambda1_(std::size_t K) {
  assert(K > 0);
  return ex<op_t>([]() -> std::wstring_view { return L"λ1"; },
                  [=]() -> ExprPtr {
                    using namespace sequant::mbpt::sr;
                    return sr::pertLambda1_(K);
                  },
                  [=](qnc_t& qns) {
                    qns = combine(qnc_t{K, 0ul, 0ul, K}, qns);
                  });
}

ExprPtr pertLambda1(std::size_t K) {
  assert(K > 0);

  ExprPtr result;
  for (auto k = 1ul; k <= K; ++k) {
    result = k > 1 ? result + pertLambda1_(k) : pertLambda1_(k);
  }
  return result;
}

bool can_change_qns(const ExprPtr& op_or_op_product, const qns_t target_qns,
                    const qns_t source_qns = {}) {
  qns_t qns = source_qns;
  if (op_or_op_product.is<Product>()) {
    const auto& op_product = op_or_op_product.as<Product>();
    for (auto& op_ptr : ranges::views::reverse(op_product.factors())) {
      assert(op_ptr->template is<op_t>());
      const auto& op = op_ptr->template as<op_t>();
      qns = op(qns);
    }
    return qns.overlaps_with(target_qns);
  } else if (op_or_op_product.is<op_t>()) {
    const auto& op = op_or_op_product.as<op_t>();
    qns = op();
    return qns.overlaps_with(target_qns);
  } else
    throw std::invalid_argument(
        "sequant::mbpt::sr::contains_rank(op_or_op_product): op_or_op_product "
        "must be mbpt::sr::op_t or Product thereof");
}

bool raises_vacuum_up_to_rank(const ExprPtr& op_or_op_product,
                              const unsigned long k) {
  assert(op_or_op_product.is<op_t>() || op_or_op_product.is<Product>());
  return can_change_qns(op_or_op_product,
                        qns_t{{0ul, 0ul}, {0ul, k}, {0ul, k}, {0ul, 0ul}});
}

bool lowers_rank_or_lower_to_vacuum(const ExprPtr& op_or_op_product,
                                    const unsigned long k) {
  assert(op_or_op_product.is<op_t>() || op_or_op_product.is<Product>());
  return can_change_qns(op_or_op_product, qns_t{},
                        qns_t{{0ul, 0ul}, {0ul, k}, {0ul, k}, {0ul, 0ul}});
}

bool raises_vacuum_to_rank(const ExprPtr& op_or_op_product,
                           const unsigned long k) {
  assert(op_or_op_product.is<op_t>() || op_or_op_product.is<Product>());
  return can_change_qns(op_or_op_product, qns_t{0ul, k, k, 0ul});
}

bool lowers_rank_to_vacuum(const ExprPtr& op_or_op_product,
                           const unsigned long k) {
  assert(op_or_op_product.is<op_t>() || op_or_op_product.is<Product>());
  return can_change_qns(op_or_op_product, qns_t{}, qns_t{0ul, k, k, 0ul});
}

ExprPtr vac_av(
    ExprPtr expr,
    std::vector<std::pair<std::wstring, std::wstring>> op_connections) {
  // use cloned expr to avoid side effects
  expr = expr->clone();

  auto vac_av_product = [&op_connections](ExprPtr expr) {
    assert(expr.is<Product>());
    // compute connections
    std::vector<std::pair<int, int>> connections;
    {
      std::map<std::wstring, std::vector<int>>
          oplbl2pos;  // maps operator labels to the operator positions in the
                      // product
      int pos = 0;
      for (const auto& factor : expr.as<Product>()) {
        if (factor.is<op_t>()) {
          const auto& op = factor.as<op_t>();
          const std::wstring op_lbl = std::wstring(op.label());
          const auto it = oplbl2pos.find(op_lbl);
          if (it == oplbl2pos.end()) {  // new label
            oplbl2pos.emplace(op_lbl, std::vector<int>{pos});
          } else {
            it->second.emplace_back(pos);
          }
          ++pos;
        } else if (factor.is<FNOperator>() || factor.is<BNOperator>()) {
          ++pos;  // skip FNOperator and BNOperator
        }
      }

      for (const auto& [op1_lbl, op2_lbl] : op_connections) {
        auto it1 = oplbl2pos.find(op1_lbl);
        auto it2 = oplbl2pos.find(op2_lbl);
        if (it1 == oplbl2pos.end() || it2 == oplbl2pos.end())
          continue;  // one of the op labels is not present in the product
        const auto& [dummy1, op1_indices] = *it1;
        const auto& [dummy2, op2_indices] = *it2;
        for (const auto& op1_idx : op1_indices) {
          for (const auto& op2_idx : op2_indices) {
            using std::min;
            using std::max;
            connections.emplace_back(min(op1_idx, op2_idx),
                                     max(op1_idx, op2_idx));
          }
        }
      }
    }

    // lower to tensor form
    auto lower_to_tensor_form = [](ExprPtr& expr) {
      auto op_lowerer = [](ExprPtr& leaf) {
        if (leaf.is<op_t>()) leaf = leaf.as<op_t>().tensor_form();
      };
      expr->visit(op_lowerer, /* atoms only = */ true);
    };
    lower_to_tensor_form(expr);
    expr = simplify(expr);

    // compute VEV
    return mbpt::sr::vac_av(expr, connections, /* use_topology = */ true);
  };

  ExprPtr result;
  if (expr.is<Product>()) {
    return vac_av_product(expr);
  } else if (expr.is<Sum>()) {
    result = sequant::transform_reduce(
        *expr, ex<Sum>(),
        [](const ExprPtr& running_total, const ExprPtr& summand) {
          return running_total + summand;
        },
        [&op_connections](const auto& op_product) {
          return vac_av(op_product, op_connections);
        });
    return result;
  } else if (expr.is<op_t>()) {
    return ex<Constant>(
        0);  // expectation value of a normal-ordered operator is 0
  } else if (expr.is<Constant>()) {
    return expr;  // vacuum is normalized
  }
  throw std::invalid_argument(
      "mpbt::sr::op::vac_av(expr): unknown expression type");
}

}  // namespace op

}  // namespace sr

// must be defined including op.ipp since it's used there
template <>
bool is_vacuum<sr::qns_t>(sr::qns_t qns) {
  return qns == sr::qns_t{};
}

}  // namespace mbpt

template <Statistics S>
std::wstring to_latex(const mbpt::Operator<mbpt::sr::qns_t, S>& op) {
  using namespace sequant::mbpt;
  using namespace sequant::mbpt::sr;

  auto result = L"{\\hat{" + utf_to_latex(op.label()) + L"}";
  auto it = label2optype.find(std::wstring(op.label()));
  OpType optype = OpType::invalid;
  if (it != label2optype.end()) {  // handle special cases
    optype = it->second;
    if (to_class(optype) == OpClass::gen) {
      result += L"}";
      return result;
    }
  }

  // generic operator ... can only handle definite case
  const auto dN = op();
  if (!is_definite(ncre_occ(dN)) || !is_definite(nann_occ(dN)) ||
      !is_definite(ncre_uocc(dN)) || !is_definite(nann_uocc(dN))) {
    throw std::invalid_argument(
        "to_latex(const Operator<qns_t, S>& op): "
        "can only handle  generic operators with definite cre/ann numbers");
  }
  // pure quasiparticle creator/annihilator?
  const auto qprank_cre = nann_occ(dN).lower() + ncre_uocc(dN).lower();
  const auto qprank_ann = ncre_occ(dN).lower() + nann_uocc(dN).lower();
  const auto qppure = qprank_cre == 0 || qprank_ann == 0;
  auto qpaction = to_class(optype);
  if (qppure) {
    if (qprank_cre) {
      // if operator's action implied by the label and actual action agrees, use
      // subscript always
      std::wstring baseline_char = (qpaction != OpClass::deex ? L"_" : L"^");
      if (nann_occ(dN).lower() == ncre_uocc(dN).lower())
        result +=
            baseline_char + L"{" + std::to_wstring(nann_occ(dN).lower()) + L"}";
      else
        result += baseline_char + L"{" + std::to_wstring(nann_occ(dN).lower()) +
                  L"," + std::to_wstring(ncre_uocc(dN).lower()) + L"}";
    } else {
      // if operator's action implied by the label and actual action agrees, use
      // subscript always
      std::wstring baseline_char = (qpaction != OpClass::deex ? L"^" : L"_");
      if (nann_uocc(dN).lower() == ncre_occ(dN).lower()) {
        result +=
            baseline_char + L"{" + std::to_wstring(ncre_occ(dN).lower()) + L"}";
      } else
        result += baseline_char + L"{" + std::to_wstring(ncre_occ(dN).lower()) +
                  L"," + std::to_wstring(nann_uocc(dN).lower()) + L"}";
    }
  } else {  // not pure qp creator/annihilator
    result += L"_{" + std::to_wstring(nann_occ(dN).lower()) + L"," +
              std::to_wstring(ncre_uocc(dN).lower()) + L"}^{" +
              std::to_wstring(ncre_occ(dN).lower()) + L"," +
              std::to_wstring(nann_uocc(dN).lower()) + L"}";
  }
  result += L"}";
  return result;
}

}  // namespace sequant

#include "SeQuant/domain/mbpt/op.ipp"

namespace sequant {
namespace mbpt {
template class Operator<sr::qns_t, Statistics::FermiDirac>;
template class Operator<sr::qns_t, Statistics::BoseEinstein>;
}  // namespace mbpt
}  // namespace sequant
