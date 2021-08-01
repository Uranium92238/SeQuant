//
// Created by Bimal Gaudel on 7/31/21.
//

#ifndef SEQUANT_EVAL_SCF_BTAS_HPP
#define SEQUANT_EVAL_SCF_BTAS_HPP

#include <btas/btas.h>
#include <memory>

#include <SeQuant/core/container.hpp>
#include <SeQuant/core/eval_node.hpp>
#include <SeQuant/core/parse_expr.hpp>
#include <SeQuant/domain/eval/cache_manager.hpp>
#include <SeQuant/domain/eval/eval_btas.hpp>

#include "examples/eval/btas/data_world_btas.hpp"
#include "examples/eval/calc_info.hpp"
#include "examples/eval/data_info.hpp"
#include "examples/eval/scf.hpp"

namespace sequant::eval::btas {

template <typename Tensor_t>
class SequantEvalScfBTAS final : public SequantEvalScf {
 private:
  container::vector<EvalNode> nodes_;
  CacheManager<Tensor_t const> cman_;
  DataWorldBTAS<Tensor_t> data_world_;

  Tensor_t const& f_vo() const {
    static Tensor_t tnsr =
        data_world_(parse_expr(L"f{a1;i1}", Symmetry::nonsymm)->as<Tensor>());
    return tnsr;
  }

  Tensor_t const& g_vvoo() const {
    static Tensor_t tnsr = data_world_(
        parse_expr(L"g{a1,a2;i1,i2}", Symmetry::nonsymm)->as<Tensor>());
    return tnsr;
  }

  double energy_spin_orbital() const {
    auto const& T1 = data_world_.amplitude(1);
    auto const& T2 = data_world_.amplitude(2);
    auto const& G_vvoo = g_vvoo();
    auto const& F_vo = f_vo();

    Tensor_t temp;
    ::btas::contract(1.0, G_vvoo, {'a', 'b', 'i', 'j'},  //
                     T1, {'a', 'i'},                     //
                     0.0, temp, {'b', 'j'});
    return 0.5 * ::btas::dot(temp, T1) + 0.25 * ::btas::dot(G_vvoo, T2) +
           ::btas::dot(F_vo, T1);
  }

  double energy_spin_free_orbital() const {
    auto const& T1 = data_world_.amplitude(1);
    auto const& T2 = data_world_.amplitude(2);
    auto const& G_vvoo = g_vvoo();
    auto const& F_vo = f_vo();

    Tensor_t temp;
    ::btas::contract(1.0, T1, {'a', 'i'}, T1, {'b', 'j'}, 0.0, temp,
                     {'a', 'b', 'i', 'j'});

    return 2.0 * (::btas::dot(F_vo, T1) + ::btas::dot(G_vvoo, T2) +
                  ::btas::dot(G_vvoo, temp)) -
           ::btas::dot(G_vvoo, temp);
  }

  void reset_cache_decaying() override { cman_.reset_decaying(); }

  void reset_cache_all() override { cman_.reset_all(); }

  double norm() const override {
    // todo use all Ts instead of only T2
    auto const& T2 = data_world_.amplitude(2);
    return std::sqrt(::btas::dot(T2, T2));
  }

  double solve() override {
    auto rs = ranges::views::repeat_n(Tensor_t{}, info_.eqn_opts.excit) |
              ranges::to_vector;
    for (auto&& [r, n] : ranges::views::zip(rs, nodes_))
      r = info_.eqn_opts.spintrace
              ? eval::btas::eval_symm(n, data_world_, cman_)
              : eval::btas::eval_antisymm(n, data_world_, cman_);
    data_world_.update_amplitudes(rs);
    return info_.eqn_opts.spintrace ? energy_spin_free_orbital()
                                    : energy_spin_orbital();
  }

 public:
  SequantEvalScfBTAS(CalcInfo const& calc_info)
      : SequantEvalScf{calc_info},
        cman_{{}, {}},
        data_world_{calc_info.eqn_opts.excit, calc_info.fock_eri} {
    assert(info_.eqn_opts.excit >= 2 &&
           "At least double excitation (CCSD) is required!");
    // todo time it
    auto const exprs = info_.exprs();

    // todo time it
    nodes_ = info_.nodes(exprs);

    cman_ = info_.cache_manager<Tensor_t const>(nodes_);
  }
};

}  // namespace sequant::eval::btas

#endif  // SEQUANT_EVAL_SCF_BTAS_HPP
