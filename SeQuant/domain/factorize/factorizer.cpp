#include "factorizer.hpp"

#include <SeQuant/core/container.hpp>
#include <SeQuant/core/expr.hpp>
#include <SeQuant/core/tensor_network.hpp>
// IndexSpace type based hashing of tensors for AdjacencyMatrix
#include <SeQuant/domain/evaluate/eval_tree.hpp>

#include <iomanip>
#include <ios>
#include <tuple>

#define DEBUG_PRINT 0

namespace sequant::factorize {

using pos_type = AdjacencyMatrix::pos_type;
using color_mat_type = AdjacencyMatrix::color_mat_type;

AdjacencyMatrix::AdjacencyMatrix(
    const ExprPtr& expr,
    const TensorNetwork::named_indices_t& external_indices)
    : colorMatrix_(  // allocates color matrix
          expr->size(),
          color_mat_type::value_type(
              expr->size(), color_mat_type::value_type::value_type{})) {
  // get a Product out of a TensorNetwork
  auto tn_to_prod = [](const auto& tn) {
    auto prod = std::make_shared<Product>();
    for (const auto& tnsr : tn.tensors())
      prod->append(1, std::dynamic_pointer_cast<Tensor>(tnsr));
    return prod;
  };

  // fill the color values
  for (auto ii = 0; ii < expr->size(); ++ii)
    for (auto jj = ii + 1; jj < expr->size(); ++jj) {
      // set color data
      if (are_connected(expr->at(ii), expr->at(jj))) {
        // auto prod = std::make_shared<Product>(Product{expr->at(ii),
        // expr->at(jj)});
        auto tnet =
            TensorNetwork(*(expr->at(ii)->clone() * expr->at(jj)->clone()));
        tnet.canonicalize(TensorCanonicalizer::cardinal_tensor_labels(), false,
                          &external_indices);
        colorMatrix_[ii][jj] = colorMatrix_[jj][ii] =
            evaluate::EvalTree(tn_to_prod(tnet)).hash_value();
      }
    }
}

AdjacencyMatrix::AdjacencyMatrix(
    const container::svector<ExprPtr>& tensors,
    const TensorNetwork::named_indices_t& external_indices)
    : AdjacencyMatrix(
          std::make_shared<Product>(1, tensors.begin(), tensors.end()),
          external_indices) {}

bool AdjacencyMatrix::are_connected(const ExprPtr& t1, const ExprPtr& t2) {
  auto tnsr1 = t1->as<Tensor>();
  auto tnsr2 = t2->as<Tensor>();
  // iterate through the bra and the ket labels of tnsr1 and tnsr2
  // if any index label is common, they are connected.
  for (const auto& idx1 : tnsr1.const_braket())
    for (const auto& idx2 : tnsr2.const_braket()) {
      if (idx1.label() == idx2.label()) return true;
    }
  return false;
}

const color_mat_type& AdjacencyMatrix::color_mat() const {
  return colorMatrix_;
}

size_t AdjacencyMatrix::num_verts() const { return color_mat().size(); }

bool AdjacencyMatrix::are_connected(pos_type pos1, pos_type pos2) const {
  return color_mat()[pos1][pos2] != color_mat_type::value_type::value_type{};
}

color_mat_type::value_type::value_type AdjacencyMatrix::color(
    pos_type pos1, pos_type pos2) const {
  return color_mat()[pos1][pos2];
}

// expr is Product type
// tnsr is Tensor type
bool tensor_exists(const ExprPtr& expr, const ExprPtr& tnsr) {
  if (expr->is<Tensor>()) {
    return *expr == *tnsr;
  }

  for (const auto& xpr : *expr)
    if (tensor_exists(xpr, tnsr)) return true;
  return false;
};

/***********************************
 * Functions for factorization     *
 ***********************************/

// get target indices
TensorNetwork::named_indices_t target_indices(const ExprPtr& expr) {
  TensorNetwork::named_indices_t result;
  for (const auto& tnsr : *expr) {
    for (const auto& idx : tnsr->as<Tensor>().const_braket()) {
      if (result.contains(idx))
        result.erase(idx);
      else
        result.insert(idx);
    }
  }
  return result;
}

// Get positions of common type of tensors in a pair of Exprs'.
std::tuple<container::set<pos_type>, container::set<pos_type>> common_tensors(
    const ExprPtr& expr1, const ExprPtr& expr2) {
  container::set<pos_type> commonT_1, commonT_2;
  for (pos_type ii = 0; ii < expr1->size(); ++ii)
    for (pos_type jj = 0; jj < expr2->size(); ++jj) {
      //
      // NOTE: As an example: t_{i j}^{a b} has the same
      // hash value as t_{a b}^{i j}. To hash such expressions
      // differently, use EvalTree(expr, false).
      //
      if (evaluate::EvalTree(expr1->at(ii)).hash_value() ==
          evaluate::EvalTree(expr2->at(jj)).hash_value()) {
        commonT_1.insert(ii);
        commonT_2.insert(jj);
      }
    }
  return std::tuple(commonT_1, commonT_2);
}

container::map<std::tuple<pos_type, pos_type>, std::tuple<pos_type, pos_type>>
common_pairs(const AdjacencyMatrix& mat1, const AdjacencyMatrix& mat2) {
  auto result = container::map<std::tuple<pos_type, pos_type>,
                               std::tuple<pos_type, pos_type>>{};
  for (auto ii = 0; ii < mat1.num_verts(); ++ii)
    for (auto jj = ii + 1; jj < mat1.num_verts(); ++jj)
      for (auto kk = 0; kk < mat2.num_verts(); ++kk)
        for (auto ll = kk + 1; ll < mat2.num_verts(); ++ll)
          if (auto color = mat1.color(ii, jj);
              color != color_mat_type::value_type::value_type{} &&
              color == mat2.color(kk, ll)) {
            result.insert(decltype(result)::value_type{std::tuple(ii, jj),
                                                       std::tuple(kk, ll)});
            break;
          }

  return result;
}

std::tuple<container::svector<container::set<AdjacencyMatrix::pos_type>>,
           container::svector<container::set<AdjacencyMatrix::pos_type>>>
common_nets(const container::map<std::tuple<pos_type, pos_type>,
                                 std::tuple<pos_type, pos_type>>& pairs) {
  // make a copy of pairs
  auto common_p = pairs;
  // to hold processed nets
  container::svector<container::set<AdjacencyMatrix::pos_type>> common_net1,
      common_net2;
  decltype(common_p)::iterator pair_iter;
  while (!common_p.empty()) {
    pair_iter = common_p.begin();

    decltype(common_net1)::value_type net1, net2;

    net1.insert(std::get<0>(pair_iter->first));
    net1.insert(std::get<1>(pair_iter->first));

    net2.insert(std::get<0>(pair_iter->second));
    net2.insert(std::get<1>(pair_iter->second));

    pair_iter = common_p.erase(pair_iter);
    while (pair_iter != common_p.end()) {
      auto [pFirst1, pFirst2] = pair_iter->first;
      auto [pSecond1, pSecond2] = pair_iter->second;

      if (net1.contains(pFirst1) || net1.contains(pFirst2)) {
        assert(net2.contains(pSecond1) || net2.contains(pSecond2));
        pair_iter = common_p.erase(pair_iter);
      } else {
        assert(!(net2.contains(pSecond1) || net2.contains(pSecond2)));
        ++pair_iter;
      }
    }
    common_net1.push_back(net1);
    common_net2.push_back(net2);
  }
  return std::tuple(common_net1, common_net2);
}

std::tuple<ExprPtr, ExprPtr> factorize_pair(const ExprPtr& expr1,
                                             const ExprPtr& expr2) {
  // get common type of tensor's positions
  auto [commonIdx1, commonIdx2] = common_tensors(expr1, expr2);

  container::svector<ExprPtr> commonExpr1, commonExpr2;
  for (auto idx : commonIdx1) {
    commonExpr1.push_back(expr1->at(idx));
  }
  for (auto idx : commonIdx2) {
    commonExpr2.push_back(expr2->at(idx));
  }

#ifndef DEBUG_PRINT
  std::wcout << "\nprinting common tensors\n"
             << "-----------------------\n";
  for (const auto& xpr : commonExpr1) std::wcout << xpr->to_latex() << " ";
  std::wcout << "\n";
  for (const auto& xpr : commonExpr2) std::wcout << xpr->to_latex() << " ";
  std::wcout << "\n";
#endif

  // get the target indices
  auto target1 = target_indices(expr1);
  auto target2 = target_indices(expr2);

#ifndef DEBUG_PRINT
  std::wcout << "\nprinting target indices\n"
             << "-----------------------\n";
  for (const auto& idx : target1) std::wcout << idx.to_latex() << " ";
  std::wcout << "\n";
  for (const auto& idx : target2) std::wcout << idx.to_latex() << " ";
  std::wcout << "\n";
#endif

  // form adjacency matrices for common tensors
  auto adjMat1 = AdjacencyMatrix(commonExpr1, target1);
  auto adjMat2 = AdjacencyMatrix(commonExpr2, target2);

  auto print_adj_mat = [](const auto& mat, bool color = false) {
    for (auto ii = 0; ii < mat.num_verts(); ++ii) {
      for (auto jj = 0; jj < mat.num_verts(); ++jj) {
        if (color)
          std::wcout << std::setw(27) << mat.color(ii, jj);
        else
          std::wcout << mat.are_connected(ii, jj);
        std::wcout << "  ";
      }
      std::wcout << "\n";
    }
  };

#ifndef DEBUG_PRINT
  std::wcout << "\nprinting adjacency matrix1\n"
             << "--------------------------\n";
  print_adj_mat(adjMat1, true);
  std::wcout << "\nprinting adjacency matrix2\n"
             << "--------------------------\n";
  print_adj_mat(adjMat2, true);
#endif

  // find common pairs
  auto common_p = common_pairs(adjMat1, adjMat2);

#ifndef DEBUG_PRINT
  std::wcout << "\nprinting common pair indices\n"
             << "----------------------------\n";
  for (const auto& p : common_p) {
    auto [tup1, tup2] = p;
    std::wcout << "(" << std::get<0>(tup1) << ", " << std::get<1>(tup1)
               << ")  ";
    std::wcout << "(" << std::get<0>(tup2) << ", " << std::get<1>(tup2) << ")";
    std::wcout << "\n";
  }
  std::wcout << std::endl;
#endif

  // find common nets
  auto [netIdx1, netIdx2] = common_nets(common_p);

  // form selectivev ExprPtr out of an iterable of
  // pos_types and a given reference Expr
  auto iter_to_expr = [](const auto& iterable, const auto& expr) {
    auto result = std::make_shared<Product>();
    for (auto idx : iterable) result->append(1, expr.at(idx));
    return result;
  };

  auto subnet1 = std::make_shared<Product>();
  for (const auto& group : netIdx1)
    subnet1->append(1, iter_to_expr(group, commonExpr1));

  auto subnet2 = std::make_shared<Product>();
  for (const auto& group : netIdx2)
    subnet2->append(1, iter_to_expr(group, commonExpr2));

  // append left out tensors from the original expr to individual subnets
  auto get_left_out = [](const auto& commonidx, const ExprPtr& factorized,
                         const ExprPtr& original) {
    auto left_out = std::make_shared<Product>();

    for (pos_type ii = 0; ii < original->size(); ++ii)
      if (auto tnsr = original->at(ii);
          !(tensor_exists(factorized, tnsr)))
        left_out->append(1, tnsr->clone());
    return left_out;
  };

  auto left1 = get_left_out(commonIdx1, subnet1, expr1);
  auto left2 = get_left_out(commonIdx2, subnet2, expr2);

#ifndef DEBUG_PRINT
  std::wcout << "subnet1 = " << subnet1->to_latex() << "\n"
             << "subnet2 = " << subnet2->to_latex() << "\n"
             << "left1   = " << left1->to_latex() << "\n"
             << "left2   = " << left2->to_latex() << "\n";
#endif

  auto combine_expr = [](const ExprPtr& exprA, const ExprPtr& exprB) {
    if (exprA->size() == 0) return exprB->clone();
    if (exprB->size() == 0) return exprA->clone();
    auto combined = std::make_shared<Product>();
    combined->append(exprA->clone());
    combined->append(exprB->clone());
    return std::dynamic_pointer_cast<Expr>(combined);
  };

  auto factorForm1 = combine_expr(subnet1, left1);
  auto factorForm2 = combine_expr(subnet2, left2);

  return std::tuple(factorForm1, factorForm2);
}

}  // namespace sequant::factorize
