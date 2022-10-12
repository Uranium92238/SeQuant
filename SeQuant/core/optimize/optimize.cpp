#include <SeQuant/core/optimize.hpp>
#include <SeQuant/core/clone_packed.hpp>

namespace sequant {

namespace opt {

namespace detail {
std::vector<size_t> on_bits_pos(size_t n, size_t num_bits) {
  auto result = std::vector<size_t>{};
  result.reserve(num_bits);
  for (auto i = 0; i < num_bits; ++i)
    if (n & (1 << i)) result.push_back(i);
  return result;
}
}  // namespace detail

ExprPtr tail_factor(ExprPtr const& expr) noexcept {
  if (expr->is<Tensor>())
    return expr->clone();

  else if (expr->is<Product>()) {
    auto scalar = expr->as<Product>().scalar();
    auto facs = ranges::views::tail(*expr);
    return ex<Product>(Product{scalar, ranges::begin(facs), ranges::end(facs)});
  } else {
    // sum
    auto summands = *expr | ranges::views::transform(
                                [](auto const& x) { return tail_factor(x); });
    return ex<Sum>(Sum{ranges::begin(summands), ranges::end(summands)});
  }
}

void pull_scalar(ExprPtr expr) noexcept {
  if (!expr->is<Product>()) return;
  auto& prod = expr->as<Product>();

  auto scal = prod.scalar();
  for (auto&& x : *expr)
    if (x->is<Product>()) {
      auto& p = x->as<Product>();
      scal *= p.scalar();
      p.scale(1.0 / p.scalar());
    }

  prod.scale(1.0 / prod.scalar());
  prod.scale(scal);
}

}  // namespace opt

}  // namespace sequant