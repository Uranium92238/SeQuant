//
// Created by Eduard Valeyev on 5/17/23.
//

#ifndef SEQUANT_CORE_MATH_HPP
#define SEQUANT_CORE_MATH_HPP

#include <assert.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <cstddef>

namespace sequant {

constexpr std::size_t pow2(std::size_t n) {
  assert(n <= 63);
  return 1ul << n;
}

using int128_t = boost::multiprecision::int128_t;
constexpr inline int128_t factorial(std::size_t n) {
  assert(n >= 0);
  const int128_t values[] = {1,
                             1,
                             2,
                             6,
                             24,
                             120,
                             720,
                             5040,
                             40320,
                             362880,
                             3628800,
                             39916800,
                             479001600,
                             6227020800,
                             87178291200,
                             1307674368000,
                             20922789888000,
                             355687428096000,
                             6402373705728000,
                             121645100408832000,
                             2432902008176640000};
  if (n < 20)
    return values[n];
  else {
    auto fac = [](std::size_t n) {
      int128_t result = 1;
      for (auto i = 2; i <= n; ++i) {
        result *= i;
      }
      return result;
    };
    return fac(n);
  }
}

}  // namespace sequant

#endif  // SEQUANT_CORE_MATH_HPP
