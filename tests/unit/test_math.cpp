//
// Created by Eduard Valeyev on 5/18/23.
//

#include "catch.hpp"

#include <iostream>

#include "SeQuant/core/rational.hpp"
#include "SeQuant/core/wstring.hpp"

TEST_CASE("Rational", "[elements]") {
  using namespace sequant;

  auto print = [](rational r) {
    return sequant::to_wstring(r.numerator()) + L"/" +
           sequant::to_wstring(r.denominator());
  };
  SECTION("to_rational") {
    REQUIRE(to_rational(1. / 3) == rational{1, 3});
    REQUIRE(to_rational(1. / 7) == rational{1, 7});
    REQUIRE(to_rational(M_PI) == rational{99023, 31520});
    REQUIRE(to_rational(M_E) == rational{23225, 8544});
    REQUIRE_THROWS_AS(to_rational(std::nan("NaN")), std::invalid_argument);
  }

}  // TEST_CASE("Rational")
