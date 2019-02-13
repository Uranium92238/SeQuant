//
// Created by Eduard Valeyev on 3/27/18.
//

#ifndef SEQUANT2_CONTAINER_HPP
#define SEQUANT2_CONTAINER_HPP

#include <boost/container/flat_map.hpp>
#include <boost/container/flat_set.hpp>
#include <boost/container/small_vector.hpp>
#include <boost/range.hpp>
#include <map>
#include <set>
#include <vector>

namespace sequant2 {

namespace container {

template<typename T> using vector = std::vector<T>;
template <typename T, std::size_t N = 8>
using svector = boost::container::small_vector<T, N>;

template<typename Key, typename Compare = std::less<Key>> using set = boost::container::flat_set<Key,Compare>;
template<typename Key, typename Value, typename Compare = std::less<Key>> using map = boost::container::flat_map<Key,Value,Compare>;

using boost::begin;
using boost::end;
}  // namespace sequant2::container

}  // namespace sequant2

#endif //SEQUANT2_CONTAINER_HPP