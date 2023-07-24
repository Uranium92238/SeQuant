//
// Created by Eduard Valeyev on 3/23/18.
//

#ifndef SEQUANT_WICK_HPP
#define SEQUANT_WICK_HPP

#include <bitset>
#include <mutex>
#include <utility>

#include "SeQuant/core/math.hpp"
#include "op.hpp"
#include "ranges.hpp"
#include "runtime.hpp"
#include "tensor.hpp"

namespace sequant {

/// Applies Wick's theorem to a sequence of normal-ordered operators.
///
/// @tparam S particle statistics
template <Statistics S>
class WickTheorem {
 public:
  // see
  // https://softwareengineering.stackexchange.com/questions/257705/unit-test-private-method-in-c-using-a-friend-class?answertab=votes#tab-top
  template <class T>
  struct access_by;
  template <class T>
  friend struct access_by;

  static constexpr const Statistics statistics = S;

  WickTheorem(WickTheorem &&) = default;
  WickTheorem &operator=(WickTheorem &&) = default;

 private:
  // make copying private to be able to adjust state
  WickTheorem(const WickTheorem &) = default;
  WickTheorem &operator=(const WickTheorem &) = default;

 public:
  explicit WickTheorem(const NormalOperatorSequence<S> &input) {
    init_input(input);
    assert(input.size() <= max_input_size);
    assert(input.empty() || input.vacuum() != Vacuum::Invalid);
    assert(input.empty() || input.vacuum() != Vacuum::Invalid);
    if constexpr (statistics == Statistics::BoseEinstein) {
      assert(input.empty() || input.vacuum() == Vacuum::Physical);
    }
  }

  explicit WickTheorem(ExprPtr expr_input) : expr_input_(expr_input) {}

  /// constructs WickTheorem from @c other with expression input set to @c
  /// expr_input
  WickTheorem(ExprPtr expr_input, const WickTheorem &other)
      : WickTheorem(other) {
    input_ = {};  // reset input_ so that it is deduced from expr_input_
    // copy ctor does not do anything useful, so this is OK
    expr_input_ = expr_input;
    reset_stats();
  }

  /// Controls whether next call to compute() will full contractions only or all
  /// (including partial) contractions. By default compute() generates full
  /// contractions only.
  /// @param sf if false, will evaluate all contractions.
  /// @return reference to @c *this , for daisy-chaining
  WickTheorem &full_contractions(bool fc) {
    full_contractions_ = fc;
    return *this;
  }
  /// Controls whether next call to compute() will assume spin-free or
  /// spin-orbital normal-ordered operators By default compute() assumes
  /// spin-orbital operators.
  /// @param sf if true, will complete full contractions only.
  WickTheorem &spinfree(bool sf) {
    spinfree_ = sf;
    return *this;
  }
  /// Controls whether:
  /// - Op's of the same type within each NormalOperator are
  /// assumed topologically equivalent, and
  /// - (<b>if an Expr is given as input</b>) NormalOperator objects attached to
  /// Tensor objects are
  ///   considered equivalent. (If a NormalOperatorSequence is given as input,
  ///   such use of topological equivalence if enabled by invoking
  ///   set_nop_partitions() ).
  ///
  /// This is useful to to eliminate the topologically-equivalent contractions
  /// when fully-contracted result (i.e. the vacuum average) is sought.
  /// By default the use of topology is not enabled.
  /// @param sf if true, will utilize the topology to minimize work.
  WickTheorem &use_topology(bool ut) {
    use_topology_ = ut;
    return *this;
  }

  /// Specifies the external indices; by default assume all indices are summed
  /// over
  /// @param ext_inds external (nonsummed) indices
  template <typename IndexContainer>
  WickTheorem &set_external_indices(IndexContainer &&external_indices) {
    if constexpr (std::is_convertible_v<IndexContainer,
                                        decltype(external_indices_)>)
      external_indices_ = std::forward<IndexContainer>(external_indices);
    else {
      ranges::for_each(std::forward<IndexContainer>(external_indices),
                       [this](auto &&v) {
                         auto result = this->external_indices_.emplace(v);
                         assert(result.second);
                       });
    }
    return *this;
  }

  /// @name operator connectivity specifiers
  ///
  /// Ensures that the given pairs of normal operators are connected; by default
  /// will not constrain connectivity
  /// @param op_index_pairs the list of pairs of op indices to be connected in
  /// the result
  /// @throw std::invalid_argument if @p op_index_pairs contains duplicates
  ///@{

  /// @tparam IndexPairContainer a sequence of std::pair<Integer,Integer>
  template <typename IndexPairContainer>
  WickTheorem &set_nop_connections(IndexPairContainer &&op_index_pairs) {
    auto has_duplicates = [](const auto &op_index_pairs) {
      const auto the_end = end(op_index_pairs);
      for (auto it = begin(op_index_pairs); it != the_end; ++it) {
        const auto found_dup_it = std::find(it + 1, the_end, *it);
        if (found_dup_it != the_end) {
          return true;
        }
      }
      return false;
    };
    if (has_duplicates(op_index_pairs)) {
      throw std::invalid_argument(
          "WickTheorem::set_nop_connections(arg): arg contains duplicates");
    }

    if (expr_input_ == nullptr || !nop_connections_input_.empty()) {
      for (const auto &opidx_pair : op_index_pairs) {
        if (opidx_pair.first < 0 || opidx_pair.first >= input_.size()) {
          throw std::invalid_argument(
              "WickTheorem::set_nop_connections: nop index out of range");
        }
        if (opidx_pair.second < 0 || opidx_pair.second >= input_.size()) {
          throw std::invalid_argument(
              "WickTheorem::set_nop_connections: nop index out of range");
        }
      }
      if (op_index_pairs.size() != 0) {
        nop_connections_.resize(input_.size());
        for (auto &v : nop_connections_) {
          v.set();
        }
        for (const auto &opidx_pair : op_index_pairs) {
          nop_connections_[opidx_pair.first].reset(opidx_pair.second);
          nop_connections_[opidx_pair.second].reset(opidx_pair.first);
        }
      }
      nop_nconnections_total_ = nop_connections_input_.size();
      nop_connections_input_.clear();
    } else {
      ranges::for_each(op_index_pairs, [this](const auto &idxpair) {
        nop_connections_input_.push_back(idxpair);
      });
    }

    return *this;
  }

  /// @tparam Integer an integral type
  template <typename Integer = long>
  WickTheorem &set_nop_connections(
      std::initializer_list<std::pair<Integer, Integer>> op_index_pairs) {
    return this->set_nop_connections<const decltype(op_index_pairs) &>(
        op_index_pairs);
  }
  ///@}

  /// @name specifiers of partitions composed of topologically-equivalent
  ///       normal operators
  ///
  /// Specifies topological partition of normal operators; free (non-connected)
  /// operators in the same partition are considered topologically equivalent,
  /// hence if only full contractions are needed only contractions to the first
  /// available operator in a partition is needed (multiplied by the degeneracy)
  /// @param nop_partitions list of normal operator partitions
  /// @note if this partitions are not given, every operator is assumed to be in
  /// its own partition
  /// @internal this performs only the first phase of initialization of
  /// nop_topological_partition_
  ///           since the number of operators is not guaranteed to be known
  ///           until compute() is called (e.g. if op product is given as an
  ///           Expr ). The second initialization phase (i.e. resizing to make
  ///           sure every operator is assigned to a partition, including those
  ///           not mentioned in @c op_partitions) will be completed in
  ///           compute_nopseq().
  ///
  ///@{

  /// @tparam IndexListContainer a sequence of sequences of Integer types
  template <typename IndexListContainer>
  auto &set_nop_partitions(IndexListContainer &&nop_partitions) const {
    using std::size;
    size_t partition_cnt = 0;
    auto current_nops = size(nop_partition_idx_);
    for (auto &&partition : nop_partitions) {
      for (auto &&nop_ord : partition) {
        assert(nop_ord >= 0);
        if (nop_ord >= current_nops) {
          current_nops = upsize_topological_partitions(
              nop_ord + 1, TopologicalPartitionType::NormalOperator);
        }
        nop_partition_idx_[nop_ord] = partition_cnt + 1;
      }
      ++partition_cnt;
    }
    nop_npartitions_ = size(nop_partitions);
    assert(partition_cnt == nop_npartitions_);
    return *this;
  }

  /// @tparam Integer an integral type
  template <typename Integer = long>
  auto &set_nop_partitions(std::initializer_list<std::initializer_list<Integer>>
                               nop_partitions) const {
    return this->set_nop_partitions<const decltype(nop_partitions) &>(
        nop_partitions);
  }
  ///@}

  /// @name specifiers of partitions composed of topologically-equivalent
  ///       normal operators
  ///
  /// Specifies sets of topologically-equivalent indices that can be used to
  /// produce topologically-unique Wick contractions
  /// @param op_partitions list of index partitions
  /// @note if this partitions are not given, every Index is assumed to be in
  /// its own partition
  ///
  ///@{

  /// @tparam IndexListContainer a sequence of sequences of Integer types
  template <typename IndexListContainer>
  auto &set_op_partitions(IndexListContainer &&op_partitions) const {
    using std::size;
    // assert that we don't have empty partitions due to broken logic upstream
    assert(ranges::any_of(op_partitions, [](auto &&partition) {
             return size(partition) == 0;
           }) == false);

    // every op needs to be in a partition AND partitions need to be sorted
    // if op_partitions only specifues nontrivial partitions, may need to add
    // more partitions create initial set of partitions, then update if needed
    op_npartitions_ = size(op_partitions);
    op_partitions_.resize(op_npartitions_);
    auto partition_idx = 0;
    ranges::for_each(op_partitions, [&](auto &&op_partition) {
      ranges::for_each(op_partition, [&](auto &&idx) {
        op_partitions_[partition_idx].emplace(idx);
      });
      partition_idx++;
    });
    assert(ranges::is_sorted(op_partitions_, [](const auto &x, const auto &y) {
      return *(x.begin()) < *(y.begin());
    }));

    bool done = false;
    while (!done) {
      op_partition_idx_.resize(input_.opsize());
      ranges::fill(op_partition_idx_, 0);
      for (auto &&[partition_idx, partition] :
           ranges::views::enumerate(op_partitions_)) {
        for (auto &&op_ord : partition) {
          assert(op_ord >= 0);
          assert(op_ord < input_.opsize());
          assert(op_partition_idx_[op_ord] == 0);
          op_partition_idx_[op_ord] = partition_idx + 1;
        }
      }

      // assert that every op is in a partition
      if (ranges::contains(op_partition_idx_, 0)) {
        std::size_t idx_ord = 0;
        for (auto &&[ord, partition_idx] :
             ranges::views::enumerate(op_partition_idx_)) {
          if (partition_idx == 0) {
            op_partitions_.emplace_back(container::set<std::size_t>{ord});
          }
        }
        ranges::sort(op_partitions_, [](const auto &x, const auto &y) {
          return *(x.begin()) < *(y.begin());
        });
        op_npartitions_ = size(op_partitions_);
      } else {
        done = true;
      }
    }

    return *this;
  }

  /// @tparam Integer an integral type
  template <typename Integer = long>
  auto &set_op_partitions(std::initializer_list<std::initializer_list<Integer>>
                              op_partitions) const {
    return this->set_op_partitions<const decltype(op_partitions) &>(
        op_partitions);
  }

  /// makes a default set of partitions with each Op is in its own partition
  auto &make_default_op_partitions() const {
    return set_op_partitions(ranges::views::iota(0ul, input_.opsize()) |
                             ranges::views::transform([](const std::size_t v) {
                               return std::array<std::size_t, 1>{{v}};
                             }) |
                             ranges::to_vector);
  }
  ///@}

  /// Computes and returns the result
  /// @param count_only if true, will return the total number of terms, as a
  /// Constant.
  /// @return the result of applying Wick's theorem; either a Constant, a
  /// Product, or a Sum
  /// @warning this is not reentrant, but is optionally threaded internally
  ExprPtr compute(const bool count_only = false);

  /// Collects compute statistics
  class Stats {
   public:
    Stats() : num_attempted_contractions(0), num_useful_contractions(0) {}
    Stats(const Stats &other) noexcept {
      num_attempted_contractions.store(other.num_attempted_contractions.load());
      num_useful_contractions.store(other.num_useful_contractions.load());
    }
    Stats &operator=(const Stats &other) noexcept {
      num_attempted_contractions.store(other.num_attempted_contractions.load());
      num_useful_contractions.store(other.num_useful_contractions.load());
      return *this;
    }

    void reset() {
      num_attempted_contractions = 0;
      num_useful_contractions = 0;
    }

    Stats &operator+=(const Stats &other) {
      num_attempted_contractions += other.num_attempted_contractions;
      num_useful_contractions += other.num_useful_contractions;
      return *this;
    }

    std::atomic<size_t> num_attempted_contractions;
    std::atomic<size_t> num_useful_contractions;
  };

  /// Statistics accessor
  /// @return statistics of compute calls since creation, or since the last call
  /// to reset_stats()
  const Stats &stats() const { return stats_; }

  /// Statistics accessor
  /// @return statistics of compute calls since creation, or since the last call
  /// to reset_stats()
  Stats &stats() { return stats_; }

  /// Statistics reset
  void reset_stats() { stats_.reset(); }

 private:
  static constexpr size_t max_input_size =
      32;  // max # of operators in the input sequence

  // if nonnull, apply wick to the whole expression recursively, else input_ is
  // set this is mutated by compute
  mutable ExprPtr expr_input_;

  mutable NormalOperatorSequence<S> input_;
  bool full_contractions_ = true;
  bool spinfree_ = false;
  bool use_topology_ = false;
  mutable Stats stats_;

  container::set<Index> external_indices_;
  /// for each operator specifies the reverse bitmask of target connections
  /// (0 = must connect)
  container::svector<std::bitset<max_input_size>> nop_connections_;
  std::size_t nop_nconnections_total_ =
      0;  // # of total (bidirectional) connections in nop_connections_ (i.e.
          // not double counting 1->2 and 2->1)
  container::svector<std::pair<size_t, size_t>>
      nop_connections_input_;  // only used to cache input to
                               // set_nop_connections_

  enum class TopologicalPartitionType { NormalOperator, Index };

  /// the number of partitions of the topologically equivalent NormalOperator's
  /// objects in the NormalOperatorSequence
  mutable std::size_t nop_npartitions_ = 0;
  /// for each NormalOperator specifies its topological partition (0 =
  /// topologically unique)
  mutable container::svector<size_t> nop_partition_idx_;

  /// the number of partitions of the topologically equivalent Op<S>'s in the
  /// NormalOperatorSequence
  mutable std::size_t op_npartitions_ = 0;
  /// sequence of topological partitions of Op<S> objects
  mutable container::vector<container::set<size_t>> op_partitions_;
  /// for each op specifies its topological partition (1-based, 0 =
  /// topologically unique)
  mutable container::svector<size_t> op_partition_idx_;
  /// to map op object to the topological partitions need to be able to map them
  /// to their ordinals in input_
  mutable container::map<Op<S>, std::size_t> op_to_input_ordinal_;
  friend class NontensorWickState;  // NontensorWickState needs to access
                                    // members of this

  /// upsizes `{nop,index}_topological_partition_`, filling new entries with
  /// zeroes noop if current size > new_size
  /// @param new_size the number of items in
  /// `{nop,index}_topological_partition_`
  /// @param type the type of partitions to update; if
  /// `type==TopologicalPartitionType::NormalOperator` updates
  /// `nop_partition_idx_`, else updates `op_topological_partitions_`
  /// @return the (updated) size of `{nop,index}_topological_partition_`
  size_t upsize_topological_partitions(size_t new_size,
                                       TopologicalPartitionType type) const {
    auto &topological_partitions =
        type == TopologicalPartitionType::NormalOperator ? nop_partition_idx_
                                                         : op_partition_idx_;
    using std::size;
    const auto current_size = size(topological_partitions);
    if (new_size > current_size) {
      topological_partitions.resize(new_size);
      for (size_t i = current_size; i != new_size; ++i)
        topological_partitions[i] = 0;
      return new_size;
    } else
      return current_size;
  }

  /// initializes input_
  /// @param nopseq the NormalOperatorSequence to initialize input_ with
  WickTheorem &init_input(const NormalOperatorSequence<S> &nopseq) {
    input_ = nopseq;

    // need to be able to look up ordinals of ops in the input expression to
    // make index partitions usable
    using nopseq_view_type = flattened_rangenest<NormalOperatorSequence<S>>;
    auto nopseq_view = nopseq_view_type(&input_);
    std::size_t op_ord = 0;
    op_to_input_ordinal_.reserve(input_.opsize());
    ranges::for_each(nopseq_view, [&](const auto &op) {
      op_to_input_ordinal_.emplace(op, op_ord);
      ++op_ord;
    });

    return *this;
  }

  /// Evaluates wick_ theorem for a single NormalOperatorSequence
  /// @return the result of applying Wick's theorem
  ExprPtr compute_nopseq(const bool count_only) const {
    if (spinfree_)
      throw std::logic_error(
          "WickTheorem::compute: spinfree=true not yet supported");
    // process cached nop_connections_input_, if needed
    if (!nop_connections_input_.empty())
      const_cast<WickTheorem<S> &>(*this).set_nop_connections(
          nop_connections_input_);
    // size nop_topological_partition_ to match input_, if needed
    upsize_topological_partitions(input_.size(),
                                  TopologicalPartitionType::NormalOperator);
    // initialize op partitions, if not done so
    if (input_.opsize() > 0 && op_npartitions_ == 0)
      make_default_op_partitions();

    // now compute
    auto result = compute_nontensor_wick(count_only);
    return std::move(result);
  }

  /// carries state down the stack of recursive calls
  struct NontensorWickState {
   public:
    /// @return the number of elements in a lower triangle of an `n` by `n`
    /// matrix
    template <typename T>
    static auto ntri(T n) {
      assert(n > 0);
      return n * (n - 1) / 2;
    }

    template <typename T>
    static auto lowtri_idx(T i, T j) {
      assert(i > j);
      return i * (i - 1) / 2 + j;
    }

    template <typename T, typename U>
    static auto uptri_idx(T i, T j, U n) {
      assert(i >= 0);
      assert(j >= 0);
      assert(i < n);
      assert(j < n);
      assert(i < j);
      return i * (2 * n - i - 1) / 2 + j - i - 1;
    }

    template <typename T>
    auto uptri_nop(T i, T j) const {
      return uptri_idx(i, j, this->nopseq.size());
    }

    template <typename T>
    auto uptri_op(T i, T j) const {
      return uptri_idx(i, j, this->wick.op_npartitions_);
    }

    NontensorWickState(const WickTheorem<S> &wt,
                       const NormalOperatorSequence<S> &nopseq)
        : wick(wt),
          nopseq(nopseq),
          nopseq_size(nopseq.opsize()),
          level(0),
          left_op_offset(0),
          count_only(false),
          count(0),
          nop_connections(nopseq.size()),
          nop_adjacency_matrix(ntri(nopseq.size()), 0),
          nop_nconnections(nopseq.size(), 0) {
      init_topological_partitions();
      init_input_index_columns();
    }

    NontensorWickState(const NontensorWickState &) = delete;
    NontensorWickState(NontensorWickState &&) = delete;
    NontensorWickState &operator=(const NontensorWickState &) = delete;
    NontensorWickState &operator=(NontensorWickState &&) = delete;

    const WickTheorem<S> &wick;        //!< the WickTheorem object using this
    NormalOperatorSequence<S> nopseq;  //!< current state of operator sequence
    std::size_t nopseq_size;           //!< current size of nopseq
    Product sp;                        //!< current prefactor
    int level;                         //!< level in recursive wick call stack
    size_t left_op_offset;  //!< where to start looking for contractions
                            //!< (ordinal in the current state of nopseq)
    bool count_only;  //!< if true, only track the total number of summands in
                      //!< the result (i.e. 1 (the normal product) + the number
                      //!< of contractions (if normal wick result is wanted) or
                      //!< the number of complete constractions (if want
                      //!< complete contractions only)
    std::atomic<size_t> count;  //!< if count_only is true, will count the
                                //!< total number of terms
    container::svector<std::bitset<max_input_size>>
        nop_connections;  //!< bitmask of connections for each nop (1 =
                          //!< connected)
    container::svector<size_t>
        nop_adjacency_matrix;  //!< number of connections between each nop, only
                               //!< lower triangle is kept

    /// for each NormalOperator specifies how many connections it currently has
    /// @note exists to avoid the need to traverse nop_adjacency_matrix
    container::svector<size_t> nop_nconnections;
    /// current state of nop partitions (will only match contents of
    /// nop_to_partition before any contractions have occurred)
    /// - when a normal operator is connected it's removed from the partition
    /// - when it is disconnected fully it's re-added to the partition
    container::svector<container::set<size_t>> nop_partitions;

    container::svector<size_t>
        op_partition_cdeg_matrix;  //!< contraction degree
                                   //!< (number of contractions) between
                                   //!< each topologically-equivalent group
                                   //!< of Op<S> objects, only the upper
                                   //!< triangle is kept

    /// for each Op<S> partition specifies how many contractions it currently
    /// has
    /// @note exists to avoid the need to traverse op_partition_cdeg_matrix
    container::svector<size_t> op_partition_ncontractions;

    container::svector<std::pair<Index, Index>>
        input_partner_indices;  //!< list of {cre,ann} pairs of Index objects in
                                //!< the input whose corresponding Op<S> objects
                                //!< act on the same particle

    /// "merges" partner index pair from input_index_columns with contracted
    /// Index pairs in this->sp
    auto make_target_partner_indices() const {
      // copy all pairs in the input product
      container::svector<std::pair<Index, Index>> result(input_partner_indices);
      // for every contraction so far encountered ...
      for (auto &contr : sp) {
        // N.B. sp is composed of 1-particle contractions
        assert(contr->template is<Tensor>() &&
               contr->template as<Tensor>().rank() == 1 &&
               contr->template as<Tensor>().label() ==
                   sequant::overlap_label());
        const auto &contr_t = contr->template as<Tensor>();
        const auto &bra_idx = contr_t.bra().at(0);
        const auto &ket_idx = contr_t.ket().at(0);
        // ... if both bra and ket indices were in the input list, "merge" their
        // pairs
        auto bra_it = ranges::find_if(result, [&bra_idx](const auto &p) {
          assert(p.first != bra_idx);
          return p.second == bra_idx;
        });
        if (bra_it != result.end()) {
          auto ket_it = ranges::find_if(result, [&ket_idx](const auto &p) {
            assert(p.second != ket_idx);
            return p.first == ket_idx;
          });
          if (ket_it != result.end()) {
            assert(bra_it != ket_it);
            if (ket_it > bra_it) {
              bra_it->second = std::move(ket_it->second);
              result.erase(ket_it);
            } else {
              ket_it->first = std::move(bra_it->first);
              result.erase(bra_it);
            }
          }
        }
      }
      return result;
    }

    // populates partitions using the data from nop_topological_partition
    void init_topological_partitions() {
      // nop partitions
      {
        // partition indices in wick.nop_partition_idx_ are 1-based
        const auto npartitions = wick.nop_npartitions_;
        nop_partitions.resize(npartitions);
        size_t cnt = 0;
        ranges::for_each(wick.nop_partition_idx_,
                         [this, &cnt](size_t partition_idx) {
                           if (partition_idx > 0) {  // in a partition
                             nop_partitions.at(partition_idx - 1).insert(cnt);
                           }
                           ++cnt;
                         });
        // assert that we don't have empty partitions due to broken logic
        // upstream
        assert(ranges::any_of(nop_partitions, [](auto &&partition) {
                 return partition.empty();
               }) == false);
      }

      // op partitions
      {
        // partition indices in nop_to_partition are 1-based
        // unlike nops, each op is in a partition
        const auto npartitions = wick.op_npartitions_;
        op_partition_cdeg_matrix.resize(ntri(npartitions));
        ranges::fill(op_partition_cdeg_matrix, 0);
        op_partition_ncontractions.resize(npartitions);
        ranges::fill(op_partition_ncontractions, 0);
      }
    }

    // populates target_particle_ops
    void init_input_index_columns() {
      // for each NormalOperator
      for (auto &nop : nopseq) {
        using ranges::views::reverse;
        using ranges::views::zip;

        // zip in reverse order to handle non-number-conserving ops (i.e.
        // nop.creators().size() != nop.annihilators().size()) reverse after
        // insertion to restore canonical partner index order (in the order of
        // particle indices in the normal operators) 7/18/2022 N.B. reverse(zip)
        // for some reason is broken, hence the ugliness
        std::size_t ninserted = 0;
        for (auto &&[cre, ann] :
             zip(reverse(nop.creators()), reverse(nop.annihilators()))) {
          input_partner_indices.emplace_back(cre.index(), ann.index());
          ++ninserted;
        }
        std::reverse(input_partner_indices.rbegin(),
                     input_partner_indices.rbegin() + ninserted);
      }
    }

    /// @brief Updates connectivity if contraction satisfies target connectivity

    /// If the target connectivity will be violated by this contraction, keep
    /// the state unchanged and return false
    template <typename Cursor>
    inline bool connect(const container::svector<std::bitset<max_input_size>>
                            &target_nop_connections,
                        const Cursor &op1_cursor, const Cursor &op2_cursor) {
      assert(op1_cursor.ordinal() < op2_cursor.ordinal());

      auto update_nop_metadata = [this](size_t nop_idx) {
        const auto nconnections = nop_nconnections[nop_idx];
        // if using topological partitions for normal ops, and this operator is
        // in one of them, remove it on first connection
        if (!nop_partitions.empty()) {
          auto partition_idx = wick.nop_partition_idx_[nop_idx];
          if (nconnections == 0 && partition_idx > 0) {
            --partition_idx;  // to 0-based
            assert(nop_partitions.at(partition_idx).size() > 0);
            auto removed = nop_partitions[partition_idx].erase(nop_idx);
            assert(removed);
          }
        }
        ++nop_nconnections[nop_idx];
      };

      auto update_op_metadata = [this](const Op<S> &op1, const Op<S> &op2) {
        if (!this->wick.op_partition_idx_.empty()) {
          assert(this->wick.op_to_input_ordinal_.contains(op1));
          const auto op1_ord = this->wick.op_to_input_ordinal_[op1];
          auto op1_partition_idx = wick.op_partition_idx_[op1_ord];
          assert(op1_partition_idx > 0);
          --op1_partition_idx;  // now partition index is 0-based

          assert(this->wick.op_to_input_ordinal_.contains(op2));
          const auto op2_ord = this->wick.op_to_input_ordinal_[op2];
          auto op2_partition_idx = wick.op_partition_idx_[op2_ord];
          assert(op2_partition_idx > 0);
          --op2_partition_idx;  // now partition index is 0-based

          // op ordinals and partition indices are in increasing order
          assert(op1_ord < op2_ord);
          assert(op1_partition_idx < op2_partition_idx);

          assert(op_partition_cdeg_matrix.size() >
                 uptri_op(op1_partition_idx, op2_partition_idx));
          op_partition_cdeg_matrix[uptri_op(op1_partition_idx,
                                            op2_partition_idx)] += 1;
          ++op_partition_ncontractions[op1_partition_idx];
          ++op_partition_ncontractions[op2_partition_idx];
        }
      };

      // local vars
      const auto nop1_idx = op1_cursor.range_ordinal();
      const auto nop2_idx = op2_cursor.range_ordinal();
      if (target_nop_connections
              .empty()) {  // if no constraints, all is fair game
        update_nop_metadata(nop1_idx);
        update_nop_metadata(nop2_idx);
        update_op_metadata(*op1_cursor.elem_iter(), *op2_cursor.elem_iter());
        return true;
      }
      const auto nop1_nop2_connected = nop_connections[nop1_idx].test(nop2_idx);

      // update the connectivity
      if (!nop1_nop2_connected) {
        nop_connections[nop1_idx].set(nop2_idx);
        nop_connections[nop2_idx].set(nop1_idx);
      }

      // test if nop1 has enough remaining indices to satisfy
      const auto nidx_nop1_remain =
          op1_cursor.range_iter()->size() -
          1;  // how many indices nop1 has minus this index
      const auto nidx_nop1_needs =
          (nop_connections[nop1_idx] | target_nop_connections[nop1_idx])
              .flip()
              .count();
      if (nidx_nop1_needs > nidx_nop1_remain) {
        if (!nop1_nop2_connected) {
          nop_connections[nop1_idx].reset(nop2_idx);
          nop_connections[nop2_idx].reset(nop1_idx);
        }
        return false;
      }

      // test if nop2 has enough remaining indices to satisfy
      const auto nidx_nop2_remain =
          op2_cursor.range_iter()->size() -
          1;  // how many indices nop2 has minus this index
      const auto nidx_nop2_needs =
          (nop_connections[nop2_idx] | target_nop_connections[nop2_idx])
              .flip()
              .count();
      if (nidx_nop2_needs > nidx_nop2_remain) {
        if (!nop1_nop2_connected) {
          nop_connections[nop1_idx].reset(nop2_idx);
          nop_connections[nop2_idx].reset(nop1_idx);
        }
        return false;
      }

      nop_adjacency_matrix[uptri_nop(nop1_idx, nop2_idx)] += 1;
      update_nop_metadata(nop1_idx);
      update_nop_metadata(nop2_idx);
      update_op_metadata(*op1_cursor.elem_iter(), *op2_cursor.elem_iter());

      return true;
    }

    /// @brief Updates connectivity when contraction is reversed
    template <typename Cursor>
    inline void disconnect(const container::svector<std::bitset<max_input_size>>
                               &target_nop_connections,
                           const Cursor &op1_cursor, const Cursor &op2_cursor) {
      assert(op1_cursor.ordinal() < op2_cursor.ordinal());

      auto update_nop_metadata = [this](size_t nop_idx) {
        assert(nop_nconnections.at(nop_idx) > 0);
        const auto nconnections = --nop_nconnections[nop_idx];
        if (!nop_partitions.empty()) {
          auto partition_idx = wick.nop_partition_idx_[nop_idx];
          if (nconnections == 0 && partition_idx > 0) {
            --partition_idx;  // to 0-based
            auto inserted = nop_partitions.at(partition_idx).insert(nop_idx);
            assert(inserted.second);
          }
        }
      };

      auto update_op_metadata = [this](const Op<S> &op1, const Op<S> &op2) {
        if (!this->wick.op_partition_idx_.empty()) {
          assert(this->wick.op_to_input_ordinal_.contains(op1));
          const auto op1_ord = this->wick.op_to_input_ordinal_[op1];
          auto op1_partition_idx = wick.op_partition_idx_[op1_ord];
          assert(op1_partition_idx > 0);
          --op1_partition_idx;  // now partition index is 0-based

          assert(this->wick.op_to_input_ordinal_.contains(op2));
          const auto op2_ord = this->wick.op_to_input_ordinal_[op2];
          auto op2_partition_idx = wick.op_partition_idx_[op2_ord];
          assert(op2_partition_idx > 0);
          --op2_partition_idx;  // now partition index is 0-based

          // op ordinals and partition indices are in increasing order
          assert(op1_ord < op2_ord);
          assert(op1_partition_idx < op2_partition_idx);

          assert(op_partition_cdeg_matrix.size() >
                 uptri_op(op1_partition_idx, op2_partition_idx));
          assert(op_partition_cdeg_matrix[uptri_op(op1_partition_idx,
                                                   op2_partition_idx)] > 0);
          op_partition_cdeg_matrix[uptri_op(op1_partition_idx,
                                            op2_partition_idx)] -= 1;
          assert(op_partition_ncontractions[op1_partition_idx] > 0);
          assert(op_partition_ncontractions[op2_partition_idx] > 0);
          --op_partition_ncontractions[op1_partition_idx];
          --op_partition_ncontractions[op2_partition_idx];
        }
      };

      // local vars
      const auto nop1_idx = op1_cursor.range_ordinal();
      const auto nop2_idx = op2_cursor.range_ordinal();
      update_nop_metadata(nop1_idx);
      update_nop_metadata(nop2_idx);
      update_op_metadata(*op1_cursor.elem_iter(), *op2_cursor.elem_iter());
      if (target_nop_connections.empty())  // if no constraints, we don't keep
                                           // track of individual connections
        return;
      assert(nop_connections[nop1_idx].test(nop2_idx));

      auto &adjval = nop_adjacency_matrix[uptri_nop(nop1_idx, nop2_idx)];
      assert(adjval > 0);
      adjval -= 1;
      if (adjval == 0) {
        nop_connections[nop1_idx].reset(nop2_idx);
        nop_connections[nop2_idx].reset(nop1_idx);
      }
    }
  };  // NontensorWickState

  /// Applies most naive version of Wick's theorem, where the sign rule involves
  /// counting Ops
  /// @return the result
  ExprPtr compute_nontensor_wick(const bool count_only) const {
    std::vector<std::pair<Product, std::shared_ptr<NormalOperator<S>>>>
        result;      //!< current value of the result
    std::mutex mtx;  // used in critical sections updating the result
    auto result_plus_mutex = std::make_pair(&result, &mtx);
    NontensorWickState state(*this, input_);
    state.count_only = count_only;
    // TODO extract index->particle maps

    if (Logger::get_instance().wick_contract) {
      std::wcout << "nop topological partitions: {\n";
      for (auto &&toppart : state.nop_partitions) {
        std::wcout << "{" << std::endl;
        for (auto &&op_idx : toppart) {
          std::wcout << op_idx << std::endl;
        }
        std::wcout << "}" << std::endl;
      }
      std::wcout << "}" << std::endl;
    }

    recursive_nontensor_wick(result_plus_mutex, state);

    // if computing everything, and the user does not insist on some
    // target contractions, include the contraction-free term
    if (!full_contractions_ && nop_nconnections_total_ == 0) {
      if (count_only) {
        ++state.count;
      } else {
        auto [phase, normop] = normalize(input_, state.input_partner_indices);
        result_plus_mutex.first->push_back(
            std::make_pair(Product(phase, {}), std::move(normop)));
      }
    }

    // convert result to an Expr
    // if result.size() == 0, return null ptr
    ExprPtr result_expr;
    if (count_only) {  // count only? return the total number as a Constant
      assert(result.empty());
      result_expr = ex<Constant>(state.count.load());
    } else if (result.size() == 1) {  // if result.size() == 1, return Product
      auto product = std::make_shared<Product>(std::move(result.at(0).first));
      if (full_contractions_)
        assert(result.at(0).second == nullptr);
      else {
        if (result.at(0).second)
          product->append(1, std::move(result.at(0).second));
      }
      result_expr = product;
    } else if (result.size() > 1) {
      auto sum = std::make_shared<Sum>();
      for (auto &&term : result) {
        if (full_contractions_) {
          assert(term.second == nullptr);
          sum->append(ex<Product>(std::move(term.first)));
        } else {
          auto term_product = std::make_shared<Product>(std::move(term.first));
          if (term.second) {
            term_product->append(1, term.second);
          }
          sum->append(term_product);
        }
      }
      result_expr = sum;
    } else if (result_expr == nullptr)
      result_expr = ex<Constant>(0);
    return result_expr;
  }

 public:
  virtual ~WickTheorem();

 private:
  void recursive_nontensor_wick(
      std::pair<
          std::vector<std::pair<Product, std::shared_ptr<NormalOperator<S>>>> *,
          std::mutex *> &result,
      NontensorWickState &state) const {
    using nopseq_view_type = flattened_rangenest<NormalOperatorSequence<S>>;
    auto nopseq_view = nopseq_view_type(&state.nopseq);
    using std::begin;
    using std::end;

    // if full contractions needed, make contractions involving first index with
    // another index, else contract any index i with index j (i<j)
    auto left_op_offset = state.left_op_offset;
    auto op_left_iter = ranges::next(begin(nopseq_view),
                                     left_op_offset);  // left op to contract

    // optimization: can't contract fully if first op is not a qp annihilator
    if (full_contractions_ && !is_qpannihilator(*op_left_iter, input_.vacuum()))
      return;

    const auto op_left_iter_fence =
        full_contractions_ ? ranges::next(op_left_iter) : end(nopseq_view);
    for (; op_left_iter != op_left_iter_fence;
         ++op_left_iter, ++left_op_offset) {
      auto &op_left_cursor = ranges::get_cursor(op_left_iter);
      const auto op_left_input_ordinal =
          op_to_input_ordinal_.find(*op_left_iter)
              ->second;  // ordinal of op_left in input_

      auto op_right_iter = ranges::next(op_left_iter);
      for (; op_right_iter != end(nopseq_view);) {
        auto &op_right_cursor = ranges::get_cursor(op_right_iter);
        const auto op_right_input_ordinal =
            op_to_input_ordinal_.find(*op_right_iter)
                ->second;  // ordinal of op_left in input_

        if (op_right_iter != op_left_iter &&
            op_left_cursor.range_iter() !=
                op_right_cursor
                    .range_iter()  // can't contract within same normop
        ) {
          // clang-format off
          /// reports whether the contraction specified by `op_{left,right}_cursors` is unique;
          /// @return `{is_unique, nop_topological_weight}` where `is_unique` is true if the contraction is unique;
          ///         for unique_contractions `nop_topological_weight` specifies the topological weight for the contraction due to equivalences of NormalOperator's
          // clang-format on
          auto is_topologically_unique = [&]() {
            // with current way I exploit op symmetry, equivalence of nops is
            // not exploitable for pruning search tree early ... this is due to
            // the fact that nonuniqueness of contractions due to nop symmetry
            // can only be decided when final contraction matrix order is
            // available
            constexpr bool use_nop_symmetry = false;
            std::size_t nop_topological_weight = 1;

            bool is_unique = true;
            if (use_topology_) {
              auto &nop_right = *(op_right_cursor.range_iter());
              auto &op_right_it = op_right_cursor.elem_iter();

              // check against the degeneracy deduced by index partition
              constexpr bool use_op_partition_groups = true;
              constexpr bool test_vs_old_code = false;

              size_t ref_op_psymm_weight = 1;
              if constexpr (test_vs_old_code) {
                auto op_right_idx_in_opseq =
                    op_right_it - ranges::begin(nop_right);
                auto &hug_right = nop_right.hug();
                auto &group_right = hug_right->group(op_right_idx_in_opseq);
                if (group_right.second.find(op_right_idx_in_opseq) ==
                    group_right.second.begin())
                  ref_op_psymm_weight =
                      hug_right->group_size(op_right_idx_in_opseq);
                else
                  ref_op_psymm_weight = 0;
              }

              {
                const auto op_left_partition_idx =
                    state.wick.op_partition_idx_[op_left_input_ordinal] - 1;
                const auto op_right_partition_idx =
                    state.wick.op_partition_idx_[op_right_input_ordinal] - 1;
                const auto past_op_right_partition_idx =
                    op_right_partition_idx + 1;

                // contract only the first free op in left partition
                // this is ensured automatically if full_contractions_==true
                if (!full_contractions_) {
                  const auto left_partition_ncontr_total =
                      state.op_partition_ncontractions[op_left_partition_idx];
                  const auto &op_left_partition =
                      state.wick.op_partitions_[op_left_partition_idx];
                  const auto op_left_ord_in_partition =
                      op_left_partition.find(op_left_input_ordinal) -
                      op_left_partition.begin();
                  is_unique =
                      left_partition_ncontr_total == op_left_ord_in_partition;
                }

                // also skip contractions that would connect op1_partition with
                // op2_partition (op1_partition<op2_partition) if there is
                // another partition p>op2_partition that op1_partition
                // is connected to
                if (use_op_partition_groups && is_unique &&
                    past_op_right_partition_idx < this->op_npartitions_) {
                  const auto left_partition_ncontr_past_right_partition =
                      ranges::span<size_t>(
                          state.op_partition_cdeg_matrix.data() +
                              state.uptri_op(op_left_partition_idx,
                                             past_op_right_partition_idx),
                          op_npartitions_ - past_op_right_partition_idx);

                  if (ranges::any_of(
                          left_partition_ncontr_past_right_partition,
                          [](auto x) {
                            return x > 0;
                          })) {  // only contract with a partition if had not
                                 // already contracted to partitions past it
                    is_unique = false;
                  }
                }

                // contract to the first free op in the right partition
                if (is_unique) {
                  const auto right_partition_ncontr_total =
                      state.op_partition_ncontractions[op_right_partition_idx];
                  const auto &op_right_partition =
                      state.wick.op_partitions_[op_right_partition_idx];
                  const auto op_right_ord_in_partition =
                      op_right_partition.find(op_right_input_ordinal) -
                      op_right_partition.begin();
                  is_unique =
                      right_partition_ncontr_total == op_right_ord_in_partition;

                  if (is_unique && test_vs_old_code) {
                    // old code assumes bra/ket of each NormalOperator forms
                    // a single partition
                    const auto nop_right_input =
                        input_.at(op_right_cursor.range_ordinal());
                    const auto op_right_ord_in_nop_input =
                        ranges::find(nop_right_input, *op_right_it) -
                        ranges::begin(nop_right_input);
                    if (op_right_partition.size() ==
                        nop_right_input.hug()->group_size(
                            op_right_ord_in_nop_input)) {
                      assert(ref_op_psymm_weight ==
                             op_right_partition.size() -
                                 op_right_ord_in_partition);
                    }
                  }
                }
              }

              // account for topologically-equivalent normal operators
              if (use_nop_symmetry && is_unique &&
                  !state.nop_partitions.empty()) {
                auto nop_right_idx =
                    ranges::get_cursor(op_right_iter)
                        .range_ordinal();  // the index of normal operator
                auto nop_right_partition_idx = state.wick.nop_partition_idx_.at(
                    nop_right_idx);  // the partition to which this normal
                                     // operator belongs to (0 = none)
                if (nop_right_partition_idx >
                    0) {                      // if part of a partition ...
                  --nop_right_partition_idx;  // to 0-based
                  const auto &nop_right_partition =
                      state.nop_partitions.at(nop_right_partition_idx);
                  if (!nop_right_partition.empty()) {  // ... and the partition
                                                       // is not empty ...
                    const auto it = nop_right_partition.find(nop_right_idx);
                    // .. and not missing from the partition (because then it's
                    // topologically unique) ...
                    if (it != nop_right_partition.end()) {
                      // ... and first in the partition
                      if (it == nop_right_partition.begin()) {
                        // account for the entire partition by scaling the
                        // contribution from the first contraction from this
                        // normal operator
                        nop_topological_weight = nop_right_partition.size();
                      } else
                        is_unique = false;
                    }
                  }
                }
              }  // use_nop_symmetry

            }  // use_topology
            return std::make_tuple(is_unique, nop_topological_weight);
          };

          // clang-format off
          /// @return the permutational symmetry factor of a contraction order matrix
          // clang-format on
          auto op_permutational_degeneracy = [&]() {
            rational result = 1;
            const auto &contraction_order_matrix_uptri =
                state.op_partition_cdeg_matrix;
            for (auto i :
                 ranges::views::iota(std::size_t{0}, op_npartitions_)) {
              const auto partition_i_size = op_partitions_[i].size();
              if (partition_i_size > 1) {
                result *= factorial(partition_i_size);
                for (auto j : ranges::views::iota(i + 1, op_npartitions_)) {
                  const auto ncontr_ij =
                      contraction_order_matrix_uptri[state.uptri_op(i, j)];
                  if (ncontr_ij > 1) result /= factorial(ncontr_ij);
                }
                // if partially contracted account for non-contracted ops in the
                // partition # of currently contracted
                const auto partition_i_ncontr =
                    state.op_partition_ncontractions[i];
                // # of currently non-contracted ops
                const auto partition_i_noncontr =
                    partition_i_size - partition_i_ncontr;
                // sanity check: only full contractions encountered
                // if full_contractions_==true
                assert(!full_contractions_ || partition_i_noncontr == 0);
                if (partition_i_noncontr > 1)
                  result /= factorial(partition_i_noncontr);
              }
            }
            return result;
          };

          // check if can contract these indices and
          // check connectivity constraints (if needed)
          if (can_contract(*op_left_iter, *op_right_iter, input_.vacuum())) {
            auto &&[is_unique, nop_top_degen] = is_topologically_unique();
            if (is_unique) {
              if (state.connect(nop_connections_,
                                ranges::get_cursor(op_left_iter),
                                ranges::get_cursor(op_right_iter))) {
                if (Logger::get_instance().wick_contract) {
                  std::wcout << "level " << state.level << ":contracting "
                             << to_latex(*op_left_iter) << " with "
                             << to_latex(*op_right_iter)
                             << " (nop_top_degen=" << nop_top_degen << ")"
                             << std::endl;
                  std::wcout << " current nopseq = " << to_latex(state.nopseq)
                             << std::endl;
                }

                // update the phase, if needed
                int64_t phase = 1;
                if (statistics == Statistics::FermiDirac) {
                  const auto distance =
                      ranges::get_cursor(op_right_iter).ordinal() -
                      ranges::get_cursor(op_left_iter).ordinal() - 1;
                  if (distance % 2) {
                    phase *= -1;
                  }
                }

                // update the prefactor and nopseq
                Product sp_copy = state.sp;
                state.sp.append(
                    static_cast<int64_t>(nop_top_degen) * phase,
                    contract(*op_left_iter, *op_right_iter, input_.vacuum()));

                // update the stats
                ++stats_.num_attempted_contractions;

                // remove from back to front
                Op<S> right = *op_right_iter;
                ranges::get_cursor(op_right_iter).erase();
                --state.nopseq_size;
                Op<S> left = *op_left_iter;
                ranges::get_cursor(op_left_iter).erase();
                --state.nopseq_size;

                //            std::wcout << "  nopseq after contraction = " <<
                //            to_latex(state.opseq) << std::endl;

                // if have a nonzero result ...
                if (!state.sp.empty()) {
                  // update the result if nothing left to contract and
                  if (!full_contractions_ ||
                      (full_contractions_ && state.nopseq_size == 0)) {
                    if (!state.count_only) {
                      if (full_contractions_) {
                        result.second->lock();
                        //              std::wcout << "got " <<
                        //              to_latex(state.sp)
                        //              << std::endl;
                        result.first->push_back(std::make_pair(
                            std::move(state.sp.deep_copy().scale(
                                op_permutational_degeneracy())),
                            std::shared_ptr<NormalOperator<S>>{}));
                        //              std::wcout << "now up to " <<
                        //              result.first->size()
                        //              << " terms" << std::endl;
                        result.second->unlock();
                      } else {
                        auto [phase, op] = normalize(
                            state.nopseq, state.make_target_partner_indices());
                        result.second->lock();
                        result.first->push_back(std::make_pair(
                            std::move(state.sp.deep_copy().scale(
                                phase * op_permutational_degeneracy())),
                            op->empty() ? nullptr : std::move(op)));
                        result.second->unlock();
                      }
                    } else
                      ++state.count;

                    // update the stats: count this contraction as useful
                    ++stats_.num_useful_contractions;
                  }
                }

                if (state.nopseq_size != 0) {
                  const auto current_num_useful_contractions =
                      stats_.num_useful_contractions.load();
                  ++state.level;
                  state.left_op_offset = left_op_offset;
                  recursive_nontensor_wick(result, state);
                  --state.level;
                  // this contraction is useful if it leads to useful
                  // contractions as a result
                  if (current_num_useful_contractions !=
                      stats_.num_useful_contractions.load())
                    ++stats_.num_useful_contractions;
                }

                // restore the prefactor and nopseq
                state.sp = std::move(sp_copy);
                // restore from front to back
                ranges::get_cursor(op_left_iter).insert(std::move(left));
                ++state.nopseq_size;
                ranges::get_cursor(op_right_iter).insert(std::move(right));
                ++state.nopseq_size;
                state.disconnect(nop_connections_,
                                 ranges::get_cursor(op_left_iter),
                                 ranges::get_cursor(op_right_iter));
                //            std::wcout << "  restored nopseq = " <<
                //            to_latex(state.opseq) << std::endl;
              }  // connect succeeded
            }    // topologically-unique contraction
          }      // can_contract
          ++op_right_iter;
        } else {
          ++op_right_iter;
        }
      }  // right op iter
    }    // left op iter
  }

 public:
  static bool can_contract(const Op<S> &left, const Op<S> &right,
                           Vacuum vacuum = get_default_context().vacuum()) {
    // for bosons can only do Wick's theorem for physical vacuum (or similar)
    if constexpr (statistics == Statistics::BoseEinstein)
      assert(vacuum == Vacuum::Physical);

    if (is_qpannihilator<S>(left, vacuum) && is_qpcreator<S>(right, vacuum)) {
      const auto qpspace_left = qpannihilator_space<S>(left, vacuum);
      const auto qpspace_right = qpcreator_space<S>(right, vacuum);
      const auto qpspace_common = intersection(qpspace_left, qpspace_right);
      if (qpspace_common != IndexSpace::null_instance()) return true;
    }
    return false;
  }

  static ExprPtr contract(const Op<S> &left, const Op<S> &right,
                          Vacuum vacuum = get_default_context().vacuum()) {
    assert(can_contract(left, right, vacuum));
    //    assert(
    //        !left.index().has_proto_indices() &&
    //            !right.index().has_proto_indices());  // I don't think the
    //            logic is
    // correct for dependent indices
    if (is_pure_qpannihilator<S>(left, vacuum) &&
        is_pure_qpcreator<S>(right, vacuum))
      return make_overlap(left.index(), right.index());
    else {
      const auto qpspace_left = qpannihilator_space<S>(left, vacuum);
      const auto qpspace_right = qpcreator_space<S>(right, vacuum);
      const auto qpspace_common = intersection(qpspace_left, qpspace_right);
      const auto index_common = Index::make_tmp_index(qpspace_common);

      // preserve bra/ket positions of left & right
      const auto left_is_ann = left.action() == Action::annihilate;
      assert(left_is_ann || right.action() == Action::annihilate);

      if (qpspace_common != left.index().space() &&
          qpspace_common !=
              right.index().space()) {  // may need 2 overlaps if neither space
        // is pure qp creator/annihilator
        auto result = std::make_shared<Product>();
        result->append(1, left_is_ann
                              ? make_overlap(left.index(), index_common)
                              : make_overlap(index_common, left.index()));
        result->append(1, left_is_ann
                              ? make_overlap(index_common, right.index())
                              : make_overlap(right.index(), index_common));
        return result;
      } else {
        return left_is_ann ? make_overlap(left.index(), right.index())
                           : make_overlap(right.index(), left.index());
      }
    }
  }

  /// @param[in,out] on input, Wick's theorem result, on output the result of
  /// reducing the overlaps
  void reduce(ExprPtr &expr) const;
};

using BWickTheorem = WickTheorem<Statistics::BoseEinstein>;
using FWickTheorem = WickTheorem<Statistics::FermiDirac>;

}  // namespace sequant

#include "wick.impl.hpp"

#endif  // SEQUANT_WICK_HPP
