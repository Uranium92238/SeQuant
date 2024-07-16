#ifndef SEQUANT_CORE_EXPORT_GENERATOR_HPP
#define SEQUANT_CORE_EXPORT_GENERATOR_HPP

#include <SeQuant/core/export/context.hpp>
#include <SeQuant/core/expr_fwd.hpp>
#include <SeQuant/core/index.hpp>

#include <string>
#include <type_traits>

namespace sequant {

/// Abstract base class for all (code) generators
template <typename C>
class Generator {
 public:
  using Context = C;
  static_assert(std::is_default_constructible_v<Context>,
                "Generator context objects must be default-constructible");
  static_assert(
      std::is_base_of_v<ExportContext, Context>,
      "Generator context class must inherit from sequant::ExportContext");

  virtual ~Generator() = default;

  virtual std::string get_format_name() const = 0;

  virtual std::string represent(const Tensor &tensor,
                                const Context &ctx = {}) const = 0;
  virtual std::string represent(const Variable &variable,
                                const Context &ctx = {}) const = 0;
  virtual std::string represent(const Constant &constant,
                                const Context &ctx = {}) const = 0;

  virtual void create(const Tensor &tensor, bool zero_init,
                      const Context &ctx = {}) = 0;
  virtual void load(const Tensor &tensor, bool set_to_zero,
                    const Context &ctx = {}) = 0;
  virtual void set_to_zero(const Tensor &tensor, const Context &ctx = {}) = 0;
  virtual void unload(const Tensor &tensor, const Context &ctx = {}) = 0;
  virtual void destroy(const Tensor &tensor, const Context &ctx = {}) = 0;
  virtual void persist(const Tensor &tensor, const Context &ctx = {}) = 0;

  virtual void create(const Variable &variable, bool zero_init,
                      const Context &ctx = {}) = 0;
  virtual void load(const Variable &variable, const Context &ctx = {}) = 0;
  virtual void set_to_zero(const Variable &variable,
                           const Context &ctx = {}) = 0;
  virtual void unload(const Variable &variable, const Context &ctx = {}) = 0;
  virtual void destroy(const Variable &variable, const Context &ctx = {}) = 0;
  virtual void persist(const Variable &tensor, const Context &ctx = {}) = 0;

  virtual void compute(const Expr &expression, const Tensor &result,
                       const Context &ctx = {}) = 0;
  virtual void compute(const Expr &expression, const Variable &result,
                       const Context &ctx = {}) = 0;

  virtual void declare(const Index &idx, const Context &ctx = {}) = 0;
  virtual void declare(const Variable &variable, const Context &ctx = {}) = 0;
  virtual void declare(const Tensor &tensor, const Context &ctx = {}) = 0;

  virtual void insert_comment(const std::string &comment,
                              const Context &ctx = {}) = 0;
  virtual void insert_blank_lines(std::size_t count,
                                  const Context &ctx = {}) = 0;

  virtual std::string get_generated_code() const = 0;

 protected:
  Generator() = default;
};

}  // namespace sequant

#endif  // SEQUANT_CORE_EXPORT_GENERATOR_HPP
