#ifndef SEQUANT_EXTERNAL_INTERFACE_UTILS_HPP
#define SEQUANT_EXTERNAL_INTERFACE_UTILS_HPP

#include <SeQuant/core/container.hpp>
#include <SeQuant/core/expr_fwd.hpp>
#include <SeQuant/core/index.hpp>
#include <SeQuant/core/space.hpp>
#include <SeQuant/core/utility/indices.hpp>

#include <map>
#include <optional>
#include <string>

class IndexSpaceMeta {
public:
	struct Entry {
		std::wstring tag;
		std::wstring name;
	};

	IndexSpaceMeta() = default;

	std::size_t getSize(const sequant::IndexSpace &space) const;

	std::size_t getSize(const sequant::Index &index) const;

	std::wstring getLabel(const sequant::IndexSpace &space) const;

	std::wstring getName(const sequant::IndexSpace &space) const;

	std::wstring getTag(const sequant::IndexSpace &space) const;

	void registerSpace(sequant::IndexSpace space, Entry entry);

	std::map< sequant::IndexSpace, Entry > get_data() const;	
private:
	std::map< sequant::IndexSpace, Entry > m_entries;
};

sequant::container::svector< sequant::container::svector< sequant::Index > >
	getExternalIndexPairs(const sequant::ExprPtr &expression);

std::optional< sequant::ExprPtr > popTensor(sequant::ExprPtr &expression, std::wstring_view label);

bool needsSymmetrization(const sequant::ExprPtr &expression);

sequant::ExprPtr generateResultSymmetrization(std::wstring_view precursorName,
											  const sequant::IndexGroups< std::vector< sequant::Index > > &externals);

#endif
