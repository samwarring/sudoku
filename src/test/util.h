#ifndef INCLUDED_UTIL_HPP
#define INCLUDED_UTIL_HPP

#include <iostream>

template<class T>
void BOOST_REQUIRE_EQUAL_VECTORS(const std::vector<T>& lhs, const std::vector<T>& rhs)
{
    BOOST_REQUIRE_EQUAL_COLLECTIONS(begin(lhs), end(lhs), begin(rhs), end(rhs));
}

#endif
