#ifndef INCLUDED_UTIL_HPP
#define INCLUDED_UTIL_HPP

#include <iostream>
#include <vector>

template<class T>
void BOOST_REQUIRE_EQUAL_VECTORS(const std::vector<T>& lhs, const std::vector<T>& rhs)
{
    BOOST_REQUIRE_EQUAL_COLLECTIONS(begin(lhs), end(lhs), begin(rhs), end(rhs));
}

namespace std
{
    /**
     * For reasons I don't fully understand, we need to define this overloaded
     * operator<< inside the std namespace. I tried defining the operator in
     * the global namespace with the argument as `const std::vector<T>&`, but
     * compilation failed with boost template instantiation error.
     * 
     * Hint: https://stackoverflow.com/a/33884671
     */
    template <typename T>
    ostream& operator<<(ostream& out, const vector<T>& vec)
    {
        out << "{ ";
        for (const T& item : vec) {
            out << item << ' ';
        }
        out << '}';
        return out;
    }
}

#endif
