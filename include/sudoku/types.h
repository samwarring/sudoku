#ifndef INCLUDED_SUDOKU_TYPES_H
#define INCLUDED_SUDOKU_TYPES_H

#include <limits>
#include <stdexcept>

/**
 * \file types.h
 * 
 * \brief Centralizes typedefs for trivial types like CellCount, CellValue, etc.
 */

namespace sudoku
{
    using CellCount = size_t;
    using CellValue = unsigned char;
    using CellBlockCount = int;
    using ValueBlockCount = size_t;
    using GroupCount = size_t;

    class CastException : public std::runtime_error { using std::runtime_error::runtime_error; };

    /**
     * Cast a value to another type after checking the value 'fits' in the target type.
     */
    template <typename TargetType, typename SrcType>
    TargetType cast(SrcType uncheckedValue)
    {
        size_t checkedValue = uncheckedValue;
        size_t maxValue = std::numeric_limits<TargetType>::max();
        if (checkedValue > maxValue) {
            throw CastException("Value too large for type");
        }
        return static_cast<CellValue>(checkedValue);
    }

    /**
     * Cast to a CellValue
     */
    template <typename T>
    CellValue castCellValue(T uncheckedValue) { return cast<CellValue, T>(uncheckedValue); }

}

#endif
