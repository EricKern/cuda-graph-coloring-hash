#pragma once
#include <type_traits>


namespace apa22_coloring {

// reverses bit order
template <typename T>
struct brev_classic {
	static_assert(std::is_unsigned_v<T> && sizeof(T) == 4);
	__forceinline__ __host__ T operator()(T a) const noexcept {
		a = (a >> 16) | (a << 16); // swap halfwords
		T m{0x00ff00ff};
		a = ((a >> 8) & m) | ((a << 8) & ~m); // swap bytes
		m = m ^ (m << 4);
		a = ((a >> 4) & m) | ((a << 4) & ~m); // swap nibbles
		m = m ^ (m << 2);
		a = ((a >> 2) & m) | ((a << 2) & ~m);
		m = m ^ (m << 1);
		a = ((a >> 1) & m) | ((a << 1) & ~m);
		return a;
	}
};

/** Functor brev_classic_comp
 * This functor will compare two values while reversing their least significant bit to their most significant bit
 */
template <typename T>
struct brev_classic_cmp {
	static_assert(std::is_unsigned_v<T>);
	__forceinline__ __host__ bool operator()(T a, T b) const noexcept {
		brev_classic<T> brev{};
		return brev(a) < brev(b);
	}
};

} // namespace apa22_coloring