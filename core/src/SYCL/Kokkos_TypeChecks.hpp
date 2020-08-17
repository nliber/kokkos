#ifndef KOKKOS_TYPECHECKS_HPP
#define KOKKOS_TYPECHECKS_HPP

#include <type_traits>
#define isTriviallyCopyable(...)\
  static_assert(std::is_trivially_copyable<__VA_ARGS__>::value);\
  static_assert(std::is_copy_constructible<__VA_ARGS__>::value ||\
                    std::is_move_constructible<__VA_ARGS__>::value ||\
                    std::is_copy_assignable<__VA_ARGS__>::value ||\
                    std::is_move_assignable<__VA_ARGS__>::value,\
                #__VA_ARGS__ "copy/move constructors/assignments deleted");\
  static_assert(std::is_trivially_copy_constructible<__VA_ARGS__>::value ||\
                    !std::is_copy_constructible<__VA_ARGS__>::value,\
                #__VA_ARGS__ "not trivially copy constructible");\
  static_assert(std::is_trivially_move_constructible<__VA_ARGS__>::value ||\
                    !std::is_move_constructible<__VA_ARGS__>::value,\
                #__VA_ARGS__ "not trivially move constructible");\
  static_assert(std::is_trivially_copy_assignable<__VA_ARGS__>::value ||\
                    !std::is_copy_assignable<__VA_ARGS__>::value,\
                #__VA_ARGS__ "not trivially copy assignable");\
  static_assert(std::is_trivially_move_assignable<__VA_ARGS__>::value ||\
                    !std::is_move_assignable<__VA_ARGS__>::value,\
                #__VA_ARGS__ "not trivially move assignable");\
  static_assert(std::is_trivially_destructible<__VA_ARGS__>::value,\
                #__VA_ARGS__ "not trivially destructible");

#endif // KOKKOS_TYPECHECKS_HPP
