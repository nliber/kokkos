/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <cstdio>
#include <stdexcept>
#include <sstream>
#include <iostream>

#include <Kokkos_Core.hpp>

namespace Test {

namespace MDParallelism {

// add_pointers adds P pointers to type T

template <typename T, size_t P = 1>
struct add_pointers;

template <typename T, size_t P = 1>
using add_pointers_t = typename add_pointers<T, P>::type;

template <typename T, size_t P>
struct add_pointers {
  using type = add_pointers_t<std::add_pointer_t<T>, P - 1>;
};

template <typename T>
struct add_pointers<T, 1> {
  using type = std::add_pointer_t<T>;
};

template <typename T>
struct add_pointers<T, 0> {
  using type = std::remove_reference_t<T>;
};

template <size_t RemainingRank>
struct CheckResult {
  static constexpr size_t remaining_rank = RemainingRank;

  template <typename HostView, typename Check>
  static void check_result(int const* startIdx, HostView const& view,
                           Check const& check) {
    int dim = HostView::rank - RemainingRank;
    for (int i = 0; i < view.extent(dim); ++i) {
      CheckResult<RemainingRank - 1>::check_result(
          startIdx, view,
          [j = i + startIdx[dim], &check](auto... is) { check(j, is...); });
    }
  }
};
template <>
struct CheckResult<0> {
  static constexpr size_t remain_rank = 0;

  template <typename HostView, typename Check>
  static void check_result(int const*, HostView const&, Check const& check) {
    check();
  }
};

template <typename ExecSpace>
struct TestMDParallelFor {
  // The difference between test_for2 and test_for2_with_direction is
  // the call to MDThreadVectorRange.  test_for2 deduces all the
  // parameters for the return type, whily test_for2_with_direction
  // specifies the outer and inner directions (different code path).

// #ifdef FOUND_CUDA_WORKAROUND
  // This is used to deduce the initial RemainingRank for CheckResult
  template <typename HostView, typename Check>
  static void check_result(int const* startIdx, HostView const& view,
                           Check const& check) {
    CheckResult<HostView::Rank>::check_result(startIdx, view, check);
  }

  // template <typename HostViewType>
  // static void check_result_3D(std::string testName, HostViewType h_view,
  //                             const int startIdx[3], int expectedValue) {
  //   check_result(
  //       startIdx, h_view,
  //       [expectedValue, &h_view](const int i, const int j, const int k) {
  //         ASSERT_EQ(expectedValue, h_view(i, j, k));
  //       });
  // }

  template <typename DeferredMDTeamThreadRange, typename... Ns>
  static void test_for_MDTeamThreadRange(
      DeferredMDTeamThreadRange const& md_team_thread_range, const int n0,
      Ns... ns) {
    using DataType = int;
    using ViewType =
        typename Kokkos::View<add_pointers_t<DataType, 1 + sizeof...(ns)>,
                              ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;
    using TeamType     = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

    const int startIdx[1 + sizeof...(ns)] = {0};
    const int initValue                   = 3;

    ViewType v("v", n0, ns...);

    const int nsArray[sizeof...(ns)] = {ns...};

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(n0, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          // The static_assert is here because we needed the calling
          // signature of md_team_thread_range when we have "team"
          static_assert(Kokkos::Impl::IsMDTeamThreadRangeBoundariesStruct<
                            decltype(md_team_thread_range(team, nsArray))>::value,
                        "md_team_thread_range(team, nsArray) must return an "
                        "MDTeamThreadRangeBoundariesStruct");

          Kokkos::parallel_for(
              // md_team_thread_range(team, ns...),
              md_team_thread_range(team, nsArray),
              [=](auto... is) { v(leagueRank, is...) = initValue; });
        });

    HostViewType h_view = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(h_view, v);

    check_result(startIdx, h_view, [&h_view, initValue](auto... is) {
      ASSERT_EQ(initValue, h_view(is...));
    });
  }

  // static void test_for3_MDTeamThreadRange(const int N0, const int N1,
  //                                         const int N2) {
  //   test_for_MDTeamThreadRange(
  //       [](auto const& team, auto... ns) {
  //         return Kokkos::MDTeamThreadRange(team, ns...);
  //       },
  //       N0, N1, N2);
  // }

  // template <Kokkos::Iterate Direction>
  // static void test_for3_MDTeamThreadRange_with_direction(const int N0,
  //                                                        const int N1,
  //                                                        const int N2) {
  //   test_for_MDTeamThreadRange(
  //       [](auto const& team, auto... ns) {
  //         return Kokkos::MDTeamThreadRange<Direction>(team, ns...);
  //       },
  //       N0, N1, N2);
  // }
// #endif //FOUND_CUDA_WORKAROUND

  template <typename HostViewType>
  static void check_result_3D(std::string testName, HostViewType h_view,
                              const int startIdx[3], int expectedValue) {
    int counter = 0;
    int dim     = Kokkos::rank(h_view);

    for (int i = startIdx[0]; i < h_view.extent(0); ++i) {
      for (int j = startIdx[1]; j < h_view.extent(1); ++j) {
        for (int k = startIdx[2]; k < h_view.extent(2); ++k) {
          if (h_view(i, j, k) == expectedValue) {
            ++counter;
          }
        }
      }
    }

    int expectedCounter = 1;
    for (int i = 0; i != h_view.rank; ++i) expectedCounter *= h_view.extent(i);

    ASSERT_EQ(expectedCounter, counter);
  }

  static void test_for3_MDTeamThreadRange(const int N0, const int N1,
                                          const int N2) {
    using DataType     = int;
    using ViewType     = typename Kokkos::View<DataType***, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;
    using TeamType     = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

    const int startIdx[3] = {0};
    const int initValue   = 3;

    ViewType v("v", N0, N1, N2);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(N0, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange = Kokkos::MDTeamThreadRange(team, N1, N2);

          Kokkos::parallel_for(
            teamRange,
            [=](int i, int j) {
              v(leagueRank, i, j) = initValue;
            });
        }
    );

    HostViewType h_view = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(h_view, v);

    check_result_3D("test_for3_MDTeamThreadRange", h_view, startIdx, initValue);
  }

  template <typename HostViewType>
  static void check_result_8D(std::string testName, HostViewType h_view,
                              const int startIdx[8], int expectedValue) {
    int counter = 0;
    int dim     = Kokkos::rank(h_view);

    for (int i = startIdx[0]; i < h_view.extent(0); ++i) {
      for (int j = startIdx[1]; j < h_view.extent(1); ++j) {
        for (int k = startIdx[2]; k < h_view.extent(2); ++k) {
          for (int l = startIdx[3]; l < h_view.extent(3); ++l) {
            for (int m = startIdx[4]; m < h_view.extent(4); ++m) {
              for (int n = startIdx[5]; n < h_view.extent(5); ++n) {
                for (int o = startIdx[6]; o < h_view.extent(6); ++o) {
                  for (int p = startIdx[7]; p < h_view.extent(7); ++p) {
                    if (h_view(i, j, k, l, m, n, o, p) == expectedValue) {
                      ++counter;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    int expectedCounter = 1;
    for (int i = 0; i != 8; ++i) expectedCounter *= h_view.extent(i);

    ASSERT_EQ(expectedCounter, counter);
  }

  static void test_for8_MDTeamThreadRange(const int N0, const int N1,
                                          const int N2, const int N3,
                                          const int N4, const int N5,
                                          const int N6, const int N7) {
    using DataType     = int;
    using ViewType     = typename Kokkos::View<DataType********, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;
    using TeamType     = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

    const int startIdx[8] = {0};
    const int initValue   = 3;

    ViewType v("v", N0, N1, N2, N3, N4, N5, N6, N7);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(N0, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          Kokkos::parallel_for(
              Kokkos::MDTeamThreadRange(team, N1, N2, N3, N4, N5, N6, N7),
              [=](int i, int j, int k, int l, int m, int n, int o) {
                v(leagueRank, i, j, k, l, m, n, o) = initValue;
              });
        });

    HostViewType h_view = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(h_view, v);

    check_result_8D("test_for8_MDTeamThreadRange", h_view, startIdx, initValue);
  }

  // template <Kokkos::Iterate Direction>
  // static void test_for8_MDTeamThreadRange_with_direction(
  //     const int N0, const int N1, const int N2, const int N3, const int N4,
  //     const int N5, const int N6, const int N7) {
  //   using DataType     = int;
  //   using ViewType     = typename Kokkos::View<DataType********, ExecSpace>;
  //   using HostViewType = typename ViewType::HostMirror;

  //   const int startIdx[8] = {0};
  //   const int initValue   = 3;

  //   ViewType v("v", N0, N1, N2, N3, N4, N5, N6, N7);

  //   Kokkos::parallel_for(
  //       Kokkos::TeamPolicy<ExecSpace>(N0, Kokkos::AUTO),
  //       KOKKOS_LAMBDA(const auto& team) {
  //         int leagueRank = team.league_rank();

  //         Kokkos::parallel_for(
  //             Kokkos::MDTeamThreadRange<Direction>(team, N1, N2, N3, N4, N5, N6,
  //                                                  N7),
  //             KOKKOS_LAMBDA(int i, int j, int k, int l, int m, int n, int o) {
  //               v(leagueRank, i, j, k, l, m, n, o) = initValue;
  //             });
  //       });

  //   HostViewType h_view = Kokkos::create_mirror_view(v);
  //   Kokkos::deep_copy(h_view, v);

  //   check_result_8D("test_for8_MDTeamThreadRange", h_view, startIdx, initValue);
  // }

  static void test_for2_MDThreadVectorRange(const int N0, const int N1) {
    using DataType     = int;
    using ViewType     = typename Kokkos::View<DataType**, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;
    using TeamType     = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

    const int s0 = 0;
    const int s1 = 0;

    ViewType v("v", N0, N1);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(1, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          Kokkos::parallel_for(
              Kokkos::MDThreadVectorRange(team, N0, N1),
              [=](int i, int j) { v(i, j) = 3; });
        });

    HostViewType h_view = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(h_view, v);

    int counter = 0;
    for (int i = s0; i < N0; ++i) {
      for (int j = s1; j < N1; ++j) {
        if (h_view(i, j) != 3) {
          ++counter;
        }
      }
    }

    if (counter != 0) {
      printf(
          "Offset Start + Default Layouts + InitTag op(): Errors in "
          "test_for2; mismatches = %d\n\n",
          counter);
    }

    ASSERT_EQ(counter, 0);
  }

#ifdef FOUND_CUDA_WORKAROUND
  static void test_for8_MDThreadVectorRange(const int N0, const int N1,
                                            const int N2, const int N3,
                                            const int N4, const int N5,
                                            const int N6, const int N7) {
    using DataType     = int;
    using ViewType     = typename Kokkos::View<DataType********, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    const int s0 = 0;
    const int s1 = 0;
    const int s2 = 0;
    const int s3 = 0;
    const int s4 = 0;
    const int s5 = 0;
    const int s6 = 0;
    const int s7 = 0;

    ViewType v("v", N0, N1, N2, N3, N4, N5, N6, N7);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(1, Kokkos::AUTO),
        KOKKOS_LAMBDA(const auto& team) {
          Kokkos::parallel_for(
              Kokkos::MDThreadVectorRange(team, N0, N1, N2, N3, N4, N5, N6, N7),
              KOKKOS_LAMBDA(int i, int j, int k, int l, int m, int n, int o,
                            int p) { v(i, j, k, l, m, n, o, p) = 3; });
        });

    HostViewType h_view = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(h_view, v);

    int counter = 0;
    for (int i = s0; i < N0; ++i) {
      for (int j = s1; j < N1; ++j) {
        for (int k = s2; k < N2; ++k) {
          for (int l = s3; l < N3; ++l) {
            for (int m = s4; m < N4; ++m) {
              for (int n = s5; n < N5; ++n) {
                for (int o = s6; o < N6; ++o) {
                  for (int p = s7; p < N7; ++p) {
                    if (h_view(i, j, k, l, m, n, o, p) == 3) {
                      ++counter;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    const int expectedCounter = N0 * N1 * N2 * N3 * N4 * N5 * N6 * N7;

    ASSERT_EQ(expectedCounter, counter);
  }

  template <Kokkos::Iterate OuterDirection, Kokkos::Iterate InnerDirection>
  static void test_for8_MDThreadVectorRange_with_direction(
      const int N0, const int N1, const int N2, const int N3, const int N4,
      const int N5, const int N6, const int N7) {
    using DataType     = int;
    using ViewType     = typename Kokkos::View<DataType********, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    const int s0 = 0;
    const int s1 = 0;
    const int s2 = 0;
    const int s3 = 0;
    const int s4 = 0;
    const int s5 = 0;
    const int s6 = 0;
    const int s7 = 0;

    ViewType v("v", N0, N1, N2, N3, N4, N5, N6, N7);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(1, Kokkos::AUTO),
        KOKKOS_LAMBDA(const auto& team) {
          Kokkos::parallel_for(
              Kokkos::MDThreadVectorRange<OuterDirection, InnerDirection>(
                  team, N0, N1, N2, N3, N4, N5, N6, N7),
              KOKKOS_LAMBDA(int i, int j, int k, int l, int m, int n, int o,
                            int p) { v(i, j, k, l, m, n, o, p) = 3; });
        });

    HostViewType h_view = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(h_view, v);

    int counter = 0;
    for (int i = s0; i < N0; ++i) {
      for (int j = s1; j < N1; ++j) {
        for (int k = s2; k < N2; ++k) {
          for (int l = s3; l < N3; ++l) {
            for (int m = s4; m < N4; ++m) {
              for (int n = s5; n < N5; ++n) {
                for (int o = s6; o < N6; ++o) {
                  for (int p = s7; p < N7; ++p) {
                    if (h_view(i, j, k, l, m, n, o, p) == 3) {
                      ++counter;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    const int expectedCounter = N0 * N1 * N2 * N3 * N4 * N5 * N6 * N7;

    ASSERT_EQ(expectedCounter, counter);
  }

  template <Kokkos::Iterate OuterDirection, Kokkos::Iterate InnerDirection>
  static void test_for2_MDThreadVectorRange_with_direction(const int N0,
                                                           const int N1) {
    using DataType     = int;
    using ViewType     = typename Kokkos::View<DataType**, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    const int s0 = 0;
    const int s1 = 0;

    ViewType v("v", N0, N1);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(1, Kokkos::AUTO),
        KOKKOS_LAMBDA(const auto& team) {
          Kokkos::parallel_for(
              Kokkos::MDThreadVectorRange<OuterDirection, InnerDirection>(
                  team, N0, N1),
              KOKKOS_LAMBDA(int i, int j) { v(i, j) = 3; });
        });

    HostViewType h_view = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(h_view, v);

    int counter = 0;
    for (int i = s0; i < N0; ++i) {
      for (int j = s1; j < N1; ++j) {
        if (h_view(i, j) != 3) {
          ++counter;
        }
      }
    }

    if (counter != 0) {
      printf(
          "Offset Start + Default Layouts + InitTag op(): Errors in "
          "test_for2; mismatches = %d\n\n",
          counter);
    }

    ASSERT_EQ(counter, 0);
  }
#endif

  static void test_for2_MDTeamVectorRange(const int numTeams, const int N0,
                                          const int N1) {
    using DataType     = int;
    using ViewType     = typename Kokkos::View<DataType***, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;
    using TeamType     = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

    const int s0 = 0;
    const int s1 = 0;
    const int s2 = 0;

    ViewType v("v", numTeams, N0, N1);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(numTeams, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int i = team.league_rank();
          Kokkos::parallel_for(
              Kokkos::MDTeamVectorRange(team, N0, N1),
              [=](int j, int k) { v(i, j, k) = 3; });
        });

    HostViewType h_view = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(h_view, v);

    int counter = 0;
    for (int i = s0; i < numTeams; ++i) {
      for (int j = s1; j < N0; ++j) {
        for (int k = s2; k < N1; ++k) {
          if (h_view(i, j, k) == 3) {
            ++counter;
          }
        }
      }
    }

    int expectedTotalCounter = numTeams * N0 * N1;
    ASSERT_EQ(counter, expectedTotalCounter);
  }

#ifdef FOUND_CUDA_WORKAROUND
  template <Kokkos::Iterate OuterDirection, Kokkos::Iterate InnerDirection>
  static void test_for2_MDTeamVectorRange_with_direction(const int numTeams,
                                                         const int N0,
                                                         const int N1) {
    using DataType     = int;
    using ViewType     = typename Kokkos::View<DataType***, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    const int s0 = 0;
    const int s1 = 0;
    const int s2 = 0;

    ViewType v("v", numTeams, N0, N1);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(numTeams, Kokkos::AUTO),
        KOKKOS_LAMBDA(const auto& team) {
          int i = team.league_rank();
          Kokkos::parallel_for(
              Kokkos::MDTeamVectorRange<OuterDirection, InnerDirection>(team,
                                                                        N0, N1),
              KOKKOS_LAMBDA(int j, int k) { v(i, j, k) = 3; });
        });

    HostViewType h_view = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(h_view, v);

    int counter = 0;
    for (int i = s0; i < numTeams; ++i) {
      for (int j = s1; j < N0; ++j) {
        for (int k = s2; k < N1; ++k) {
          if (h_view(i, j, k) == 3) {
            ++counter;
          }
        }
      }
    }

    int expectedTotalCounter = numTeams * N0 * N1;
    ASSERT_EQ(counter, expectedTotalCounter);
  }

  template <Kokkos::Iterate OuterDirection, Kokkos::Iterate InnerDirection>
  static void test_for3_with_direction(const int N0, const int N1,
                                       const int N2) {
    using DataType     = int;
    using ViewType     = typename Kokkos::View<DataType***, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    const int s0 = 0;
    const int s1 = 0;
    const int s2 = 0;

    ViewType v("v", N0, N1, N2);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(1, Kokkos::AUTO),
        KOKKOS_LAMBDA(const auto& team) {
          Kokkos::parallel_for(
              Kokkos::MDTeamVectorRange<OuterDirection, InnerDirection>(
                  team, N0, N1, N2),
              KOKKOS_LAMBDA(int i, int j, int k) { v(i, j, k) = 3; });
        });

    HostViewType h_view = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(h_view, v);

    int counter = 0;
    for (int i = s0; i < N0; ++i) {
      for (int j = s1; j < N1; ++j) {
        for (int k = s2; k < N2; ++k) {
          if (h_view(i, j, k) != 3) {
            ++counter;
          }
        }
      }
    }

    if (counter != 0) {
      printf(
          "Offset Start + Default Layouts + InitTag op(): Errors in "
          "test_for2; mismatches = %d\n\n",
          counter);
    }

    ASSERT_EQ(counter, 0);
  }
#endif
};

template <typename ExecSpace>
struct TestMDParallelReduce {
  using TeamType = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  static void test_reduce2_MDTeamThreadRange(const int N0, const int N1) {
    using DataType     = int;
    using ViewType     = typename Kokkos::View<DataType**, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = 1;
    int teamSize = 1;

    ViewType v("v", N0, N1);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N0, N1}),
        KOKKOS_LAMBDA(const int& i, const int& j) { v(i, j) = 3; });

    using IntViewType     = typename Kokkos::View<int*, ExecSpace>;
    using HostIntViewType = typename IntViewType::HostMirror;

    IntViewType sharedArray("t", leagueSize);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, teamSize),
        KOKKOS_LAMBDA(const TeamType& team) {

          auto leagueRank = team.league_rank();
          int teamSum = 0;

          Kokkos::parallel_reduce(
              Kokkos::MDTeamThreadRange(team, N0, N1),
              [=](const int& i, const int& j, int& threadSum) {
                threadSum += v(i, j);
              },
              teamSum);

          Kokkos::single(Kokkos::PerTeam(team), [&] { sharedArray(leagueRank) += teamSum; });
        }
    );

    auto h_sharedArray = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sharedArray);

    ASSERT_EQ(h_sharedArray(0), 3 * N0 * N1);
  }

  static void test_reduce2_MDThreadVectorRange(const int N0, const int N1) {
    using DataType     = int;
    using ViewType     = typename Kokkos::View<DataType**, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = 1;
    int teamSize = 1;

    ViewType v("v", N0, N1);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N0, N1}),
        KOKKOS_LAMBDA(const int& i, const int& j) { v(i, j) = 3; });

    using IntViewType     = typename Kokkos::View<int*, ExecSpace>;
    using HostIntViewType = typename IntViewType::HostMirror;

    IntViewType sharedArray("t", leagueSize);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, teamSize),
        KOKKOS_LAMBDA(const TeamType& team) {

          auto leagueRank = team.league_rank();
          int teamSum = 0;

          Kokkos::parallel_reduce(
              Kokkos::MDThreadVectorRange(team, N0, N1),
              [=](const int& i, const int& j, int& threadSum) {
                threadSum += v(i, j);
              },
              teamSum);

          Kokkos::single(Kokkos::PerTeam(team), [&] { sharedArray(leagueRank) += teamSum; });
        }
    );

    auto h_sharedArray = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sharedArray);

    ASSERT_EQ(h_sharedArray(0), 3 * N0 * N1);
  }

  static void test_reduce2_MDTeamVectorRange(const int N0, const int N1) {
    using DataType     = int;
    using ViewType     = typename Kokkos::View<DataType**, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = 1;
    int teamSize = 1;

    ViewType v("v", N0, N1);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N0, N1}),
        KOKKOS_LAMBDA(const int& i, const int& j) { v(i, j) = 3; });

    using IntViewType     = typename Kokkos::View<int*, ExecSpace>;
    using HostIntViewType = typename IntViewType::HostMirror;

    IntViewType sharedArray("t", leagueSize);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(leagueSize, teamSize),
        KOKKOS_LAMBDA(const TeamType& team) {

          auto leagueRank = team.league_rank();
          int teamSum = 0;

          Kokkos::parallel_reduce(
              Kokkos::MDTeamVectorRange(team, N0, N1),
              [=](const int& i, const int& j, int& threadSum) {
                threadSum += v(i, j);
              },
              teamSum);

          Kokkos::single(Kokkos::PerTeam(team), [&] { sharedArray(leagueRank) += teamSum; });
        }
    );

    auto h_sharedArray = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sharedArray);

    //FIXME h_sharedArray should be 2D?
    ASSERT_EQ(h_sharedArray(0), 3 * N0 * N1);
  }

#ifdef FOUND_CUDA_WORKAROUND
  template <Kokkos::Iterate Direction>
  static void test_reduce2_MDTeamThreadRange_with_direction(const int N0,
                                                            const int N1) {
    using DataType     = int;
    using ViewType     = typename Kokkos::View<DataType**, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    ViewType v("v", N0, N1);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N0, N1}),
        KOKKOS_LAMBDA(const auto& i, const auto& j) { v(i, j) = 3; });

    int totalSum = 0;

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(1, Kokkos::AUTO),
        [=, &totalSum](const auto& team) {
          int teamSum = 0;

          Kokkos::parallel_reduce(
              Kokkos::MDTeamThreadRange<Direction>(team, N0, N1),
              KOKKOS_LAMBDA(const int& i, const int& j, int& threadSum) {
                threadSum += v(i, j);
              },
              teamSum);

          totalSum = teamSum;
        });

    ASSERT_EQ(totalSum, 3 * N0 * N1);
  }

  template <Kokkos::Iterate OuterDirection, Kokkos::Iterate InnerDirection>
  static void test_reduce2_MDThreadVectorRange_with_direction(const int N0,
                                                              const int N1) {
    using DataType     = int;
    using ViewType     = typename Kokkos::View<DataType**, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    ViewType v("v", N0, N1);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N0, N1}),
        KOKKOS_LAMBDA(const auto& i, const auto& j) { v(i, j) = 3; });

    int totalSum = 0;

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(1, Kokkos::AUTO),
        [=, &totalSum](const auto& team) {
          int teamSum = 0;

          Kokkos::parallel_reduce(
              Kokkos::MDThreadVectorRange<OuterDirection, InnerDirection>(
                  team, N0, N1),
              KOKKOS_LAMBDA(const int& i, const int& j, int& threadSum) {
                threadSum += v(i, j);
              },
              teamSum);

          totalSum = teamSum;
        });

    ASSERT_EQ(totalSum, 3 * N0 * N1);
  }

  template <Kokkos::Iterate OuterDirection, Kokkos::Iterate InnerDirection>
  static void test_reduce3_MDTeamVectorRange_with_direction(const int teamSize,
                                                            const int N0,
                                                            const int N1) {
    using DataType           = int;
    using ViewType           = typename Kokkos::View<DataType***, ExecSpace>;
    using ResultViewType     = typename Kokkos::View<DataType*, ExecSpace>;
    using ResultHostViewType = typename ResultViewType::HostMirror;

    ViewType v("v", teamSize, N0, N1);
    ResultViewType r("r", teamSize);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {teamSize, N0, N1}),
        KOKKOS_LAMBDA(const auto& l, const auto& i, const auto& j) {
          v(l, i, j) = 3;
        });

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(teamSize, Kokkos::AUTO),
        KOKKOS_LAMBDA(const auto& team) {
          int l       = team.league_rank();
          int teamSum = 0;

          Kokkos::parallel_reduce(
              Kokkos::MDTeamVectorRange<OuterDirection, InnerDirection>(team,
                                                                        N0, N1),
              KOKKOS_LAMBDA(const int& i, const int& j, int& threadSum) {
                threadSum += v(l, i, j);
              },
              teamSum);

          r(l) = teamSum;
        });

    auto hostView = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r);

    int total = 0;
    for (int i = 0; i < r.extent(0); ++i) {
      total += r(i);
    }

    ASSERT_EQ(total, 3 * teamSize * N0 * N1);
  }
#endif
};

template <class ExecSpace, class ScheduleType>
struct TestMDParallelism {
  using team_member =
      typename Kokkos::TeamPolicy<ScheduleType, ExecSpace>::member_type;
  using view_type = Kokkos::View<int**, ExecSpace>;

  view_type m_flags;

  TestMDParallelism(const size_t league_size)
      : m_flags(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "flags"),
  // FIXME_OPENMPTARGET temporary restriction for team size to be at
  // least 32
#ifdef KOKKOS_ENABLE_OPENMPTARGET
            Kokkos::TeamPolicy<ScheduleType, ExecSpace>(1, 32).team_size_max(
                *this, Kokkos::ParallelReduceTag()),
#else
            Kokkos::TeamPolicy<ScheduleType, ExecSpace>(1, 1).team_size_max(
                *this, Kokkos::ParallelReduceTag()),
#endif
            league_size) {
  }

  struct VerifyInitTag {};

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member& member) const {
    const int tid =
        member.team_rank() + member.team_size() * member.league_rank();

    m_flags(member.team_rank(), member.league_rank()) = tid;
    static_assert(
        (std::is_same<typename team_member::execution_space, ExecSpace>::value),
        "TeamMember::execution_space is not the same as "
        "TeamPolicy<>::execution_space");
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const VerifyInitTag&, const team_member& member) const {
    const int tid =
        member.team_rank() + member.team_size() * member.league_rank();

    if (tid != m_flags(member.team_rank(), member.league_rank())) {
      KOKKOS_IMPL_DO_NOT_USE_PRINTF(
          "TestMDParallelism member(%d,%d) error %d != %d\n",
          member.league_rank(), member.team_rank(), tid,
          m_flags(member.team_rank(), member.league_rank()));
    }
  }

  // Included for test_small_league_size.
  TestMDParallelism() : m_flags() {}

  // Included for test_small_league_size.
  struct NoOpTag {};

  KOKKOS_INLINE_FUNCTION
  void operator()(const NoOpTag&, const team_member& /*member*/) const {}

  static void test_small_league_size() {
    int bs = 8;   // batch size (number of elements per batch)
    int ns = 16;  // total number of "problems" to process

    // Calculate total scratch memory space size.
    const int level     = 0;
    int mem_size        = 960;
    const int num_teams = ns / bs;
    Kokkos::TeamPolicy<ExecSpace, NoOpTag> policy(num_teams, Kokkos::AUTO());

    Kokkos::parallel_for(
        policy.set_scratch_size(level, Kokkos::PerTeam(mem_size),
                                Kokkos::PerThread(0)),
        TestMDParallelism());
  }

  static void test_constructors() {
    constexpr const int smallest_work = 1;
    // FIXME_OPENMPTARGET temporary restriction for team size to be at
    // least 32
#ifdef KOKKOS_ENABLE_OPENMPTARGET
    Kokkos::TeamPolicy<ExecSpace, NoOpTag> none_auto(smallest_work, 32,
                                                     smallest_work);
#else
    Kokkos::TeamPolicy<ExecSpace, NoOpTag> none_auto(
        smallest_work, smallest_work, smallest_work);
#endif
    (void)none_auto;
    Kokkos::TeamPolicy<ExecSpace, NoOpTag> both_auto(
        smallest_work, Kokkos::AUTO(), Kokkos::AUTO());
    (void)both_auto;
    // FIXME_OPENMPTARGET temporary restriction for team size to be at
    // least 32
#ifdef KOKKOS_ENABLE_OPENMPTARGET
    Kokkos::TeamPolicy<ExecSpace, NoOpTag> auto_vector(smallest_work, 32,
                                                       Kokkos::AUTO());
#else
    Kokkos::TeamPolicy<ExecSpace, NoOpTag> auto_vector(
        smallest_work, smallest_work, Kokkos::AUTO());
#endif
    (void)auto_vector;
    Kokkos::TeamPolicy<ExecSpace, NoOpTag> auto_team(
        smallest_work, Kokkos::AUTO(), smallest_work);
    (void)auto_team;
  }

  static void test_for(const size_t league_size) {
    {
      TestMDParallelism functor(league_size);
      using policy_type = Kokkos::TeamPolicy<ScheduleType, ExecSpace>;
      using policy_type_init =
          Kokkos::TeamPolicy<ScheduleType, ExecSpace, VerifyInitTag>;

      // FIXME_OPENMPTARGET temporary restriction for team size to be
      // at least 32
#ifdef KOKKOS_ENABLE_OPENMPTARGET
      const int team_size =
          policy_type(league_size, 32)
              .team_size_max(functor, Kokkos::ParallelForTag());
      const int team_size_init =
          policy_type_init(league_size, 32)
              .team_size_max(functor, Kokkos::ParallelForTag());
#else
      const int team_size =
          policy_type(league_size, 1)
              .team_size_max(functor, Kokkos::ParallelForTag());
      const int team_size_init =
          policy_type_init(league_size, 1)
              .team_size_max(functor, Kokkos::ParallelForTag());
#endif

      Kokkos::parallel_for(policy_type(league_size, team_size), functor);
      Kokkos::parallel_for(policy_type_init(league_size, team_size_init),
                           functor);
    }

    test_small_league_size();
    test_constructors();
  }

#if 0 
  struct ReduceTag {};

  using value_type = int64_t;

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member &member, value_type &update) const {
    update += member.team_rank() + member.team_size() * member.league_rank();
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const ReduceTag &, const team_member &member,
                  value_type &update) const {
    update +=
        1 + member.team_rank() + member.team_size() * member.league_rank();
  }

  static void test_reduce(const size_t league_size) {
    TestTeamPolicy functor(league_size);

    using policy_type = Kokkos::TeamPolicy<ScheduleType, ExecSpace>;
    using policy_type_reduce =
        Kokkos::TeamPolicy<ScheduleType, ExecSpace, ReduceTag>;

    // FIXME_OPENMPTARGET temporary restriction for team size to be at least 32
#ifdef KOKKOS_ENABLE_OPENMPTARGET
    const int team_size =
        policy_type_reduce(league_size, 32)
            .team_size_max(functor, Kokkos::ParallelReduceTag());
#else
    const int team_size =
        policy_type_reduce(league_size, 1)
            .team_size_max(functor, Kokkos::ParallelReduceTag());
#endif

    const int64_t N = team_size * league_size;

    int64_t total = 0;

    Kokkos::parallel_reduce(policy_type(league_size, team_size), functor,
                            total);
    ASSERT_EQ(size_t((N - 1) * (N)) / 2, size_t(total));

    Kokkos::parallel_reduce(policy_type_reduce(league_size, team_size), functor,
                            total);
    ASSERT_EQ((size_t(N) * size_t(N + 1)) / 2, size_t(total));
  }
#endif
};

}  // namespace MDParallelism

}  // namespace Test

/*--------------------------------------------------------------------------*/

#if 0
namespace Test {

template <typename ScalarType, class DeviceType, class ScheduleType>
class ReduceTeamFunctor {
 public:
  using execution_space = DeviceType;
  using policy_type     = Kokkos::TeamPolicy<ScheduleType, execution_space>;
  using size_type       = typename execution_space::size_type;

  struct value_type {
    ScalarType value[3];
  };

  const size_type nwork;

  KOKKOS_INLINE_FUNCTION
  ReduceTeamFunctor(const size_type &arg_nwork) : nwork(arg_nwork) {}

  KOKKOS_INLINE_FUNCTION
  ReduceTeamFunctor(const ReduceTeamFunctor &rhs) : nwork(rhs.nwork) {}

  KOKKOS_INLINE_FUNCTION
  void init(value_type &dst) const {
    dst.value[0] = 0;
    dst.value[1] = 0;
    dst.value[2] = 0;
  }

  KOKKOS_INLINE_FUNCTION
  void join(volatile value_type &dst, const volatile value_type &src) const {
    dst.value[0] += src.value[0];
    dst.value[1] += src.value[1];
    dst.value[2] += src.value[2];
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const typename policy_type::member_type ind,
                  value_type &dst) const {
    const int thread_rank =
        ind.team_rank() + ind.team_size() * ind.league_rank();
    const int thread_size = ind.team_size() * ind.league_size();
    const int chunk       = (nwork + thread_size - 1) / thread_size;

    size_type iwork           = chunk * thread_rank;
    const size_type iwork_end = iwork + chunk < nwork ? iwork + chunk : nwork;

    for (; iwork < iwork_end; ++iwork) {
      dst.value[0] += 1;
      dst.value[1] += iwork + 1;
      dst.value[2] += nwork - iwork;
    }
  }
};

}  // namespace Test

namespace {

template <typename ScalarType, class DeviceType, class ScheduleType>
class TestReduceTeam {
 public:
  using execution_space = DeviceType;
  using policy_type     = Kokkos::TeamPolicy<ScheduleType, execution_space>;
  using size_type       = typename execution_space::size_type;

  TestReduceTeam(const size_type &nwork) { run_test(nwork); }

  void run_test(const size_type &nwork) {
    using functor_type =
        Test::ReduceTeamFunctor<ScalarType, execution_space, ScheduleType>;
    using value_type = typename functor_type::value_type;
    using result_type =
        Kokkos::View<value_type, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;

    enum { Count = 3 };
    enum { Repeat = 100 };

    value_type result[Repeat];

    const uint64_t nw   = nwork;
    const uint64_t nsum = nw % 2 ? nw * ((nw + 1) / 2) : (nw / 2) * (nw + 1);

    policy_type team_exec(nw, 1);

    const unsigned team_size = team_exec.team_size_recommended(
        functor_type(nwork), Kokkos::ParallelReduceTag());
    const unsigned league_size = (nwork + team_size - 1) / team_size;

    team_exec = policy_type(league_size, team_size);

    for (unsigned i = 0; i < Repeat; ++i) {
      result_type tmp(&result[i]);
      Kokkos::parallel_reduce(team_exec, functor_type(nwork), tmp);
    }

    execution_space().fence();

    for (unsigned i = 0; i < Repeat; ++i) {
      for (unsigned j = 0; j < Count; ++j) {
        const uint64_t correct = 0 == j % 3 ? nw : nsum;
        ASSERT_EQ((ScalarType)correct, result[i].value[j]);
      }
    }
  }
};

}  // namespace

/*--------------------------------------------------------------------------*/

namespace Test {

template <class DeviceType, class ScheduleType>
class ScanTeamFunctor {
 public:
  using execution_space = DeviceType;
  using policy_type     = Kokkos::TeamPolicy<ScheduleType, execution_space>;
  using value_type      = int64_t;

  Kokkos::View<value_type, execution_space> accum;
  Kokkos::View<value_type, execution_space> total;

  ScanTeamFunctor() : accum("accum"), total("total") {}

  KOKKOS_INLINE_FUNCTION
  void init(value_type &error) const { error = 0; }

  KOKKOS_INLINE_FUNCTION
  void join(value_type volatile &error,
            value_type volatile const &input) const {
    if (input) error = 1;
  }

  struct JoinMax {
    using value_type = int64_t;

    KOKKOS_INLINE_FUNCTION
    void join(value_type volatile &dst,
              value_type volatile const &input) const {
      if (dst < input) dst = input;
    }
  };

  KOKKOS_INLINE_FUNCTION
  void operator()(const typename policy_type::member_type ind,
                  value_type &error) const {
    if (0 == ind.league_rank() && 0 == ind.team_rank()) {
      const int64_t thread_count = ind.league_size() * ind.team_size();
      total()                    = (thread_count * (thread_count + 1)) / 2;
    }

    // Team max:
    int64_t m = (int64_t)(ind.league_rank() + ind.team_rank());
    ind.team_reduce(Kokkos::Max<int64_t>(m));

    if (m != ind.league_rank() + (ind.team_size() - 1)) {
      KOKKOS_IMPL_DO_NOT_USE_PRINTF(
          "ScanTeamFunctor[%i.%i of %i.%i] reduce_max_answer(%li) != "
          "reduce_max(%li)\n",
          static_cast<int>(ind.league_rank()),
          static_cast<int>(ind.team_rank()),
          static_cast<int>(ind.league_size()),
          static_cast<int>(ind.team_size()),
          static_cast<long>(ind.league_rank() + (ind.team_size() - 1)),
          static_cast<long>(m));
    }

    // Scan:
    const int64_t answer = (ind.league_rank() + 1) * ind.team_rank() +
                           (ind.team_rank() * (ind.team_rank() + 1)) / 2;

    const int64_t result =
        ind.team_scan(ind.league_rank() + 1 + ind.team_rank() + 1);

    const int64_t result2 =
        ind.team_scan(ind.league_rank() + 1 + ind.team_rank() + 1);

    if (answer != result || answer != result2) {
      KOKKOS_IMPL_DO_NOT_USE_PRINTF(
          "ScanTeamFunctor[%i.%i of %i.%i] answer(%li) != scan_first(%li) or "
          "scan_second(%li)\n",
          static_cast<int>(ind.league_rank()),
          static_cast<int>(ind.team_rank()),
          static_cast<int>(ind.league_size()),
          static_cast<int>(ind.team_size()), static_cast<long>(answer),
          static_cast<long>(result), static_cast<long>(result2));

      error = 1;
    }

    const int64_t thread_rank =
        ind.team_rank() + ind.team_size() * ind.league_rank();
    ind.team_scan(1 + thread_rank, accum.data());
  }
};

template <class DeviceType, class ScheduleType>
class TestScanTeam {
 public:
  using execution_space = DeviceType;
  using value_type      = int64_t;
  using policy_type     = Kokkos::TeamPolicy<ScheduleType, execution_space>;
  using functor_type    = Test::ScanTeamFunctor<DeviceType, ScheduleType>;

  TestScanTeam(const size_t nteam) { run_test(nteam); }

  void run_test(const size_t nteam) {
    using result_type =
        Kokkos::View<int64_t, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;

    const unsigned REPEAT = 100000;
    unsigned Repeat;

    if (nteam == 0) {
      Repeat = 1;
    } else {
      Repeat = (REPEAT + nteam - 1) / nteam;  // Error here.
    }

    functor_type functor;

    policy_type team_exec(nteam, 1);
    team_exec = policy_type(
        nteam, team_exec.team_size_max(functor, Kokkos::ParallelReduceTag()));

    for (unsigned i = 0; i < Repeat; ++i) {
      int64_t accum = 0;
      int64_t total = 0;
      int64_t error = 0;
      Kokkos::deep_copy(functor.accum, total);

      Kokkos::parallel_reduce(team_exec, functor, result_type(&error));
      DeviceType().fence();

      Kokkos::deep_copy(accum, functor.accum);
      Kokkos::deep_copy(total, functor.total);

      ASSERT_EQ(error, 0);
      ASSERT_EQ(total, accum);
    }

    execution_space().fence();
  }
};

}  // namespace Test

/*--------------------------------------------------------------------------*/

namespace Test {

template <class ExecSpace, class ScheduleType>
struct SharedTeamFunctor {
  using execution_space = ExecSpace;
  using value_type      = int;
  using policy_type     = Kokkos::TeamPolicy<ScheduleType, execution_space>;

  enum { SHARED_COUNT = 1000 };

  using shmem_space = typename ExecSpace::scratch_memory_space;

  // TBD: MemoryUnmanaged should be the default for shared memory space.
  using shared_int_array_type =
      Kokkos::View<int *, shmem_space, Kokkos::MemoryUnmanaged>;

  // Tell how much shared memory will be required by this functor.
  inline unsigned team_shmem_size(int /*team_size*/) const {
    return shared_int_array_type::shmem_size(SHARED_COUNT) +
           shared_int_array_type::shmem_size(SHARED_COUNT);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const typename policy_type::member_type &ind,
                  value_type &update) const {
    const shared_int_array_type shared_A(ind.team_shmem(), SHARED_COUNT);
    const shared_int_array_type shared_B(ind.team_shmem(), SHARED_COUNT);

    if ((shared_A.data() == nullptr && SHARED_COUNT > 0) ||
        (shared_B.data() == nullptr && SHARED_COUNT > 0)) {
      KOKKOS_IMPL_DO_NOT_USE_PRINTF(
          "member( %i/%i , %i/%i ) Failed to allocate shared memory of size "
          "%lu\n",
          static_cast<int>(ind.league_rank()),
          static_cast<int>(ind.league_size()),
          static_cast<int>(ind.team_rank()), static_cast<int>(ind.team_size()),
          static_cast<unsigned long>(SHARED_COUNT));

      ++update;  // Failure to allocate is an error.
    } else {
      for (int i = ind.team_rank(); i < SHARED_COUNT; i += ind.team_size()) {
        shared_A[i] = i + ind.league_rank();
        shared_B[i] = 2 * i + ind.league_rank();
      }

      ind.team_barrier();

      if (ind.team_rank() + 1 == ind.team_size()) {
        for (int i = 0; i < SHARED_COUNT; ++i) {
          if (shared_A[i] != i + ind.league_rank()) {
            ++update;
          }

          if (shared_B[i] != 2 * i + ind.league_rank()) {
            ++update;
          }
        }
      }
    }
  }
};

}  // namespace Test

namespace {

template <class ExecSpace, class ScheduleType>
struct TestSharedTeam {
  TestSharedTeam() { run(); }

  void run() {
    using Functor = Test::SharedTeamFunctor<ExecSpace, ScheduleType>;
    using result_type =
        Kokkos::View<typename Functor::value_type, Kokkos::HostSpace,
                     Kokkos::MemoryUnmanaged>;

#ifdef KOKKOS_ENABLE_OPENMPTARGET
    const size_t team_size =
        Kokkos::TeamPolicy<ScheduleType, ExecSpace>(64, 32).team_size_max(
            Functor(), Kokkos::ParallelReduceTag());

    Kokkos::TeamPolicy<ScheduleType, ExecSpace> team_exec(32 / team_size,
                                                          team_size);
#else
    const size_t team_size =
        Kokkos::TeamPolicy<ScheduleType, ExecSpace>(8192, 1).team_size_max(
            Functor(), Kokkos::ParallelReduceTag());

    Kokkos::TeamPolicy<ScheduleType, ExecSpace> team_exec(8192 / team_size,
                                                          team_size);
#endif

    typename Functor::value_type error_count = 0;

    Kokkos::parallel_reduce(team_exec, Functor(), result_type(&error_count));
    Kokkos::fence();

    ASSERT_EQ(error_count, 0);
  }
};

}  // namespace

namespace Test {

#if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
template <class MemorySpace, class ExecSpace, class ScheduleType>
struct TestLambdaSharedTeam {
  TestLambdaSharedTeam() { run(); }

  void run() {
    using Functor     = Test::SharedTeamFunctor<ExecSpace, ScheduleType>;
    using result_type = Kokkos::View<typename Functor::value_type, MemorySpace,
                                     Kokkos::MemoryUnmanaged>;

    using shmem_space = typename ExecSpace::scratch_memory_space;

    // TBD: MemoryUnmanaged should be the default for shared memory space.
    using shared_int_array_type =
        Kokkos::View<int *, shmem_space, Kokkos::MemoryUnmanaged>;

    const int SHARED_COUNT = 1000;
#ifdef KOKKOS_ENABLE_OPENMPTARGET
    int team_size = 32;
#else
    int team_size = 1;
#endif

#ifdef KOKKOS_ENABLE_CUDA
    if (std::is_same<ExecSpace, Kokkos::Cuda>::value) team_size = 128;
#endif

    Kokkos::TeamPolicy<ScheduleType, ExecSpace> team_exec(8192 / team_size,
                                                          team_size);
    team_exec = team_exec.set_scratch_size(
        0, Kokkos::PerTeam(SHARED_COUNT * 2 * sizeof(int)));

    typename Functor::value_type error_count = 0;

    Kokkos::parallel_reduce(
        team_exec,
        KOKKOS_LAMBDA(
            const typename Kokkos::TeamPolicy<ScheduleType,
                                              ExecSpace>::member_type &ind,
            int &update) {
          const shared_int_array_type shared_A(ind.team_shmem(), SHARED_COUNT);
          const shared_int_array_type shared_B(ind.team_shmem(), SHARED_COUNT);

          if ((shared_A.data() == nullptr && SHARED_COUNT > 0) ||
              (shared_B.data() == nullptr && SHARED_COUNT > 0)) {
            KOKKOS_IMPL_DO_NOT_USE_PRINTF(
                "Failed to allocate shared memory of size %lu\n",
                static_cast<unsigned long>(SHARED_COUNT));

            ++update;  // Failure to allocate is an error.
          } else {
            for (int i = ind.team_rank(); i < SHARED_COUNT;
                 i += ind.team_size()) {
              shared_A[i] = i + ind.league_rank();
              shared_B[i] = 2 * i + ind.league_rank();
            }

            ind.team_barrier();

            if (ind.team_rank() + 1 == ind.team_size()) {
              for (int i = 0; i < SHARED_COUNT; ++i) {
                if (shared_A[i] != i + ind.league_rank()) {
                  ++update;
                }

                if (shared_B[i] != 2 * i + ind.league_rank()) {
                  ++update;
                }
              }
            }
          }
        },
        result_type(&error_count));

    Kokkos::fence();

    ASSERT_EQ(error_count, 0);
  }
};
#endif

}  // namespace Test

namespace Test {

template <class ExecSpace, class ScheduleType>
struct ScratchTeamFunctor {
  using execution_space = ExecSpace;
  using value_type      = int;
  using policy_type     = Kokkos::TeamPolicy<ScheduleType, execution_space>;

  enum { SHARED_TEAM_COUNT = 100 };
  enum { SHARED_THREAD_COUNT = 10 };

  using shmem_space = typename ExecSpace::scratch_memory_space;

  // TBD: MemoryUnmanaged should be the default for shared memory space.
  using shared_int_array_type =
      Kokkos::View<size_t *, shmem_space, Kokkos::MemoryUnmanaged>;

  KOKKOS_INLINE_FUNCTION
  void operator()(const typename policy_type::member_type &ind,
                  value_type &update) const {
    const shared_int_array_type scratch_ptr(ind.team_scratch(1),
                                            3 * ind.team_size());
    const shared_int_array_type scratch_A(ind.team_scratch(1),
                                          SHARED_TEAM_COUNT);
    const shared_int_array_type scratch_B(ind.thread_scratch(1),
                                          SHARED_THREAD_COUNT);

    if ((scratch_ptr.data() == nullptr) ||
        (scratch_A.data() == nullptr && SHARED_TEAM_COUNT > 0) ||
        (scratch_B.data() == nullptr && SHARED_THREAD_COUNT > 0)) {
      KOKKOS_IMPL_DO_NOT_USE_PRINTF(
          "Failed to allocate shared memory of size %lu\n",
          static_cast<unsigned long>(SHARED_TEAM_COUNT));

      ++update;  // Failure to allocate is an error.
    } else {
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(ind, 0, (int)SHARED_TEAM_COUNT),
          [&](const int &i) { scratch_A[i] = i + ind.league_rank(); });

      for (int i = 0; i < SHARED_THREAD_COUNT; i++) {
        scratch_B[i] = 10000 * ind.league_rank() + 100 * ind.team_rank() + i;
      }

      scratch_ptr[ind.team_rank()]                   = (size_t)scratch_A.data();
      scratch_ptr[ind.team_rank() + ind.team_size()] = (size_t)scratch_B.data();

      ind.team_barrier();

      for (int i = 0; i < SHARED_TEAM_COUNT; i++) {
        if (scratch_A[i] != size_t(i + ind.league_rank())) ++update;
      }

      for (int i = 0; i < ind.team_size(); i++) {
        if (scratch_ptr[0] != scratch_ptr[i]) ++update;
      }

      if (scratch_ptr[1 + ind.team_size()] - scratch_ptr[0 + ind.team_size()] <
          SHARED_THREAD_COUNT * sizeof(size_t)) {
        ++update;
      }

      for (int i = 1; i < ind.team_size(); i++) {
        if ((scratch_ptr[i + ind.team_size()] -
             scratch_ptr[i - 1 + ind.team_size()]) !=
            (scratch_ptr[1 + ind.team_size()] -
             scratch_ptr[0 + ind.team_size()])) {
          ++update;
        }
      }
    }
  }
};

}  // namespace Test

namespace {

template <class ExecSpace, class ScheduleType>
struct TestScratchTeam {
  TestScratchTeam() { run(); }

  void run() {
    using Functor = Test::ScratchTeamFunctor<ExecSpace, ScheduleType>;
    using result_type =
        Kokkos::View<typename Functor::value_type, Kokkos::HostSpace,
                     Kokkos::MemoryUnmanaged>;
    using p_type = Kokkos::TeamPolicy<ScheduleType, ExecSpace>;

    typename Functor::value_type error_count = 0;

    int thread_scratch_size = Functor::shared_int_array_type::shmem_size(
        Functor::SHARED_THREAD_COUNT);

#ifdef KOKKOS_ENABLE_OPENMPTARGET
    p_type team_exec = p_type(64, 32).set_scratch_size(
        1,
        Kokkos::PerTeam(Functor::shared_int_array_type::shmem_size(
            Functor::SHARED_TEAM_COUNT)),
        Kokkos::PerThread(thread_scratch_size + 3 * sizeof(int)));
#else
    p_type team_exec = p_type(8192, 1).set_scratch_size(
        1,
        Kokkos::PerTeam(Functor::shared_int_array_type::shmem_size(
            Functor::SHARED_TEAM_COUNT)),
        Kokkos::PerThread(thread_scratch_size + 3 * sizeof(int)));
#endif

    const size_t team_size =
        team_exec.team_size_max(Functor(), Kokkos::ParallelReduceTag());

    int team_scratch_size =
        Functor::shared_int_array_type::shmem_size(Functor::SHARED_TEAM_COUNT) +
        Functor::shared_int_array_type::shmem_size(3 * team_size);

#ifdef KOKKOS_ENABLE_OPENMPTARGET
    team_exec = p_type(64 / team_size, team_size);
#else
    team_exec          = p_type(8192 / team_size, team_size);
#endif

    Kokkos::parallel_reduce(
        team_exec.set_scratch_size(1, Kokkos::PerTeam(team_scratch_size),
                                   Kokkos::PerThread(thread_scratch_size)),
        Functor(), result_type(&error_count));
    Kokkos::fence();
    ASSERT_EQ(error_count, 0);
  }
};

}  // namespace

namespace Test {

template <class ExecSpace>
KOKKOS_INLINE_FUNCTION int test_team_mulit_level_scratch_loop_body(
    const typename Kokkos::TeamPolicy<ExecSpace>::member_type &team) {
  Kokkos::View<double *, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      a_team1(team.team_scratch(0), 128);
  Kokkos::View<double *, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      a_thread1(team.thread_scratch(0), 16);
  Kokkos::View<double *, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      a_team2(team.team_scratch(0), 128);
  Kokkos::View<double *, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      a_thread2(team.thread_scratch(0), 16);

  Kokkos::View<double *, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      b_team1(team.team_scratch(1), 12800);
  Kokkos::View<double *, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      b_thread1(team.thread_scratch(1), 1600);
  Kokkos::View<double *, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      b_team2(team.team_scratch(1), 12800);
  Kokkos::View<double *, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      b_thread2(team.thread_scratch(1), 1600);

  Kokkos::View<double *, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      a_team3(team.team_scratch(0), 128);
  Kokkos::View<double *, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      a_thread3(team.thread_scratch(0), 16);
  Kokkos::View<double *, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      b_team3(team.team_scratch(1), 12800);
  Kokkos::View<double *, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      b_thread3(team.thread_scratch(1), 1600);

  // The explicit types for 0 and 128 are here to test TeamThreadRange accepting
  // different types for begin and end.
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, int(0), unsigned(128)),
                       [&](const int &i) {
                         a_team1(i) = 1000000 + i + team.league_rank() * 100000;
                         a_team2(i) = 2000000 + i + team.league_rank() * 100000;
                         a_team3(i) = 3000000 + i + team.league_rank() * 100000;
                       });
  team.team_barrier();

  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, int(0), unsigned(16)),
                       [&](const int &i) {
                         a_thread1(i) = 1000000 + 100000 * team.team_rank() +
                                        16 - i + team.league_rank() * 100000;
                         a_thread2(i) = 2000000 + 100000 * team.team_rank() +
                                        16 - i + team.league_rank() * 100000;
                         a_thread3(i) = 3000000 + 100000 * team.team_rank() +
                                        16 - i + team.league_rank() * 100000;
                       });

  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, int(0), unsigned(12800)),
                       [&](const int &i) {
                         b_team1(i) = 1000000 + i + team.league_rank() * 100000;
                         b_team2(i) = 2000000 + i + team.league_rank() * 100000;
                         b_team3(i) = 3000000 + i + team.league_rank() * 100000;
                       });
  team.team_barrier();

  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, 1600),
                       [&](const int &i) {
                         b_thread1(i) = 1000000 + 100000 * team.team_rank() +
                                        16 - i + team.league_rank() * 100000;
                         b_thread2(i) = 2000000 + 100000 * team.team_rank() +
                                        16 - i + team.league_rank() * 100000;
                         b_thread3(i) = 3000000 + 100000 * team.team_rank() +
                                        16 - i + team.league_rank() * 100000;
                       });

  team.team_barrier();

  int error = 0;
  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, 0, 128), [&](const int &i) {
        if (a_team1(i) != 1000000 + i + team.league_rank() * 100000) error++;
        if (a_team2(i) != 2000000 + i + team.league_rank() * 100000) error++;
        if (a_team3(i) != 3000000 + i + team.league_rank() * 100000) error++;
      });
  team.team_barrier();

  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, 16), [&](const int &i) {
    if (a_thread1(i) != 1000000 + 100000 * team.team_rank() + 16 - i +
                            team.league_rank() * 100000)
      error++;
    if (a_thread2(i) != 2000000 + 100000 * team.team_rank() + 16 - i +
                            team.league_rank() * 100000)
      error++;
    if (a_thread3(i) != 3000000 + 100000 * team.team_rank() + 16 - i +
                            team.league_rank() * 100000)
      error++;
  });

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, 0, 12800), [&](const int &i) {
        if (b_team1(i) != 1000000 + i + team.league_rank() * 100000) error++;
        if (b_team2(i) != 2000000 + i + team.league_rank() * 100000) error++;
        if (b_team3(i) != 3000000 + i + team.league_rank() * 100000) error++;
      });
  team.team_barrier();

  Kokkos::parallel_for(
      Kokkos::ThreadVectorRange(team, 1600), [&](const int &i) {
        if (b_thread1(i) != 1000000 + 100000 * team.team_rank() + 16 - i +
                                team.league_rank() * 100000)
          error++;
        if (b_thread2(i) != 2000000 + 100000 * team.team_rank() + 16 - i +
                                team.league_rank() * 100000)
          error++;
        if (b_thread3(i) != 3000000 + 100000 * team.team_rank() + 16 - i +
                                team.league_rank() * 100000)
          error++;
      });

  return error;
}

struct TagReduce {};
struct TagFor {};

template <class ExecSpace, class ScheduleType>
struct ClassNoShmemSizeFunction {
  using member_type =
      typename Kokkos::TeamPolicy<ExecSpace, ScheduleType>::member_type;

  Kokkos::View<int, ExecSpace, Kokkos::MemoryTraits<Kokkos::Atomic> > errors;

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagFor &, const member_type &team) const {
    int error = test_team_mulit_level_scratch_loop_body<ExecSpace>(team);
    errors() += error;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagReduce &, const member_type &team,
                  int &error) const {
    error += test_team_mulit_level_scratch_loop_body<ExecSpace>(team);
  }

  void run() {
    Kokkos::View<int, ExecSpace> d_errors =
        Kokkos::View<int, ExecSpace>("Errors");
    errors = d_errors;

    const int per_team0 =
        3 *
        Kokkos::View<double *, ExecSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged> >::shmem_size(128);
    const int per_thread0 =
        3 *
        Kokkos::View<double *, ExecSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged> >::shmem_size(16);

    const int per_team1 =
        3 * Kokkos::View<
                double *, ExecSpace,
                Kokkos::MemoryTraits<Kokkos::Unmanaged> >::shmem_size(12800);
    const int per_thread1 =
        3 * Kokkos::View<
                double *, ExecSpace,
                Kokkos::MemoryTraits<Kokkos::Unmanaged> >::shmem_size(1600);

#ifdef KOKKOS_ENABLE_SYCL
    int team_size = 4;
#else
    int team_size      = 8;
#endif
    if (team_size > ExecSpace::concurrency())
      team_size = ExecSpace::concurrency();
    {
      Kokkos::TeamPolicy<TagFor, ExecSpace, ScheduleType> policy(10, team_size,
                                                                 16);

      Kokkos::parallel_for(
          policy
              .set_scratch_size(0, Kokkos::PerTeam(per_team0),
                                Kokkos::PerThread(per_thread0))
              .set_scratch_size(1, Kokkos::PerTeam(per_team1),
                                Kokkos::PerThread(per_thread1)),
          *this);
      Kokkos::fence();

      typename Kokkos::View<int, ExecSpace>::HostMirror h_errors =
          Kokkos::create_mirror_view(d_errors);
      Kokkos::deep_copy(h_errors, d_errors);
      ASSERT_EQ(h_errors(), 0);
    }

    {
      int error = 0;
      Kokkos::TeamPolicy<TagReduce, ExecSpace, ScheduleType> policy(
          10, team_size, 16);

      Kokkos::parallel_reduce(
          policy
              .set_scratch_size(0, Kokkos::PerTeam(per_team0),
                                Kokkos::PerThread(per_thread0))
              .set_scratch_size(1, Kokkos::PerTeam(per_team1),
                                Kokkos::PerThread(per_thread1)),
          *this, error);

      ASSERT_EQ(error, 0);
    }
  };
};

template <class ExecSpace, class ScheduleType>
struct ClassWithShmemSizeFunction {
  using member_type =
      typename Kokkos::TeamPolicy<ExecSpace, ScheduleType>::member_type;

  Kokkos::View<int, ExecSpace, Kokkos::MemoryTraits<Kokkos::Atomic> > errors;

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagFor &, const member_type &team) const {
    int error = test_team_mulit_level_scratch_loop_body<ExecSpace>(team);
    errors() += error;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagReduce &, const member_type &team,
                  int &error) const {
    error += test_team_mulit_level_scratch_loop_body<ExecSpace>(team);
  }

  void run() {
    Kokkos::View<int, ExecSpace> d_errors =
        Kokkos::View<int, ExecSpace>("Errors");
    errors = d_errors;

    const int per_team1 =
        3 * Kokkos::View<
                double *, ExecSpace,
                Kokkos::MemoryTraits<Kokkos::Unmanaged> >::shmem_size(12800);
    const int per_thread1 =
        3 * Kokkos::View<
                double *, ExecSpace,
                Kokkos::MemoryTraits<Kokkos::Unmanaged> >::shmem_size(1600);

    int team_size = 8;
    if (team_size > ExecSpace::concurrency())
      team_size = ExecSpace::concurrency();

    {
      Kokkos::TeamPolicy<TagFor, ExecSpace, ScheduleType> policy(10, team_size,
                                                                 16);

      Kokkos::parallel_for(
          policy.set_scratch_size(1, Kokkos::PerTeam(per_team1),
                                  Kokkos::PerThread(per_thread1)),
          *this);
      Kokkos::fence();

      typename Kokkos::View<int, ExecSpace>::HostMirror h_errors =
          Kokkos::create_mirror_view(d_errors);
      Kokkos::deep_copy(h_errors, d_errors);
      ASSERT_EQ(h_errors(), 0);
    }

    {
      int error = 0;
      Kokkos::TeamPolicy<TagReduce, ExecSpace, ScheduleType> policy(
          10, team_size, 16);

      Kokkos::parallel_reduce(
          policy.set_scratch_size(1, Kokkos::PerTeam(per_team1),
                                  Kokkos::PerThread(per_thread1)),
          *this, error);

      ASSERT_EQ(error, 0);
    }
  };

  unsigned team_shmem_size(int team_size) const {
    const int per_team0 =
        3 *
        Kokkos::View<double *, ExecSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged> >::shmem_size(128);
    const int per_thread0 =
        3 *
        Kokkos::View<double *, ExecSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged> >::shmem_size(16);
    return per_team0 + team_size * per_thread0;
  }
};

template <class ExecSpace, class ScheduleType>
void test_team_mulit_level_scratch_test_lambda() {
#ifdef KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA
  Kokkos::View<int, ExecSpace, Kokkos::MemoryTraits<Kokkos::Atomic> > errors;
  Kokkos::View<int, ExecSpace> d_errors("Errors");
  errors = d_errors;

  const int per_team0 =
      3 *
      Kokkos::View<double *, ExecSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >::shmem_size(128);
  const int per_thread0 =
      3 *
      Kokkos::View<double *, ExecSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >::shmem_size(16);

  const int per_team1 =
      3 *
      Kokkos::View<double *, ExecSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >::shmem_size(12800);
  const int per_thread1 =
      3 *
      Kokkos::View<double *, ExecSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >::shmem_size(1600);

#ifdef KOKKOS_ENABLE_SYCL
  int team_size = 4;
#else
  int team_size = 8;
#endif
  if (team_size > ExecSpace::concurrency())
    team_size = ExecSpace::concurrency();

  Kokkos::TeamPolicy<ExecSpace, ScheduleType> policy(10, team_size, 16);

  Kokkos::parallel_for(
      policy
          .set_scratch_size(0, Kokkos::PerTeam(per_team0),
                            Kokkos::PerThread(per_thread0))
          .set_scratch_size(1, Kokkos::PerTeam(per_team1),
                            Kokkos::PerThread(per_thread1)),
      KOKKOS_LAMBDA(
          const typename Kokkos::TeamPolicy<ExecSpace>::member_type &team) {
        int error = test_team_mulit_level_scratch_loop_body<ExecSpace>(team);
        errors() += error;
      });
  Kokkos::fence();

  typename Kokkos::View<int, ExecSpace>::HostMirror h_errors =
      Kokkos::create_mirror_view(errors);
  Kokkos::deep_copy(h_errors, d_errors);
  ASSERT_EQ(h_errors(), 0);

  int error = 0;
  Kokkos::parallel_reduce(
      policy
          .set_scratch_size(0, Kokkos::PerTeam(per_team0),
                            Kokkos::PerThread(per_thread0))
          .set_scratch_size(1, Kokkos::PerTeam(per_team1),
                            Kokkos::PerThread(per_thread1)),
      KOKKOS_LAMBDA(
          const typename Kokkos::TeamPolicy<ExecSpace>::member_type &team,
          int &count) {
        count += test_team_mulit_level_scratch_loop_body<ExecSpace>(team);
      },
      error);
  ASSERT_EQ(error, 0);
#endif
}

}  // namespace Test

namespace {

template <class ExecSpace, class ScheduleType>
struct TestMultiLevelScratchTeam {
  TestMultiLevelScratchTeam() { run(); }

  void run() {
#ifdef KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA
    Test::test_team_mulit_level_scratch_test_lambda<ExecSpace, ScheduleType>();
#endif
    Test::ClassNoShmemSizeFunction<ExecSpace, ScheduleType> c1;
    c1.run();

    Test::ClassWithShmemSizeFunction<ExecSpace, ScheduleType> c2;
    c2.run();
  }
};

}  // namespace

namespace Test {

template <class ExecSpace>
struct TestShmemSize {
  TestShmemSize() { run(); }

  void run() {
    using view_type = Kokkos::View<int64_t ***, ExecSpace>;

    size_t d1 = 5;
    size_t d2 = 6;
    size_t d3 = 7;

    size_t size = view_type::shmem_size(d1, d2, d3);

    ASSERT_EQ(size, (d1 * d2 * d3 + 1) * sizeof(int64_t));

    test_layout_stride();
  }

  void test_layout_stride() {
    int rank       = 3;
    int order[3]   = {2, 0, 1};
    int extents[3] = {100, 10, 3};
    auto s1 =
        Kokkos::View<double ***, Kokkos::LayoutStride, ExecSpace>::shmem_size(
            Kokkos::LayoutStride::order_dimensions(rank, order, extents));
    auto s2 =
        Kokkos::View<double ***, Kokkos::LayoutRight, ExecSpace>::shmem_size(
            extents[0], extents[1], extents[2]);
    ASSERT_EQ(s1, s2);
  }
};

}  // namespace Test

/*--------------------------------------------------------------------------*/

namespace Test {

namespace {

template <class ExecSpace, class ScheduleType, class T, class Enabled = void>
struct TestTeamBroadcast;

template <class ExecSpace, class ScheduleType, class T>
struct TestTeamBroadcast<
    ExecSpace, ScheduleType, T,
    typename std::enable_if<(sizeof(T) == sizeof(char)), void>::type> {
  using team_member =
      typename Kokkos::TeamPolicy<ScheduleType, ExecSpace>::member_type;
  using memory_space = typename ExecSpace::memory_space;
  using value_type   = T;

  const value_type offset;

  TestTeamBroadcast(const size_t /*league_size*/, const value_type os_)
      : offset(os_) {}

  struct BroadcastTag {};

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member &teamMember, value_type &update) const {
    int lid = teamMember.league_rank();
    int tid = teamMember.team_rank();
    int ts  = teamMember.team_size();

    value_type parUpdate = 0;
    value_type value     = (value_type)(tid % 0xFF) + offset;

    // broadcast boolean and value to team from source thread
    teamMember.team_broadcast(value, lid % ts);

    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(teamMember, ts),
        [&](const int /*j*/, value_type &teamUpdate) { teamUpdate |= value; },
        Kokkos::BOr<value_type, memory_space>(parUpdate));

    if (teamMember.team_rank() == 0) update |= parUpdate;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const BroadcastTag &, const team_member &teamMember,
                  value_type &update) const {
    int lid = teamMember.league_rank();
    int tid = teamMember.team_rank();
    int ts  = teamMember.team_size();

    value_type parUpdate = 0;
    value_type value     = (value_type)(tid % 0xFF) + offset;

    teamMember.team_broadcast([&](value_type &var) { var -= offset; }, value,
                              lid % ts);

    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(teamMember, ts),
        [&](const int /*j*/, value_type &teamUpdate) { teamUpdate |= value; },
        Kokkos::BOr<value_type, memory_space>(parUpdate));

    if (teamMember.team_rank() == 0) update |= parUpdate;
  }

  static void test_teambroadcast(const size_t league_size,
                                 const value_type off) {
    TestTeamBroadcast functor(league_size, off);

    using policy_type = Kokkos::TeamPolicy<ScheduleType, ExecSpace>;
    using policy_type_f =
        Kokkos::TeamPolicy<ScheduleType, ExecSpace, BroadcastTag>;

    // FIXME_OPENMPTARGET temporary restriction for team size to be at least 32
#ifdef KOKKOS_ENABLE_OPENMPTARGET
    int fake_team_size =
        std::is_same<ExecSpace, Kokkos::Experimental::OpenMPTarget>::value ? 32
                                                                           : 1;
#else
    int fake_team_size = 1;
#endif
    const int team_size =
        policy_type_f(league_size, fake_team_size)
            .team_size_max(
                functor,
                Kokkos::
                    ParallelReduceTag());  // printf("team_size=%d\n",team_size);

    // team_broadcast with value
    value_type total = 0;

    Kokkos::parallel_reduce(policy_type(league_size, team_size), functor,
                            Kokkos::BOr<value_type, Kokkos::HostSpace>(total));

    value_type expected_result = 0;
    for (unsigned int i = 0; i < league_size; i++) {
      value_type val = (value_type((i % team_size % 0xFF)) + off);
      expected_result |= val;
    }
    ASSERT_EQ(expected_result, total);
    // printf("team_broadcast with value --"
    //"expected_result=%x,"
    //"total=%x\n",expected_result, total);

    // team_broadcast with function object
    total = 0;

    Kokkos::parallel_reduce(policy_type_f(league_size, team_size), functor,
                            Kokkos::BOr<value_type, Kokkos::HostSpace>(total));

    expected_result = 0;
    for (unsigned int i = 0; i < league_size; i++) {
      value_type val = ((value_type)((i % team_size % 0xFF)));
      expected_result |= val;
    }
    ASSERT_EQ(expected_result, total);
    // printf("team_broadcast with function object --"
    // "expected_result=%x,"
    // "total=%x\n",expected_result, total);
  }
};

template <class ExecSpace, class ScheduleType, class T>
struct TestTeamBroadcast<
    ExecSpace, ScheduleType, T,
    typename std::enable_if<(sizeof(T) > sizeof(char)), void>::type> {
  using team_member =
      typename Kokkos::TeamPolicy<ScheduleType, ExecSpace>::member_type;
  using value_type = T;

  const value_type offset;

  TestTeamBroadcast(const size_t /*league_size*/, const value_type os_)
      : offset(os_) {}

  struct BroadcastTag {};

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member &teamMember, value_type &update) const {
    int lid = teamMember.league_rank();
    int tid = teamMember.team_rank();
    int ts  = teamMember.team_size();

    value_type parUpdate = 0;
    value_type value     = (value_type)(tid * 3) + offset;

    // setValue is used to determine if the update should be
    // performed at the bottom.  The thread id must match the
    // thread id used to broadcast the value.  It is the
    // thread id that matches the league rank mod team size
    // this way each league rank will use a different thread id
    // which is likely not 0
    bool setValue = ((lid % ts) == tid);

    // broadcast boolean and value to team from source thread
    teamMember.team_broadcast(value, lid % ts);
    teamMember.team_broadcast(setValue, lid % ts);

    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(teamMember, ts),
        [&](const int /*j*/, value_type &teamUpdate) { teamUpdate += value; },
        parUpdate);

    if (teamMember.team_rank() == 0 && setValue) update += parUpdate;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const BroadcastTag &, const team_member &teamMember,
                  value_type &update) const {
    int lid = teamMember.league_rank();
    int tid = teamMember.team_rank();
    int ts  = teamMember.team_size();

    value_type parUpdate = 0;
    value_type value     = (value_type)(tid * 3) + offset;

    // setValue is used to determine if the update should be
    // performed at the bottom.  The thread id must match the
    // thread id used to broadcast the value.  It is the
    // thread id that matches the league rank mod team size
    // this way each league rank will use a different thread id
    // which is likely not 0. Note the logic is switched from
    // above because the functor switches it back.
    bool setValue = ((lid % ts) != tid);

    teamMember.team_broadcast([&](value_type &var) { var *= 2; }, value,
                              lid % ts);
    teamMember.team_broadcast([&](bool &bVar) { bVar = !bVar; }, setValue,
                              lid % ts);

    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(teamMember, ts),
        [&](const int /*j*/, value_type &teamUpdate) { teamUpdate += value; },
        parUpdate);

    if (teamMember.team_rank() == 0 && setValue) update += parUpdate;
  }

  template <class ScalarType>
  static inline
      typename std::enable_if<!std::is_integral<ScalarType>::value, void>::type
      compare_test(ScalarType A, ScalarType B, double epsilon_factor) {
    if (std::is_same<ScalarType, double>::value ||
        std::is_same<ScalarType, float>::value) {
      ASSERT_NEAR((double)A, (double)B,
                  epsilon_factor * std::abs(A) *
                      std::numeric_limits<ScalarType>::epsilon());
    } else {
      ASSERT_EQ(A, B);
    }
  }

  template <class ScalarType>
  static inline
      typename std::enable_if<std::is_integral<ScalarType>::value, void>::type
      compare_test(ScalarType A, ScalarType B, double) {
    ASSERT_EQ(A, B);
  }

  static void test_teambroadcast(const size_t league_size,
                                 const value_type off) {
    TestTeamBroadcast functor(league_size, off);

    using policy_type = Kokkos::TeamPolicy<ScheduleType, ExecSpace>;
    using policy_type_f =
        Kokkos::TeamPolicy<ScheduleType, ExecSpace, BroadcastTag>;

    // FIXME_OPENMPTARGET temporary restriction for team size to be at least 32
#ifdef KOKKOS_ENABLE_OPENMPTARGET
    int fake_team_size =
        std::is_same<ExecSpace, Kokkos::Experimental::OpenMPTarget>::value ? 32
                                                                           : 1;
#else
    int fake_team_size = 1;
#endif
    const int team_size =
        policy_type_f(league_size, fake_team_size)
            .team_size_max(
                functor,
                Kokkos::
                    ParallelReduceTag());  // printf("team_size=%d\n",team_size);
    // team_broadcast with value
    value_type total = 0;

    Kokkos::parallel_reduce(policy_type(league_size, team_size), functor,
                            total);

    value_type expected_result = 0;
    for (unsigned int i = 0; i < league_size; i++) {
      value_type val =
          (value_type((i % team_size) * 3) + off) * (value_type)team_size;
      expected_result += val;
    }
    // For comparison purposes treat the reduction as a random walk in the
    // least significant digit, which gives a typical walk distance
    // sqrt(league_size) Add 4x for larger sigma
    compare_test(expected_result, total, 4.0 * std::sqrt(league_size));

    // team_broadcast with function object
    total = 0;

    Kokkos::parallel_reduce(policy_type_f(league_size, team_size), functor,
                            total);

    expected_result = 0;
    for (unsigned int i = 0; i < league_size; i++) {
      value_type val = ((value_type)((i % team_size) * 3) + off) *
                       (value_type)(2 * team_size);
      expected_result += val;
    }
    // For comparison purposes treat the reduction as a random walk in the
    // least significant digit, which gives a typical walk distance
    // sqrt(league_size) Add 4x for larger sigma
    compare_test(expected_result, total, 4.0 * std::sqrt(league_size));
  }
};

template <class ExecSpace>
struct TestScratchAlignment {
  struct TestScalar {
    double x, y, z;
  };
  TestScratchAlignment() {
    test(true);
    test(false);
  }
  using ScratchView =
      Kokkos::View<TestScalar *, typename ExecSpace::scratch_memory_space>;
  using ScratchViewInt =
      Kokkos::View<int *, typename ExecSpace::scratch_memory_space>;
  void test(bool allocate_small) {
    int shmem_size = ScratchView::shmem_size(11);
#ifdef KOKKOS_ENABLE_OPENMPTARGET
    int team_size = 32;
#else
    int team_size      = 1;
#endif
    if (allocate_small) shmem_size += ScratchViewInt::shmem_size(1);
    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(1, team_size)
            .set_scratch_size(0, Kokkos::PerTeam(shmem_size)),
        KOKKOS_LAMBDA(
            const typename Kokkos::TeamPolicy<ExecSpace>::member_type &team) {
          if (allocate_small) ScratchViewInt p(team.team_scratch(0), 1);
          ScratchView a(team.team_scratch(0), 11);
          if (ptrdiff_t(a.data()) % sizeof(TestScalar) != 0)
            Kokkos::abort("Error: invalid scratch view alignment\n");
        });
    Kokkos::fence();
  }
};

}  // namespace

namespace {

template <class ExecSpace>
struct TestTeamPolicyHandleByValue {
  using scalar     = double;
  using exec_space = ExecSpace;
  using mem_space  = typename ExecSpace::memory_space;

  TestTeamPolicyHandleByValue() { test(); }

  void test() {
#if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
    const int M = 1, N = 1;
    Kokkos::View<scalar **, mem_space> a("a", M, N);
    Kokkos::View<scalar **, mem_space> b("b", M, N);
    Kokkos::deep_copy(a, 0.0);
    Kokkos::deep_copy(b, 1.0);
    Kokkos::parallel_for(
        "test_tphandle_by_value",
        Kokkos::TeamPolicy<exec_space>(M, Kokkos::AUTO(), 1),
        KOKKOS_LAMBDA(
            const typename Kokkos::TeamPolicy<exec_space>::member_type team) {
          const int i = team.league_rank();
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, N),
                               [&](const int j) { a(i, j) += b(i, j); });
        });
#endif
  }
};

}  // namespace

}  // namespace Test
#endif

/*--------------------------------------------------------------------------*/

namespace Test {
TEST(TEST_CATEGORY, MDParallelFor) {
  using namespace MDParallelism;
  // int dims[] = {15, 16, 16, 16, 16, 16, 16, 16};
  int dims[] = {4, 4, 4, 4, 4, 4, 4, 4};

  int teamSize = 4;
  int N0       = dims[0];
  int N1       = dims[1];
  int N2       = dims[2];
  int N3       = dims[3];
  int N4       = dims[4];
  int N5       = dims[5];
  int N6       = dims[6];
  int N7       = dims[7];

  // {
  //   TestMDParallelFor<TEST_EXECSPACE>::test_for3_MDTeamThreadRange(N0, N1, N2);
  //   TestMDParallelFor<TEST_EXECSPACE>::test_for8_MDTeamThreadRange(N0, N1, N2, N3, N4, N5, N6, N7);

  //   TestMDParallelFor<TEST_EXECSPACE>::test_for2_MDThreadVectorRange(N0, N1);
  //   TestMDParallelFor<TEST_EXECSPACE>::test_for2_MDTeamVectorRange(teamSize, N0,
  //                                                                  N1);
  // }

  // {
  //   TestMDParallelFor<TEST_EXECSPACE>::test_for3_MDTeamThreadRange(N0, N1, N2);
  //   TestMDParallelFor<TEST_EXECSPACE>::
  //       test_for3_MDTeamThreadRange_with_direction<Kokkos::Iterate::Left>(
  //           N0, N1, N2);
  //   TestMDParallelFor<TEST_EXECSPACE>::
  //       test_for3_MDTeamThreadRange_with_direction<Kokkos::Iterate::Right>(
  //           N0, N1, N2);

  //   TestMDParallelFor<TEST_EXECSPACE>::test_for8_MDTeamThreadRange(
  //       N0, N1, N2, N3, N4, N5, N6, N7);

  //   TestMDParallelFor<TEST_EXECSPACE>::
  //       test_for8_MDTeamThreadRange_with_direction<Kokkos::Iterate::Left>(
  //           N0, N1, N2, N3, N4, N5, N6, N7);
  //   TestMDParallelFor<TEST_EXECSPACE>::
  //       test_for8_MDTeamThreadRange_with_direction<Kokkos::Iterate::Right>(
  //           N0, N1, N2, N3, N4, N5, N6, N7);
  // }

  {
    auto md_thread_vector_range = [](auto const& team, auto... ns) {
      const int nsArray[sizeof...(ns)] = {ns...};
      return Kokkos::MDTeamThreadRange(team, nsArray);
    };

    auto md_thread_vector_range_left = [](auto const& team, auto... ns) {
      const int nsArray[sizeof...(ns)] = {ns...};
      return Kokkos::MDTeamThreadRange<Kokkos::Iterate::Left>(team, nsArray);
    };

    TestMDParallelFor<TEST_EXECSPACE>::test_for_MDTeamThreadRange(
        md_thread_vector_range, N0, N1, N2);
    // TestMDParallelFor<TEST_EXECSPACE>::test_for_MDTeamThreadRange(
    //     md_thread_vector_range, N0, N1, N2, N3);
    // TestMDParallelFor<TEST_EXECSPACE>::test_for_MDTeamThreadRange(
    //     md_thread_vector_range_left, N0, N1, N2);
    // TestMDParallelFor<TEST_EXECSPACE>::test_for_MDTeamThreadRange(
    //     md_thread_vector_range_left, N0, N1, N2, N3);
    // TestMDParallelFor<TEST_EXECSPACE>::test_for_MDTeamThreadRange(
    //     md_thread_vector_range_left, N0, N1, N2, N3, N4, N5, N6, N7);
  }

#if 0
  {
    TestMDParallelFor<TEST_EXECSPACE>::test_for2_MDThreadVectorRange(N0, N1);

    TestMDParallelFor<TEST_EXECSPACE>::
        test_for2_MDThreadVectorRange_with_direction<Kokkos::Iterate::Left,
                                                     Kokkos::Iterate::Left>(N0,
                                                                            N1);
    TestMDParallelFor<TEST_EXECSPACE>::
        test_for2_MDThreadVectorRange_with_direction<Kokkos::Iterate::Left,
                                                     Kokkos::Iterate::Right>(
            N0, N1);
    TestMDParallelFor<TEST_EXECSPACE>::
        test_for2_MDThreadVectorRange_with_direction<Kokkos::Iterate::Right,
                                                     Kokkos::Iterate::Left>(N0,
                                                                            N1);
    TestMDParallelFor<TEST_EXECSPACE>::
        test_for2_MDThreadVectorRange_with_direction<Kokkos::Iterate::Right,
                                                     Kokkos::Iterate::Right>(
            N0, N1);

    TestMDParallelFor<TEST_EXECSPACE>::test_for8_MDThreadVectorRange(
        N0, N1, N2, N3, N4, N5, N6, N7);

    TestMDParallelFor<TEST_EXECSPACE>::
        test_for8_MDThreadVectorRange_with_direction<Kokkos::Iterate::Left,
                                                     Kokkos::Iterate::Left>(
            N0, N1, N2, N3, N4, N5, N6, N7);
    TestMDParallelFor<TEST_EXECSPACE>::
        test_for8_MDThreadVectorRange_with_direction<Kokkos::Iterate::Left,
                                                     Kokkos::Iterate::Right>(
            N0, N1, N2, N3, N4, N5, N6, N7);
    TestMDParallelFor<TEST_EXECSPACE>::
        test_for8_MDThreadVectorRange_with_direction<Kokkos::Iterate::Right,
                                                     Kokkos::Iterate::Left>(
            N0, N1, N2, N3, N4, N5, N6, N7);
    TestMDParallelFor<TEST_EXECSPACE>::
        test_for8_MDThreadVectorRange_with_direction<Kokkos::Iterate::Right,
                                                     Kokkos::Iterate::Right>(
            N0, N1, N2, N3, N4, N5, N6, N7);
  }

  {
    TestMDParallelFor<TEST_EXECSPACE>::test_for2_MDTeamVectorRange(teamSize, N0,
                                                                   N1);

    TestMDParallelFor<TEST_EXECSPACE>::
        test_for2_MDTeamVectorRange_with_direction<Kokkos::Iterate::Left,
                                                   Kokkos::Iterate::Left>(
            teamSize, N0, N1);
  }
#endif // if 0

}

TEST(TEST_CATEGORY, MDParallelReduce) {
  using namespace MDParallelism;
  // int dims[] = {15, 16, 16, 16, 16, 16, 16, 16};
  int dims[] = {4, 4, 4, 4, 4, 4, 4, 4};

  int teamSize = 4;
  int N0       = dims[0];
  int N1       = dims[1];
  int N2       = dims[2];
  int N3       = dims[3];
  int N4       = dims[4];
  int N5       = dims[5];
  int N6       = dims[6];
  int N7       = dims[7];

{
    TestMDParallelReduce<TEST_EXECSPACE>::test_reduce2_MDTeamThreadRange(N0,
                                                                         N1);
    TestMDParallelReduce<TEST_EXECSPACE>::test_reduce2_MDThreadVectorRange(N0,
                                                                           N1);
    TestMDParallelReduce<TEST_EXECSPACE>::test_reduce2_MDTeamVectorRange(N0,
                                                                         N1);
}


#if 0
  {
    TestMDParallelReduce<TEST_EXECSPACE>::test_reduce2_MDTeamThreadRange(N0,
                                                                         N1);

    TestMDParallelReduce<TEST_EXECSPACE>::
        test_reduce2_MDTeamThreadRange_with_direction<Kokkos::Iterate::Left>(
            N0, N1);
  }

  {
    TestMDParallelReduce<TEST_EXECSPACE>::test_reduce2_MDThreadVectorRange(N0,
                                                                           N1);

    TestMDParallelReduce<TEST_EXECSPACE>::
        test_reduce2_MDThreadVectorRange_with_direction<Kokkos::Iterate::Left,
                                                        Kokkos::Iterate::Left>(
            N0, N1);
  }

  {
    TestMDParallelReduce<TEST_EXECSPACE>::test_reduce3_MDTeamVectorRange(
        teamSize, N0, N1);

    TestMDParallelReduce<TEST_EXECSPACE>::
        test_reduce3_MDTeamVectorRange_with_direction<Kokkos::Iterate::Left,
                                                      Kokkos::Iterate::Left>(
            teamSize, N0, N1);
  }

#endif
}

}  // namespace Test
