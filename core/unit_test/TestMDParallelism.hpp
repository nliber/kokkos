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

struct FillFlattenedIndex {
  explicit FillFlattenedIndex(int n0, int n1, int n2, int n3 = 0, int n4 = 0,
                              int n5 = 0, int n6 = 0, int n7 = 0)
      : initValue{n0, n1, n2, n3, n4, n5, n6, n7} {}

  int operator()(int n0, int n1, int n2, int n3 = 0, int n4 = 0, int n5 = 0,
                 int n6 = 0, int n7 = 0) const {
    return ((((((n7 * initValue[7] + n6) * initValue[6] + n5) * initValue[5] +
               n4) *
                  initValue[4] +
              n3) *
                 initValue[3] +
             n2) *
                initValue[2] +
            n1) *
               initValue[1] +
           n0;
  }

  int initValue[8];
};

struct FillConstant {
  explicit FillConstant(int initValue_) : initValue(initValue_) {}

  int operator()(int n0, int n1, int n2) const { return initValue; }

  int initValue;
};

template <typename ExecSpace>
struct TestMDParallelFor {
  // The difference between test_for2 and test_for2_with_direction is
  // the call to MDThreadVectorRange.  test_for2 deduces all the
  // parameters for the return type, whily test_for2_with_direction
  // specifies the outer and inner directions (different code path).

  using DataType = int;
  using TeamType = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  template <typename HostViewType, typename FillFunctor>
  static void check_result_3D(HostViewType h_view,
                              FillFunctor const& fillFunctor) {
    for (int i = 0; i < h_view.extent(0); ++i) {
      for (int j = 0; j < h_view.extent(1); ++j) {
        for (int k = 0; k < h_view.extent(2); ++k) {
          EXPECT_EQ(h_view(i, j, k), fillFunctor(i, j, k));
        }
      }
    }
  }

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_for_3D_MDTeamThreadRange(int* dims,
                                                     const int initValue) {
    using ViewType     = typename Kokkos::View<DataType***, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int numTeams = dims[0];
    int n0       = dims[1];
    int n1       = dims[2];

    ViewType v("v", numTeams, n0, n1);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(numTeams, n0, n1);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(numTeams, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange = Kokkos::MDTeamThreadRange<Direction>(team, n0, n1);

          Kokkos::parallel_for(teamRange, [=](int i, int j) {
            v(leagueRank, i, j) += fillFlattenedIndex(leagueRank, i, j);
          });
        });

    HostViewType h_view =
        Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), v);

    check_result_3D(h_view, fillFlattenedIndex);
  }

  template <typename HostViewType, typename FillFunctor>
  static void check_result_4D(HostViewType h_view, FillFunctor& fillFunctor) {
    for (int i = 0; i < h_view.extent(0); ++i) {
      for (int j = 0; j < h_view.extent(1); ++j) {
        for (int k = 0; k < h_view.extent(2); ++k) {
          for (int l = 0; l < h_view.extent(3); ++l) {
            EXPECT_EQ(h_view(i, j, k, l), fillFunctor(i, j, k, l));
          }
        }
      }
    }
  }

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_for_4D_MDTeamThreadRange(int* dims,
                                                     const int initValue) {
    using ViewType     = typename Kokkos::View<DataType****, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int numTeams = dims[0];
    int n0       = dims[1];
    int n1       = dims[2];
    int n2       = dims[3];

    ViewType v("v", numTeams, n0, n1, n2);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(numTeams, n0, n1, n2);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(numTeams, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange =
              Kokkos::MDTeamThreadRange<Direction>(team, n0, n1, n2);

          Kokkos::parallel_for(teamRange, [=](int i, int j, int k) {
            v(leagueRank, i, j, k) += fillFlattenedIndex(leagueRank, i, j, k);
          });
        });

    HostViewType h_view =
        Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), v);

    check_result_4D(h_view, fillFlattenedIndex);
  }

  template <typename HostViewType, typename FillFunctor>
  static void check_result_5D(HostViewType h_view, FillFunctor& fillFunctor) {
    for (int i = 0; i < h_view.extent(0); ++i) {
      for (int j = 0; j < h_view.extent(1); ++j) {
        for (int k = 0; k < h_view.extent(2); ++k) {
          for (int l = 0; l < h_view.extent(3); ++l) {
            for (int m = 0; m < h_view.extent(4); ++m) {
              EXPECT_EQ(h_view(i, j, k, l, m), fillFunctor(i, j, k, l, m));
            }
          }
        }
      }
    }
  }

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_for_5D_MDTeamThreadRange(int* dims,
                                                     const int initValue) {
    using ViewType     = typename Kokkos::View<DataType*****, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int numTeams = dims[0];
    int n0       = dims[1];
    int n1       = dims[2];
    int n2       = dims[3];
    int n3       = dims[4];

    ViewType v("v", numTeams, n0, n1, n2, n3);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(numTeams, n0, n1, n2, n3);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(numTeams, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange =
              Kokkos::MDTeamThreadRange<Direction>(team, n0, n1, n2, n3);

          Kokkos::parallel_for(teamRange, [=](int i, int j, int k, int l) {
            v(leagueRank, i, j, k, l) +=
                fillFlattenedIndex(leagueRank, i, j, k, l);
          });
        });

    HostViewType h_view =
        Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), v);

    check_result_5D(h_view, fillFlattenedIndex);
  }

  template <typename HostViewType, typename FillFunctor>
  static void check_result_6D(HostViewType h_view, FillFunctor& fillFunctor) {
    for (int i = 0; i < h_view.extent(0); ++i) {
      for (int j = 0; j < h_view.extent(1); ++j) {
        for (int k = 0; k < h_view.extent(2); ++k) {
          for (int l = 0; l < h_view.extent(3); ++l) {
            for (int m = 0; m < h_view.extent(4); ++m) {
              for (int n = 0; n < h_view.extent(5); ++n) {
                EXPECT_EQ(h_view(i, j, k, l, m, n),
                          fillFunctor(i, j, k, l, m, n));
              }
            }
          }
        }
      }
    }
  }

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_for_6D_MDTeamThreadRange(int* dims,
                                                     const int initValue) {
    using ViewType     = typename Kokkos::View<DataType******, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int numTeams = dims[0];
    int n0       = dims[1];
    int n1       = dims[2];
    int n2       = dims[3];
    int n3       = dims[4];
    int n4       = dims[5];

    ViewType v("v", numTeams, n0, n1, n2, n3, n4);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(numTeams, n0, n1, n2, n3, n4);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(numTeams, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange =
              Kokkos::MDTeamThreadRange<Direction>(team, n0, n1, n2, n3, n4);

          Kokkos::parallel_for(
              teamRange, [=](int i, int j, int k, int l, int m) {
                v(leagueRank, i, j, k, l, m) +=
                    fillFlattenedIndex(leagueRank, i, j, k, l, m);
              });
        });

    HostViewType h_view =
        Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), v);

    check_result_6D(h_view, fillFlattenedIndex);
  }

  template <typename HostViewType, typename FillFunctor>
  static void check_result_7D(HostViewType h_view, FillFunctor& fillFunctor) {
    for (int i = 0; i < h_view.extent(0); ++i) {
      for (int j = 0; j < h_view.extent(1); ++j) {
        for (int k = 0; k < h_view.extent(2); ++k) {
          for (int l = 0; l < h_view.extent(3); ++l) {
            for (int m = 0; m < h_view.extent(4); ++m) {
              for (int n = 0; n < h_view.extent(5); ++n) {
                for (int o = 0; o < h_view.extent(6); ++o) {
                  EXPECT_EQ(h_view(i, j, k, l, m, n, o),
                            fillFunctor(i, j, k, l, m, n, o));
                }
              }
            }
          }
        }
      }
    }
  }

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_for_7D_MDTeamThreadRange(int* dims,
                                                     const int initValue) {
    using ViewType     = typename Kokkos::View<DataType*******, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int numTeams = dims[0];
    int n0       = dims[1];
    int n1       = dims[2];
    int n2       = dims[3];
    int n3       = dims[4];
    int n4       = dims[5];
    int n5       = dims[6];

    ViewType v("v", numTeams, n0, n1, n2, n3, n4, n5);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(numTeams, n0, n1, n2, n3, n4, n5);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(numTeams, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange = Kokkos::MDTeamThreadRange<Direction>(team, n0, n1,
                                                                n2, n3, n4, n5);

          Kokkos::parallel_for(
              teamRange, [=](int i, int j, int k, int l, int m, int n) {
                v(leagueRank, i, j, k, l, m, n) +=
                    fillFlattenedIndex(leagueRank, i, j, k, l, m, n);
              });
        });

    HostViewType h_view =
        Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), v);

    check_result_7D(h_view, fillFlattenedIndex);
  }

  template <typename HostViewType, typename FillFunctor>
  static void check_result_8D(HostViewType h_view, FillFunctor& fillFunctor) {
    for (int i = 0; i < h_view.extent(0); ++i) {
      for (int j = 0; j < h_view.extent(1); ++j) {
        for (int k = 0; k < h_view.extent(2); ++k) {
          for (int l = 0; l < h_view.extent(3); ++l) {
            for (int m = 0; m < h_view.extent(4); ++m) {
              for (int n = 0; n < h_view.extent(5); ++n) {
                for (int o = 0; o < h_view.extent(6); ++o) {
                  for (int p = 0; p < h_view.extent(7); ++p) {
                    EXPECT_EQ(h_view(i, j, k, l, m, n, o, p),
                              fillFunctor(i, j, k, l, m, n, o, p));
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  template <Kokkos::Iterate Direction = Kokkos::Iterate::Default>
  static void test_parallel_for_8D_MDTeamThreadRange(int* dims,
                                                     const int initValue) {
    using ViewType     = typename Kokkos::View<DataType********, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int numTeams = dims[0];
    int n0       = dims[1];
    int n1       = dims[2];
    int n2       = dims[3];
    int n3       = dims[4];
    int n4       = dims[5];
    int n5       = dims[6];
    int n6       = dims[7];

    ViewType v("v", numTeams, n0, n1, n2, n3, n4, n5, n6);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(numTeams, n0, n1, n2, n3, n4, n5, n6);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(numTeams, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange = Kokkos::MDTeamThreadRange<Direction>(
              team, n0, n1, n2, n3, n4, n5, n6);

          Kokkos::parallel_for(
              teamRange, [=](int i, int j, int k, int l, int m, int n, int o) {
                v(leagueRank, i, j, k, l, m, n, o) +=
                    fillFlattenedIndex(leagueRank, i, j, k, l, m, n, o);
              });
        });

    HostViewType h_view =
        Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), v);

    check_result_8D(h_view, fillFlattenedIndex);
  }

  template <Kokkos::Iterate OuterDirection = Kokkos::Iterate::Default,
            Kokkos::Iterate InnerDirection = Kokkos::Iterate::Default>
  static void test_parallel_for_3D_MDThreadVectorRange(int* dims,
                                                       const int initValue) {
    using ViewType     = typename Kokkos::View<DataType***, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int numTeams = dims[0];
    int n0       = dims[1];
    int n1       = dims[2];

    ViewType v("v", numTeams, n0, n1);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(numTeams, n0, n1);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(numTeams, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange =
              Kokkos::MDThreadVectorRange<OuterDirection, InnerDirection>(
                  team, n0, n1);

          Kokkos::parallel_for(teamRange, [=](int i, int j) {
            v(leagueRank, i, j) += fillFlattenedIndex(leagueRank, i, j);
          });
        });

    HostViewType h_view =
        Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), v);

    check_result_3D(h_view, fillFlattenedIndex);
  }

  template <Kokkos::Iterate OuterDirection = Kokkos::Iterate::Default,
            Kokkos::Iterate InnerDirection = Kokkos::Iterate::Default>
  static void test_parallel_for_4D_MDThreadVectorRange(int* dims,
                                                       const int initValue) {
    using ViewType     = typename Kokkos::View<DataType****, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int numTeams = dims[0];
    int n0       = dims[1];
    int n1       = dims[2];
    int n2       = dims[3];

    ViewType v("v", numTeams, n0, n1, n2);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(numTeams, n0, n1, n2);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(numTeams, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange =
              Kokkos::MDThreadVectorRange<OuterDirection, InnerDirection>(
                  team, n0, n1, n2);

          Kokkos::parallel_for(teamRange, [=](int i, int j, int k) {
            v(leagueRank, i, j, k) += fillFlattenedIndex(leagueRank, i, j, k);
          });
        });

    HostViewType h_view =
        Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), v);

    check_result_4D(h_view, fillFlattenedIndex);
  }

  template <Kokkos::Iterate OuterDirection = Kokkos::Iterate::Default,
            Kokkos::Iterate InnerDirection = Kokkos::Iterate::Default>
  static void test_parallel_for_5D_MDThreadVectorRange(int* dims,
                                                       const int initValue) {
    using ViewType     = typename Kokkos::View<DataType*****, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int numTeams = dims[0];
    int n0       = dims[1];
    int n1       = dims[2];
    int n2       = dims[3];
    int n3       = dims[4];

    ViewType v("v", numTeams, n0, n1, n2, n3);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(numTeams, n0, n1, n2, n3);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(numTeams, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange =
              Kokkos::MDThreadVectorRange<OuterDirection, InnerDirection>(
                  team, n0, n1, n2, n3);

          Kokkos::parallel_for(teamRange, [=](int i, int j, int k, int l) {
            v(leagueRank, i, j, k, l) +=
                fillFlattenedIndex(leagueRank, i, j, k, l);
          });
        });

    HostViewType h_view =
        Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), v);

    check_result_5D(h_view, fillFlattenedIndex);
  }

  template <Kokkos::Iterate OuterDirection = Kokkos::Iterate::Default,
            Kokkos::Iterate InnerDirection = Kokkos::Iterate::Default>
  static void test_parallel_for_6D_MDThreadVectorRange(int* dims,
                                                       const int initValue) {
    using ViewType     = typename Kokkos::View<DataType******, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int numTeams = dims[0];
    int n0       = dims[1];
    int n1       = dims[2];
    int n2       = dims[3];
    int n3       = dims[4];
    int n4       = dims[5];

    ViewType v("v", numTeams, n0, n1, n2, n3, n4);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(numTeams, n0, n1, n2, n3, n4);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(numTeams, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange =
              Kokkos::MDThreadVectorRange<OuterDirection, InnerDirection>(
                  team, n0, n1, n2, n3, n4);

          Kokkos::parallel_for(
              teamRange, [=](int i, int j, int k, int l, int m) {
                v(leagueRank, i, j, k, l, m) +=
                    fillFlattenedIndex(leagueRank, i, j, k, l, m);
              });
        });

    HostViewType h_view =
        Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), v);

    check_result_6D(h_view, fillFlattenedIndex);
  }

  template <Kokkos::Iterate OuterDirection = Kokkos::Iterate::Default,
            Kokkos::Iterate InnerDirection = Kokkos::Iterate::Default>
  static void test_parallel_for_7D_MDThreadVectorRange(int* dims,
                                                       const int initValue) {
    using ViewType     = typename Kokkos::View<DataType*******, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int numTeams = dims[0];
    int n0       = dims[1];
    int n1       = dims[2];
    int n2       = dims[3];
    int n3       = dims[4];
    int n4       = dims[5];
    int n5       = dims[6];

    ViewType v("v", numTeams, n0, n1, n2, n3, n4, n5);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(numTeams, n0, n1, n2, n3, n4, n5);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(numTeams, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange =
              Kokkos::MDThreadVectorRange<OuterDirection, InnerDirection>(
                  team, n0, n1, n2, n3, n4, n5);

          Kokkos::parallel_for(
              teamRange, [=](int i, int j, int k, int l, int m, int n) {
                v(leagueRank, i, j, k, l, m, n) +=
                    fillFlattenedIndex(leagueRank, i, j, k, l, m, n);
              });
        });

    HostViewType h_view =
        Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), v);

    check_result_7D(h_view, fillFlattenedIndex);
  }

  template <Kokkos::Iterate OuterDirection = Kokkos::Iterate::Default,
            Kokkos::Iterate InnerDirection = Kokkos::Iterate::Default>
  static void test_parallel_for_8D_MDThreadVectorRange(int* dims,
                                                       const int initValue) {
    using ViewType     = typename Kokkos::View<DataType********, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int numTeams = dims[0];
    int n0       = dims[1];
    int n1       = dims[2];
    int n2       = dims[3];
    int n3       = dims[4];
    int n4       = dims[5];
    int n5       = dims[6];
    int n6       = dims[7];

    ViewType v("v", numTeams, n0, n1, n2, n3, n4, n5, n6);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(numTeams, n0, n1, n2, n3, n4, n5, n6);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(numTeams, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange =
              Kokkos::MDThreadVectorRange<OuterDirection, InnerDirection>(
                  team, n0, n1, n2, n3, n4, n5, n6);

          Kokkos::parallel_for(
              teamRange, [=](int i, int j, int k, int l, int m, int n, int o) {
                v(leagueRank, i, j, k, l, m, n, o) +=
                    fillFlattenedIndex(leagueRank, i, j, k, l, m, n, o);
              });
        });

    HostViewType h_view =
        Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), v);

    check_result_8D(h_view, fillFlattenedIndex);
  }

  template <Kokkos::Iterate OuterDirection = Kokkos::Iterate::Default,
            Kokkos::Iterate InnerDirection = Kokkos::Iterate::Default>
  static void test_parallel_for_3D_MDTeamVectorRange(int* dims,
                                                     const int initValue) {
    using ViewType     = typename Kokkos::View<DataType***, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int numTeams = dims[0];
    int n0       = dims[1];
    int n1       = dims[2];

    ViewType v("v", numTeams, n0, n1);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(numTeams, n0, n1);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(numTeams, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange =
              Kokkos::MDTeamVectorRange<OuterDirection, InnerDirection>(team,
                                                                        n0, n1);

          Kokkos::parallel_for(teamRange, [=](int i, int j) {
            v(leagueRank, i, j) += fillFlattenedIndex(leagueRank, i, j);
          });
        });

    HostViewType h_view =
        Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), v);

    check_result_3D(h_view, fillFlattenedIndex);
  }

  template <Kokkos::Iterate OuterDirection = Kokkos::Iterate::Default,
            Kokkos::Iterate InnerDirection = Kokkos::Iterate::Default>
  static void test_parallel_for_4D_MDTeamVectorRange(int* dims,
                                                     const int initValue) {
    using ViewType     = typename Kokkos::View<DataType****, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int numTeams = dims[0];
    int n0       = dims[1];
    int n1       = dims[2];
    int n2       = dims[3];

    ViewType v("v", numTeams, n0, n1, n2);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(numTeams, n0, n1, n2);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(numTeams, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange =
              Kokkos::MDTeamVectorRange<OuterDirection, InnerDirection>(
                  team, n0, n1, n2);

          Kokkos::parallel_for(teamRange, [=](int i, int j, int k) {
            v(leagueRank, i, j, k) += fillFlattenedIndex(leagueRank, i, j, k);
          });
        });

    HostViewType h_view =
        Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), v);

    check_result_4D(h_view, fillFlattenedIndex);
  }

  template <Kokkos::Iterate OuterDirection = Kokkos::Iterate::Default,
            Kokkos::Iterate InnerDirection = Kokkos::Iterate::Default>
  static void test_parallel_for_5D_MDTeamVectorRange(int* dims,
                                                     const int initValue) {
    using ViewType     = typename Kokkos::View<DataType*****, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int numTeams = dims[0];
    int n0       = dims[1];
    int n1       = dims[2];
    int n2       = dims[3];
    int n3       = dims[4];

    ViewType v("v", numTeams, n0, n1, n2, n3);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(numTeams, n0, n1, n2, n3);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(numTeams, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange =
              Kokkos::MDTeamVectorRange<OuterDirection, InnerDirection>(
                  team, n0, n1, n2, n3);

          Kokkos::parallel_for(teamRange, [=](int i, int j, int k, int l) {
            v(leagueRank, i, j, k, l) +=
                fillFlattenedIndex(leagueRank, i, j, k, l);
          });
        });

    HostViewType h_view =
        Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), v);

    check_result_5D(h_view, fillFlattenedIndex);
  }

  template <Kokkos::Iterate OuterDirection = Kokkos::Iterate::Default,
            Kokkos::Iterate InnerDirection = Kokkos::Iterate::Default>
  static void test_parallel_for_6D_MDTeamVectorRange(int* dims,
                                                     const int initValue) {
    using ViewType     = typename Kokkos::View<DataType******, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int numTeams = dims[0];
    int n0       = dims[1];
    int n1       = dims[2];
    int n2       = dims[3];
    int n3       = dims[4];
    int n4       = dims[5];

    ViewType v("v", numTeams, n0, n1, n2, n3, n4);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(numTeams, n0, n1, n2, n3, n4);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(numTeams, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange =
              Kokkos::MDTeamVectorRange<OuterDirection, InnerDirection>(
                  team, n0, n1, n2, n3, n4);

          Kokkos::parallel_for(
              teamRange, [=](int i, int j, int k, int l, int m) {
                v(leagueRank, i, j, k, l, m) +=
                    fillFlattenedIndex(leagueRank, i, j, k, l, m);
              });
        });

    HostViewType h_view =
        Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), v);

    check_result_6D(h_view, fillFlattenedIndex);
  }

  template <Kokkos::Iterate OuterDirection = Kokkos::Iterate::Default,
            Kokkos::Iterate InnerDirection = Kokkos::Iterate::Default>
  static void test_parallel_for_7D_MDTeamVectorRange(int* dims,
                                                     const int initValue) {
    using ViewType     = typename Kokkos::View<DataType*******, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int numTeams = dims[0];
    int n0       = dims[1];
    int n1       = dims[2];
    int n2       = dims[3];
    int n3       = dims[4];
    int n4       = dims[5];
    int n5       = dims[6];

    ViewType v("v", numTeams, n0, n1, n2, n3, n4, n5);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(numTeams, n0, n1, n2, n3, n4, n5);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(numTeams, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange =
              Kokkos::MDTeamVectorRange<OuterDirection, InnerDirection>(
                  team, n0, n1, n2, n3, n4, n5);

          Kokkos::parallel_for(
              teamRange, [=](int i, int j, int k, int l, int m, int n) {
                v(leagueRank, i, j, k, l, m, n) +=
                    fillFlattenedIndex(leagueRank, i, j, k, l, m, n);
              });
        });

    HostViewType h_view =
        Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), v);

    check_result_7D(h_view, fillFlattenedIndex);
  }

  template <Kokkos::Iterate OuterDirection = Kokkos::Iterate::Default,
            Kokkos::Iterate InnerDirection = Kokkos::Iterate::Default>
  static void test_parallel_for_8D_MDTeamVectorRange(int* dims,
                                                     const int initValue) {
    using ViewType     = typename Kokkos::View<DataType********, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int numTeams = dims[0];
    int n0       = dims[1];
    int n1       = dims[2];
    int n2       = dims[3];
    int n3       = dims[4];
    int n4       = dims[5];
    int n5       = dims[6];
    int n6       = dims[7];

    ViewType v("v", numTeams, n0, n1, n2, n3, n4, n5, n6);
    FillConstant fillConstant(initValue);
    FillFlattenedIndex fillFlattenedIndex(numTeams, n0, n1, n2, n3, n4, n5, n6);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(numTeams, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamType& team) {
          int leagueRank = team.league_rank();

          auto teamRange =
              Kokkos::MDTeamVectorRange<OuterDirection, InnerDirection>(
                  team, n0, n1, n2, n3, n4, n5, n6);

          Kokkos::parallel_for(
              teamRange, [=](int i, int j, int k, int l, int m, int n, int o) {
                v(leagueRank, i, j, k, l, m, n, o) +=
                    fillFlattenedIndex(leagueRank, i, j, k, l, m, n, o);
              });
        });

    HostViewType h_view =
        Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), v);

    check_result_8D(h_view, fillFlattenedIndex);
  }
};

template <typename ExecSpace>
struct TestMDParallelReduce {
  using TeamType = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  static void test_reduce2_MDTeamThreadRange(const int N0, const int N1) {
    /*
    using DataType     = int;
    using ViewType     = typename Kokkos::View<DataType**, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = 1;
    int teamSize   = 1;

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
          int teamSum     = 0;

          Kokkos::parallel_reduce(
              Kokkos::MDTeamThreadRange(team, N0, N1),
              [=](const int& i, const int& j, int& threadSum) {
                threadSum += v(i, j);
              },
              teamSum);

          Kokkos::single(Kokkos::PerTeam(team),
                         [&] { sharedArray(leagueRank) += teamSum; });
        });

    auto h_sharedArray =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
    sharedArray);

    ASSERT_EQ(h_sharedArray(0), 3 * N0 * N1);
    */
  }

  static void test_reduce2_MDThreadVectorRange(const int N0, const int N1) {
    using DataType     = int;
    using ViewType     = typename Kokkos::View<DataType**, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = 1;
    int teamSize   = 1;

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
          int teamSum     = 0;

          Kokkos::parallel_reduce(
              Kokkos::MDThreadVectorRange(team, N0, N1),
              [=](const int& i, const int& j, int& threadSum) {
                threadSum += v(i, j);
              },
              teamSum);

          Kokkos::single(Kokkos::PerTeam(team),
                         [&] { sharedArray(leagueRank) += teamSum; });
        });

    auto h_sharedArray =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sharedArray);

    ASSERT_EQ(h_sharedArray(0), 3 * N0 * N1);
  }

  static void test_reduce2_MDTeamVectorRange(const int N0, const int N1) {
    using DataType     = int;
    using ViewType     = typename Kokkos::View<DataType**, ExecSpace>;
    using HostViewType = typename ViewType::HostMirror;

    int leagueSize = 1;
    int teamSize   = 1;

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
          int teamSum     = 0;

          Kokkos::parallel_reduce(
              Kokkos::MDTeamVectorRange(team, N0, N1),
              [=](const int& i, const int& j, int& threadSum) {
                threadSum += v(i, j);
              },
              teamSum);

          Kokkos::single(Kokkos::PerTeam(team),
                         [&] { sharedArray(leagueRank) += teamSum; });
        });

    auto h_sharedArray =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sharedArray);

    // FIXME h_sharedArray should be 2D?
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

}  // namespace MDParallelism

}  // namespace Test

/*--------------------------------------------------------------------------*/

namespace Test {
TEST(TEST_CATEGORY, MDParallelFor) {
  using namespace MDParallelism;
  // int dims[] = {15, 16, 16, 16, 16, 16, 16, 16};
  int dims[]          = {4, 4, 4, 4, 4, 4, 4, 4};
  const int initValue = 5;

  constexpr auto Left  = Kokkos::Iterate::Left;
  constexpr auto Right = Kokkos::Iterate::Right;

  {
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_3D_MDTeamThreadRange<
        Left>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_3D_MDTeamThreadRange<
        Right>(dims, initValue);

    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_4D_MDTeamThreadRange<
        Left>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_4D_MDTeamThreadRange<
        Right>(dims, initValue);

    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_5D_MDTeamThreadRange<
        Left>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_5D_MDTeamThreadRange<
        Right>(dims, initValue);

    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_6D_MDTeamThreadRange<
        Left>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_6D_MDTeamThreadRange<
        Right>(dims, initValue);

    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_7D_MDTeamThreadRange<
        Left>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_7D_MDTeamThreadRange<
        Right>(dims, initValue);

    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_8D_MDTeamThreadRange<
        Left>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_8D_MDTeamThreadRange<
        Right>(dims, initValue);
  }

  {
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_3D_MDThreadVectorRange<
        Left, Left>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_3D_MDThreadVectorRange<
        Left, Right>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_3D_MDThreadVectorRange<
        Right, Left>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_3D_MDThreadVectorRange<
        Right, Right>(dims, initValue);

    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_4D_MDThreadVectorRange<
        Left, Left>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_4D_MDThreadVectorRange<
        Left, Right>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_4D_MDThreadVectorRange<
        Right, Left>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_4D_MDThreadVectorRange<
        Right, Right>(dims, initValue);

    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_5D_MDThreadVectorRange<
        Left, Left>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_5D_MDThreadVectorRange<
        Left, Right>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_5D_MDThreadVectorRange<
        Right, Left>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_5D_MDThreadVectorRange<
        Right, Right>(dims, initValue);

    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_6D_MDThreadVectorRange<
        Left, Left>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_6D_MDThreadVectorRange<
        Left, Right>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_6D_MDThreadVectorRange<
        Right, Left>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_6D_MDThreadVectorRange<
        Right, Right>(dims, initValue);

    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_7D_MDThreadVectorRange<
        Left, Left>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_7D_MDThreadVectorRange<
        Left, Right>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_7D_MDThreadVectorRange<
        Right, Left>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_7D_MDThreadVectorRange<
        Right, Right>(dims, initValue);

    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_8D_MDThreadVectorRange<
        Left, Left>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_8D_MDThreadVectorRange<
        Left, Right>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_8D_MDThreadVectorRange<
        Right, Left>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_8D_MDThreadVectorRange<
        Right, Right>(dims, initValue);
  }

  {
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_3D_MDTeamVectorRange<
        Left, Left>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_3D_MDTeamVectorRange<
        Left, Right>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_3D_MDTeamVectorRange<
        Right, Left>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_3D_MDTeamVectorRange<
        Right, Right>(dims, initValue);

    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_4D_MDTeamVectorRange<
        Left, Left>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_4D_MDTeamVectorRange<
        Left, Right>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_4D_MDTeamVectorRange<
        Right, Left>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_4D_MDTeamVectorRange<
        Right, Right>(dims, initValue);

    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_5D_MDTeamVectorRange<
        Left, Left>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_5D_MDTeamVectorRange<
        Left, Right>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_5D_MDTeamVectorRange<
        Right, Left>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_5D_MDTeamVectorRange<
        Right, Right>(dims, initValue);

    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_6D_MDTeamVectorRange<
        Left, Left>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_6D_MDTeamVectorRange<
        Left, Right>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_6D_MDTeamVectorRange<
        Right, Left>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_6D_MDTeamVectorRange<
        Right, Right>(dims, initValue);

    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_7D_MDTeamVectorRange<
        Left, Left>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_7D_MDTeamVectorRange<
        Left, Right>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_7D_MDTeamVectorRange<
        Right, Left>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_7D_MDTeamVectorRange<
        Right, Right>(dims, initValue);

    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_8D_MDTeamVectorRange<
        Left, Left>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_8D_MDTeamVectorRange<
        Left, Right>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_8D_MDTeamVectorRange<
        Right, Left>(dims, initValue);
    TestMDParallelFor<TEST_EXECSPACE>::test_parallel_for_8D_MDTeamVectorRange<
        Right, Right>(dims, initValue);
  }
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
