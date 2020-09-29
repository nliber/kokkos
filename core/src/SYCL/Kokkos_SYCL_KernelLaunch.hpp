#ifndef KOKKOS_SYCL_KERNELLAUNCH_HPP_
#define KOKKOS_SYCL_KERNELLAUNCH_HPP_

#include <SYCL/Kokkos_SYCL_Error.hpp>
#include <CL/sycl.hpp>

template <class T>
class kokkos_sycl_functor;

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <class Driver>
void sycl_launch_bind(Driver tmp, cl::sycl::handler& cgh) {
  cgh.parallel_for(
      cl::sycl::range<1>(tmp.m_policy.end() - tmp.m_policy.begin()), tmp);
}

template <class Driver>
void sycl_launch(const Driver driver) {
  isTriviallyCopyable<decltype(driver.m_functor)>();
  auto& policy  = driver.m_policy;
  auto& functor = driver.m_functor;

  const Kokkos::Experimental::SYCL& space = policy.space();
  Kokkos::Experimental::Impl::SYCLInternal& instance =
      *space.impl_internal_space_instance();
  cl::sycl::queue& queue = *instance.m_queue;
  queue.wait();

  auto end   = policy.end();
  auto begin = policy.begin();
  cl::sycl::range<1> range(end - begin);

  queue.submit([&](cl::sycl::handler& cgh) {
    cgh.parallel_for(range, [=](cl::sycl::item<1> item) {
      int id = item.get_linear_id();
      functor(id);
    });
  });
  queue.wait();
}

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif  // KOKKOS_SYCL_KERNELLAUNCH_HPP_

