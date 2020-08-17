#ifndef KOKKOS_SYCL_KERNELLAUNCH_HPP_
#define KOKKOS_SYCL_KERNELLAUNCH_HPP_

#include <SYCL/Kokkos_TypeChecks.hpp>
#include <SYCL/Kokkos_SYCL_Error.hpp>
#include <CL/sycl.hpp>

namespace Kokkos {
namespace Experimental {
namespace Impl {

// Range Launch
template<class Driver>
void sycl_launch(const Driver driver) {
       driver.m_policy.space().impl_internal_space_instance()->m_queue->wait();
       driver.m_policy.space().impl_internal_space_instance()->m_queue->submit([&] (cl::sycl::handler& cgh) {
         auto exec_range = driver.m_policy.end()-driver.m_policy.begin();
         auto m_functor = driver.m_functor;
         cgh.parallel_for (
            cl::sycl::range<1>(exec_range), [=] (cl::sycl::item<1> item) {
              int id = item.get_linear_id();
                m_functor(id);        
         });
      });
      driver.m_policy.space().impl_internal_space_instance()->m_queue->wait();
}
// MDRange Launch
template<class FunctorType, class ...Traits>
void sycl_launch(const Kokkos::Impl::ParallelFor<FunctorType, MDRangePolicy<Traits...>, Kokkos::Experimental::SYCL> driver) {
       throw "FAAAA";
//       typedef decltype(driver) Driver;
//
//       typedef Kokkos::MDRangePolicy<Traits...> MDRangePolicy;
//       typedef typename MDRangePolicy::impl_range_policy Policy;
//       typedef typename MDRangePolicy::work_tag WorkTag;
//
//       typedef typename Policy::WorkRange WorkRange;
//       typedef typename Policy::member_type Member;
//
//       const Policy m_policy;  // construct as RangePolicy( 0, num_tiles
//
//       driver.m_policy.space().impl_internal_space_instance()->m_queue->wait();
//       size_t flatRange = driver.m_policy.m_lower.size();
//       std::cout << "flatRange = " << flatRange << "\n";
//       exit(0);
//       #ifndef SYCL_USE_BIND_LAUNCH
//       driver.m_policy.space().impl_internal_space_instance()->m_queue->submit([&] (cl::sycl::handler& cgh) {
//         //CGH_PARALLEL_FOR (
//         #ifdef SYCL_JUST_DONT_NAME_KERNELS
//         cgh.parallel_for (//<class kokkos_sycl_functor> (
//         #else
//         cgh.parallel_for <class kokkos_sycl_functor<Driver>> (
//         #endif
//            cl::sycl::range<1>(flatRange), [=] (cl::sycl::item<1> item) {
//              int id = item.get_linear_id();
//                driver.m_functor(id);        
//         });
//      });
//      driver.m_policy.space().impl_internal_space_instance()->m_queue->wait();
//      #else
//      driver.m_policy.space().impl_internal_space_instance()->m_queue->submit(
//        std::bind(Kokkos::Experimental::Impl::sycl_launch_bind<Driver>,driver,std::placeholders::_1));
//      #endif
}



} // Impl namespace
} // Experimental namespace
} // Kokkos namespace

#endif // KOKKOS_SYCL_KERNELLAUNCH_HPP_
