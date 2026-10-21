// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
//
// NPU UVA view support for vLLM MRv2.
//
// vLLM's UvaBuffer needs a device Tensor view of a pinned CPU tensor. The
// pinned allocation is registered by torch_npu when
// PYTORCH_NPU_ALLOC_CONF=pinned_mem_register:True is set before process start.
// get_npu_view_from_cpu_tensor only queries that existing registration.
// register_mapped_cpu_tensor separately owns registration of shared mmap storage.

#include <torch/extension.h>
#include <torch_npu/csrc/aten/common/from_blob.h>

#include <acl/acl.h>

#include <cstdint>
#include <limits>
#include <memory>

namespace py = pybind11;

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME npu_uva
#endif

namespace {

#define ACL_CHECK(expr)                                                       \
  do {                                                                        \
    const aclError ret = (expr);                                              \
    TORCH_CHECK(ret == ACL_SUCCESS, #expr, " failed with aclError=", ret);    \
  } while (0)

constexpr c10::DeviceType kNPUDeviceType = c10::DeviceType::PrivateUse1;

void check_cpu_tensor(const at::Tensor& cpu_tensor) {
  TORCH_CHECK(cpu_tensor.device().is_cpu(), "input tensor must be on CPU");
  TORCH_CHECK(cpu_tensor.is_pinned(), "CPU tensor must be pinned");
  TORCH_CHECK(cpu_tensor.numel() > 0,
              "cannot create an NPU UVA view for an empty CPU tensor");
}

void* get_registered_device_pointer(const at::Tensor& cpu_tensor) {
  void* dev_ptr = nullptr;
  // CANN signature:
  //   aclrtHostGetDevicePointer(void *pHost, void **pDevice, uint32_t flag)
  ACL_CHECK(aclrtHostGetDevicePointer(cpu_tensor.data_ptr(), &dev_ptr, 0));
  TORCH_CHECK(dev_ptr != nullptr,
              "aclrtHostGetDevicePointer returned NULL. Ensure "
              "PYTORCH_NPU_ALLOC_CONF includes pinned_mem_register:True before "
              "the process creates pinned memory, and the CANN/driver stack "
              "supports mapped pinned host memory.");
  return dev_ptr;
}

c10::Device make_npu_device(int64_t device_index) {
  TORCH_CHECK(device_index >= 0, "NPU device index must be non-negative");
  TORCH_CHECK(device_index <= std::numeric_limits<c10::DeviceIndex>::max(),
              "NPU device index is out of range: ", device_index);
  return c10::Device(kNPUDeviceType,
                     static_cast<c10::DeviceIndex>(device_index));
}

}  // namespace

// Unlike the pinned-allocator path below, Engram owns an existing MAP_SHARED
// allocation. Keep both the registration and the CPU mmap alive through every
// alias of the returned device tensor; never make a private pinned copy.
struct SharedHostRegistration {
  at::Tensor source;
  int device;
  bool registered = false;

  ~SharedHostRegistration() {
    if (!registered) {
      return;
    }
    int previous = -1;
    aclrtGetDevice(&previous);
    if (aclrtSetDevice(device) == ACL_SUCCESS) {
      const auto result = aclrtHostUnregister(source.data_ptr());
      if (result != ACL_SUCCESS) {
        TORCH_WARN("Engram aclrtHostUnregister failed on device ", device,
                   ": ", result);
      }
    }
    if (previous >= 0 && previous != device) {
      aclrtSetDevice(previous);
    }
  }
};

at::Tensor register_mapped_cpu_tensor(const at::Tensor& source,
                                      int64_t device_index) {
  TORCH_CHECK(source.device().is_cpu() && source.is_contiguous(),
              "Mapped host storage must be a contiguous CPU tensor");
  TORCH_CHECK(source.numel() > 0, "Mapped host storage must not be empty");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(source.data_ptr()) % 4096 == 0,
              "Mapped host storage must start at a 4 KiB aligned address");
  const auto device = make_npu_device(device_index);
  int current = -1;
  ACL_CHECK(aclrtGetDevice(&current));
  TORCH_CHECK(current == device_index,
              "Register shared host storage in the owning NPU device context");
  // Construct in place: a temporary registration would unregister too early.
  auto owner = std::make_shared<SharedHostRegistration>();
  owner->source = source;
  owner->device = static_cast<int>(device_index);
  ACL_CHECK(aclrtHostRegisterV2(source.data_ptr(), source.nbytes(),
                             ACL_HOST_REG_PINNED | ACL_HOST_REG_MAPPED));
  owner->registered = true;
  void* ptr = get_registered_device_pointer(source);
  auto deleter = [owner](void*) {};
  return at_npu::native::from_blob(
      ptr, source.sizes(), source.strides(), deleter,
      source.options().device(device).pinned_memory(false), device);
}

uintptr_t get_device_pointer(const at::Tensor& cpu_tensor) {
  check_cpu_tensor(cpu_tensor);
  return reinterpret_cast<uintptr_t>(get_registered_device_pointer(cpu_tensor));
}

at::Tensor get_npu_view_from_cpu_tensor(const at::Tensor& cpu_tensor,
                                        int64_t device_index) {
  check_cpu_tensor(cpu_tensor);

  void* dev_ptr = get_registered_device_pointer(cpu_tensor);
  const c10::Device npu_device = make_npu_device(device_index);
  auto options = cpu_tensor.options().device(npu_device).pinned_memory(false);

  // Keep the source CPU tensor alive. Registration/unregistration remains owned
  // by torch_npu's pinned host allocator.
  auto deleter = [base = cpu_tensor](void*) mutable {};

  return at_npu::native::from_blob(dev_ptr, cpu_tensor.sizes(),
                                   cpu_tensor.strides(), deleter, options,
                                   npu_device);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("register_mapped_cpu_tensor", &register_mapped_cpu_tensor,
        py::arg("cpu_tensor"), py::arg("device_index"),
        "Register unregistered shared CPU storage and return its owning NPU "
        "alias. Keep an alias alive until all asynchronous reads complete.");
  m.def("get_device_pointer", &get_device_pointer, py::arg("cpu_tensor"),
        "Return the CANN device pointer for a registered pinned CPU tensor");
  m.def("get_npu_view_from_cpu_tensor", &get_npu_view_from_cpu_tensor,
        py::arg("cpu_tensor"), py::arg("device_index") = 0,
        "Return a device=npu Tensor view aliasing a pinned CPU tensor");
}
