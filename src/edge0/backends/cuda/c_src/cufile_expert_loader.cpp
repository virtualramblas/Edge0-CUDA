#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cufile.h>
#include <torch/extension.h>

class GDSNativeLoader {
private:
    int fd_;
    CUfileHandle_t cf_handle_;
    bool driver_opened_;

public:
    GDSNativeLoader(const std::string& filepath) : fd_(-1), driver_opened_(false) {
        // 1. Initialize cuFile driver
        CUfileError_t status = cuFileDriverOpen();
        if (status.err != CU_FILE_SUCCESS) {
            throw std::runtime_error("Failed to initialize cuFile driver.");
        }
        driver_opened_ = true;

        // 2. Open file with direct I/O required for GDS
        fd_ = open(filepath.c_str(), O_RDONLY | O_DIRECT);
        if (fd_ < 0) {
            throw std::runtime_error("Failed to open file with O_DIRECT.");
        }

        // 3. Register OS file descriptor with cuFile
        CUfileDescr_t descr;
        memset(&descr, 0, sizeof(CUfileDescr_t));
        descr.handle.fd = fd_;
        descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

        status = cuFileHandleRegister(&cf_handle_, &descr);
        if (status.err != CU_FILE_SUCCESS) {
            close(fd_);
            throw std::runtime_error("Failed to register cuFile handle.");
        }
    }

    // Register pre-allocated GPU buffer pool with GDS driver
    void register_gpu_buffer(torch::Tensor dev_tensor) {
        void* dev_ptr = dev_tensor.data_ptr();
        size_t size = dev_tensor.numel() * dev_tensor.element_size();
        
        CUfileError_t status = cuFileBufRegister(dev_ptr, size, 0);
        if (status.err != CU_FILE_SUCCESS) {
            throw std::runtime_error("cuFileBufRegister failed for GPU buffer.");
        }
    }

    // Direct DMA from NVMe into GPU memory
    ssize_t load_expert(torch::Tensor target_tensor, int64_t file_offset, int64_t bytes_to_read) {
        void* dev_ptr = target_tensor.data_ptr();
        
        // Zero-copy direct NVMe to GPU DMA transfer
        ssize_t ret = cuFileRead(
            cf_handle_,
            dev_ptr,
            static_cast<size_t>(bytes_to_read),
            static_cast<off_t>(file_offset),
            0 /* bufPtr_offset */
        );
        return ret;
    }

    ~GDSNativeLoader() {
        if (cf_handle_) {
            cuFileHandleDeregister(cf_handle_);
        }
        if (fd_ >= 0) {
            close(fd_);
        }
        if (driver_opened_) {
            cuFileDriverClose();
        }
    }
};

// PyBind11 module bindings for PyTorch integration
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<GDSNativeLoader>(m, "GDSNativeLoader")
        .def(py::init<const std::string&>())
        .def("register_gpu_buffer", &GDSNativeLoader::register_gpu_buffer)
        .def("load_expert", &GDSNativeLoader::load_expert);
}