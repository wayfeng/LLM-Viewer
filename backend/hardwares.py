hardware_params = {
    "custom": {"bandwidth": 0, "FP16": 0, "INT8": 0, "onchip_buffer": 0},
    # NOTICES: For GPU, we use Register File Size as on-chip buffer size
    # https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf
    # NOTICE: V100 not support INT8 in tensor core, so INT8 performance is not good
    "nvidia_V100": {"bandwidth": 900e9, "FP16": 112e12, "INT8": 62e12, "onchip_buffer": 20480e3},
    # https://images.nvidia.com/aem-dam/en-zz/Solutions/technologies/NVIDIA-ADA-GPU-PROVIZ-Architecture-Whitepaper_1.1.pdf
    "nvidia_A6000": {"bandwidth": 768e9, "FP16": 154.8e12, "INT8": 309.7e12, "onchip_buffer": 21504e3},
    # https://images.nvidia.com/aem-dam/en-zz/Solutions/technologies/NVIDIA-ADA-GPU-PROVIZ-Architecture-Whitepaper_1.1.pdf
    "nvidia_A6000_Ada": {"bandwidth": 960e9, "FP16": 364.2e12, "INT8": 728.5e12, "onchip_buffer": 36352e3},
    # https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
    # Ampere's SM has 256KB RF, max 164KB Shared Mem
    "nvidia_A100": {"bandwidth": 1555e9, "FP16": 312e12, "INT8": 624e12, "onchip_buffer": 27648e3},  # use 40G data
    "nvidia_A100_40G": {"bandwidth": 1555e9, "FP16": 312e12, "INT8": 624e12, "onchip_buffer": 27648e3},
    "nvidia_A100_80G": {"bandwidth": 2039e9, "FP16": 312e12, "INT8": 624e12, "onchip_buffer": 27648e3},
    "nvidia_A800_80G_SXM": {"bandwidth": 2039e9, "FP16": 312e12, "INT8": 624e12, "onchip_buffer": 27648e3},
    "nvidia_A40": {"bandwidth": 696e9, "FP16": 149.7e12, "INT8": 299.3e12, "onchip_buffer": 21504e3},
    # https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper
    "nvidia_H100": {
        "bandwidth": 3072e9,
        "FP16": 1979e12 / 2,
        "INT8": 3958e12 / 2,
        "onchip_buffer": 33792e3,
    },  # use SXM data
    "nvidia_H100_SXM": {"bandwidth": 3072e9, "FP16": 1979e12 / 2, "INT8": 3958e12 / 2, "onchip_buffer": 33792e3},
    "nvidia_H100_PCIe": {"bandwidth": 2048e9, "FP16": 1513e12 / 2, "INT8": 3026e12 / 2, "onchip_buffer": 29184e3},
    # https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-v2.1.pdf
    # Ada SM has 256 KB Register File, and 128 KB of L1/Shared Memory
    "nvidia_L40": {"bandwidth": 864e9, "FP16": 181e12, "INT8": 362e12, "onchip_buffer": 36352e3},
    # https://www.techpowerup.com/gpu-specs/geforce-rtx-4090.c3889
    "nvidia_RTX4090": {"bandwidth": 1008e9, "FP16": 82.5e12, "INT8": 1321e12, "onchip_buffer": 72*1024*1024},
    # Intel Skylake-X (Skylake-X, Cascade Lake) Intel Xeon Phi (Knights Landing, Knights Mill) Intel Ice Lake, Tiger Lake and Rocket Lake
    # support AVX-512 & FMA (512-bit), they has throughput of 1 cycle
    # https://www.intel.com/content/www/us/en/products/sku/230496/intel-core-i913900k-processor-36m-cache-up-to-5-80-ghz/specifications.html
    "intel_ARC_B60": {"bandwidth": 456e9, "FP16": 96e12, "INT8": 197e12, "onchip_buffer": 18e6},
    # https://www.intel.com/content/www/us/en/products/sku/243916/intel-arc-pro-b60-graphics/specifications.html
    "intel_ARC_B570": {"bandwidth": 380e9, "FP16": 100e12, "INT8": 203e12, "onchip_buffer": 13.5e6},
    # https://www.intel.com/content/www/us/en/products/sku/241676/intel-arc-b570-graphics/specifications.html
    "intel_CRI": {"bandwidth": 1536e9, "FP16": 450e12, "INT8": 900e12, "onchip_buffer": 24e6},
    "intel_Ultra_155H iGPU": {"bandwidth": 5.6e9 * 8 * 2, "FP16": 9e12, "INT8": 18e12, "onchip_buffer": 16e6},
    "intel_Ultra_225H iGPU": {"bandwidth": 6.4e9 * 8 * 2, "FP16": 31.5e12, "INT8": 63e12, "onchip_buffer": 24e6},
    "intel_Ultra_255H iGPU": {"bandwidth": 6.4e9 * 8 * 2, "FP16": 37e12, "INT8": 74e12, "onchip_buffer": 24e6},
    "intel_ARC_B390 (PTL iGPU)": {"bandwidth": 9.6e9 * 8 * 2, "FP16": 61e12, "INT8": 122e12, "onchip_buffer": 32e6},
}


def get_hardware_info(hardware: str, w_bit: int, a_bit: int, kv_bit: int):
    if not hardware in hardware_params:
        raise ValueError(f"Unsupported hardware:{hardware}")
    bandwidth = hardware_params[hardware]["bandwidth"]
    if w_bit <= 8 and a_bit <= 8 and kv_bit <= 8:
        max_OPS = hardware_params[hardware]["INT8"]
    else:
        max_OPS = hardware_params[hardware]["FP16"]
    onchip_buffer = hardware_params[hardware]["onchip_buffer"]
    return bandwidth, max_OPS, onchip_buffer


def get_available_hardwares():
    return list(hardware_params.keys())

def get_hardware_params(hardware):
    if not hardware in hardware_params:
        raise ValueError(f"Unsupported hardware:{hardware}")
    params = hardware_params[hardware]
    return {
        "bandwidth": params["bandwidth"] // 1e9,  # convert to GB/s
        "FP16": params["FP16"] // 1e12,  # convert to TFLOPS
        "INT8": params["INT8"] // 1e12,  # convert to TOPS
        "onchip_buffer": params["onchip_buffer"] // 1e6,  # convert to MB
    }