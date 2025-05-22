import torch
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True,
                                         llm_int8_threshold=200.0)

load_config = {
    "torch_dtype": torch.bfloat16,
    "low_cpu_mem_usage": True,
    "device_map": "auto",
    "quantization_config": quantization_config,
}

MAX_LEN_TOKENS = 4096



#import torch
#from transformers import BitsAndBytesConfig

#quantization_config = BitsAndBytesConfig(
 #   load_in_8bit=True,           # Efficient 8-bit loading
  #  llm_int8_threshold=200.0,    # Use default threshold for Phi 2 or adjust based on your model's precision needs
  #  llm_int8_skip_modules=None,  # Optional: specify modules to skip 8-bit quantization if necessary
#)
#load_config = {
 #   "torch_dtype": torch.bfloat16,   # Efficient for Phi 2
  #  "low_cpu_mem_usage": True,       # Optimize CPU memory usage
   # "device_map": "auto",            # Automatic device placement
   # "quantization_config": quantization_config,  # Apply the quantization
#}  
cv
