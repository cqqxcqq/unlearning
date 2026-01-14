
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import CleanSlateConfig

def verify_trainable():
    print("Verifying model gradients...")
    config = CleanSlateConfig()
    
    # We mock the config to point to where we think the model is, 
    # or rely on the user having it in the default location.
    print(f"Loading from: {config.canary_checkpoint}")
    
    try:
        if config.use_lora:
            base_model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=config.dtype,
                device_map="auto" if config.device == "cuda" else None
            )
            # This is the key line we changed
            model = PeftModel.from_pretrained(base_model, config.canary_checkpoint, is_trainable=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config.canary_checkpoint,
                torch_dtype=config.dtype,
                device_map="auto" if config.device == "cuda" else None
            )
            
        # Check gradients
        trainable_params = 0
        all_param = 0
        for name, param in model.named_parameters():
            all_param += 1
            if param.requires_grad:
                trainable_params += 1
                if trainable_params <= 5:
                    print(f"  Pos: {name} requires_grad={param.requires_grad}")
        
        print(f"\nTotal params: {all_param}")
        print(f"Trainable params: {trainable_params}")
        
        if trainable_params > 0:
            print("\nSUCCESS: Model has trainable parameters!")
        else:
            print("\nFAILURE: Model has NO trainable parameters.")
            
    except Exception as e:
        print(f"\nError checking model: {e}")
        print("Ensure you are running this from the directory containing cleanslate_outputs.")

if __name__ == "__main__":
    verify_trainable()
