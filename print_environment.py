import platform, importlib
print("python", platform.python_version(), platform.platform())
for m in ["numpy","pandas","scipy","sklearn","statsmodels","torch","transformers","sentence_transformers","datasets"]:
    try: print(m, importlib.import_module(m).__version__)
    except Exception as e: print(m, "not installed")
try:
    import torch; print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
except Exception: pass
