import sys
import importlib.util
from dataclasses import is_dataclass

def load_config_from_path(config_path: str, config_name: str="Config", config_exp_name: str=None):
    try:
        spec = importlib.util.spec_from_file_location(name="config_module", location=config_path)
        if spec is None:
            raise ImportError(f"Could not load spec for module at path: {config_path}")

        config_module = importlib.util.module_from_spec(spec)
        
        spec.loader.exec_module(config_module)
        
        _config_class = getattr(config_module, config_name)
        if config_exp_name:
            config_class = getattr(_config_class, config_exp_name)
        else:
            config_class = _config_class

    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{config_path}'")
        sys.exit(1)
    except AttributeError:
        print(f"Error: Class '{config_name}' not found in '{config_path}'")
        sys.exit(1)

    # 가져온 객체가 정말 dataclass인지 확인합니다.
    if not is_dataclass(_config_class):
        raise TypeError(f"Error: '{config_name}' in '{config_path}' is not a dataclass. but {type(config_class)}")
    # dataclass의 인스턴스를 생성하여 반환합니다.
    return config_class