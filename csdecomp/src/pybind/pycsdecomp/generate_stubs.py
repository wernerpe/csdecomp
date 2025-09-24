import sys
import os
import subprocess

def generate_stubs(module_dir):
    print(f"Working directory: {os.getcwd()}")
    print(f"Module directory: {module_dir}")
    print(f"Contents before: {os.listdir(module_dir)}")
    
    # Add the module directory to Python path
    sys.path.insert(0, module_dir)
    
    try:
        # Use pybind11-stubgen which is designed specifically for pybind11 modules
        import pybind11_stubgen
        
        # Run pybind11-stubgen
        cmd = [
            sys.executable, "-m", "pybind11_stubgen",
            "pycsdecomp_bindings",
            "--output-dir", module_dir
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=module_dir)
        
        if result.returncode == 0:
            print("Successfully generated stubs using pybind11-stubgen")
            print(f"Contents after: {os.listdir(module_dir)}")
            
            # Look for any .pyi files created anywhere in the directory tree
            for root, dirs, files in os.walk(module_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    print(f"Found file: {full_path}")
                    if file.endswith('.pyi'):
                        print(f"Found stub file: {full_path}")
                        # If it's not in the right place, copy it
                        expected_path = os.path.join(module_dir, "pycsdecomp_bindings.pyi")
                        if full_path != expected_path:
                            print(f"Copying stub from {full_path} to {expected_path}")
                            with open(full_path, 'r') as src, open(expected_path, 'w') as dst:
                                dst.write(src.read())
                        
        else:
            print(f"pybind11-stubgen failed: {result.stderr}")
            create_minimal_stub(module_dir)
            
    except ImportError:
        print("pybind11-stubgen not available, creating minimal stub")
        create_minimal_stub(module_dir)
    except Exception as e:
        print(f"pybind11-stubgen failed: {e}")
        create_minimal_stub(module_dir)

def create_minimal_stub(module_dir):
    stub_file = os.path.join(module_dir, "pycsdecomp_bindings.pyi")
    with open(stub_file, 'w') as f:
        f.write("# Stub file for pycsdecomp_bindings\n")
        f.write("from typing import Any\n\n")
        f.write("def __getattr__(name: str) -> Any: ...\n")
    print(f"Created minimal stub at {stub_file}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(sys.argv)
        print("Usage: python generate_stubs.py <module_directory>")
        sys.exit(1)
        
    generate_stubs(sys.argv[1])