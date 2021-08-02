### Using VSCode
This repository includes files/configurations to build/debug DSL and your C/C++/Python applications using VSCode.

VSCode C++ Extension v0.30.0-insiders4 required for ARM64 support using Remote Extensions

Setup
- Install VSCode
- Setup Remote SSH to Jetson Nano (https://code.visualstudio.com/docs/remote/ssh)
- Install Python Extension (ms-python.python) on remote
- Install C++ Extension (ms-vscode.cpptools) on remote

To build C++ DSL lib
- `Ctrl+Shift+Build` 

To debug Python:
- Open Python file
- Place breakpoint
- Choose `Python: Current File` config (`Ctrl+Shift+D` - choose from dropdown)
- Press F5

To debug C++ using Python file:
- Place breakpoints in C++ files
- Open Python file
- Choose `CPP: Current Python File` config (`Ctrl+Shift+D` - choose from dropdown)
- Press F5
