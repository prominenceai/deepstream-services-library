### Using VSCode
This repository includes files/configurations to build/debug DSL and your C/C++/Python applications using VSCode.

**Setup**
- Install VSCode
- Setup Remote SSH to Jetson Nano (https://code.visualstudio.com/docs/remote/ssh)
- Install workspace recommended extensions on remote
  - Using the **Command Palette**  
    - Bring up the **Command Palette** (`Ctrl+Shift+P`)
    - Type `Show Recommended Extensions` to filter the commands
    - Press `Enter` to select the command which will switch to the **Extensions View** showing the the workspace recommendations
    - Install all the recommended extensions by clicking on the `Install Workspace Recommended Extensions` icon (Cloud Download Icon) in the top right of the **Extensions View**
  - Using the **Extensions View**
    - Swith to the **Extensions View** by clicking on the Extensions icon in the Activity Bar on the side of VS Code or by using the **View: Extensions** command (`Ctrl+Shift+X`)
    - In the search extensions field, type `@recommended` to show the workspace recommended extensions
    - Install all the recommended extensions by clicking on the `Install Workspace Recommended Extensions` icon (Cloud Download Icon) in the top right of the **Extensions View**

**Build**

*To run a task, either click on `Terminal->Run Task` or open the **Command Palette** (`Ctrl+Shift+P`) and choose `Run Task`*

- **DSL Test App (DEBUG)** - Run `build:test-app:debug` task
- **DSL Lib** - Run `build:lib` task
  - This might require the current user's password as this will also run `sudo make install`
- **DSL Lib (DEBUG)** - Run `build:lib:debug` task

**Debug**

*Different launch configurations have been created, with build tasks as prerequisites. Depending on the scenario, the correct configuration needs to be selected from the **Run and Debug View** (`Ctrl+Shift+D`)*

*Start Debugging by pressing F5. Depending on the selected configuration, the required builds tasks will be performed after which the debug session will be started*

*To set a breakpoint, either:*
- *In a code file, click to the left of the line number for the code file*
- *Press `F9` on the line*

---

- To debug C++ Example
  - Open C++ example file (`/examples/cpp`)
  - Place breakpoint(s) in *.cpp file
  - Choose `CPP: Current Example` config
  - Press F5
- To debug DSL C++ unit tests
  - Place breakpoint(s) in *.cpp unit test file(s) (`/test/**/*.cpp`)
  - Choose `CPP: Test App` config
  - Press F5
  - **Optional** Limit unit tests by passing tags
    - Modify the `args` property of the `CPP: Test App` launch config in `./vscode/launch.json`
      ```json
      "args": ["[OdeAction]"],
      ```
- To debug DSL C++ using Python file:
  - Place breakpoint(s) in *.cpp file
  - Open Python file
  - Choose `CPP: Current Python File` config
  - Press F5
- To debug Python Example:
  - Open Python example file (`/examples/python`)
  - Place breakpoint(s) in *.py file
  - Choose `Python: Current File` config
  - Press F5
