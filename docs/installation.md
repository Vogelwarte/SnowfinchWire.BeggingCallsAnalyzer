## Installing the application

1. 
    Install the latest version of Python 3.9 (https://www.python.org/downloads/release/python-3913/). Download the _Windows installer (64-bit)_.
    * Make sure the appropriate checkboxes for installing the "py launcher" and adding the python executable to PATH are checked.
2. 
    After the installation is complete, open PowerShell terminal and make sure Python is installed correctly by running the following command:
    ```
    py --list
    ```
    The output should contain at least the following line:
    ```
    -V:3.9           Python 3.9
    ```
    If it is not there, try restarting your computer or installing Python 3.9 again

3. 
    Download and unzip the source code of the latest release of the application https://github.com/Vogelwarte/SnowfinchWire.BeggingCallsAnalyzer/releases/latest
4. 
    Open PowerShell in the `SnowfinchWire.BeggingCallsAnalyzer` directory. You can do this by holding shift and right clicking on empty space in the folder or by launching in from the start menu and navigating to this directory. You can do this by using the `cd` command, for example `cd C:\Desktop\SnowfinchWire.BeggingCallsAnalyzer` and pressing enter.
    
    After entering the directory in PowerShell, run the installation script by inputting `.\install.ps1` and pressing enter. The script should output **SUCCESS** at the very end. If it did not, make sure Python 3.9 is installed, the command from step 2 works and you have access to the internet.

5. The best performing models useful when running classification can be found in the `models` directory of the downloaded project.