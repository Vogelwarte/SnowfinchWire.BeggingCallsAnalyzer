## Installing the application

0. Download the repository https://github.com/Vogelwarte/SnowfinchWire.BeggingCallsAnalyzer/archive/refs/heads/main.zip

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
    Run the `install.ps1` script by opening PowerShell in its parent directory. You can do this by holding shift and right on the clicking the folder . The script should output **SUCCESS** at the very end. If it did not, make sure Python 3.9 is installed, the command from step 2 works and you have access to the internet.