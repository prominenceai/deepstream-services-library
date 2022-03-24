# Azure MQTT Protocol Adapter Libraries
NVIDIA provides two Azure MQTT protocol libraries installed with DeepStream under `/opt/nvidia/deepstream/deepstream/lib`
* `libnvds_azure_proto.so` - a device client protocol for sending messages from the device to an Azure IoT Hub instance. Requires minimal setup.
* `libnvds_azure_edge_proto.so` - a module client protocol for bidirectional device-server messaging. Applications linking with DSL must be run in a deployed Docker container.

The protocol adapter libraries are used by the DSL [Message Sink](/docs/api-sink.md#dsl_message_sink_new) and [Message Broker](/docs/api-msg-broker.md) components.

## Contents
* [Common Setup for both Protocol Adapters](#common-setup-for-both-protocol-adapters)
  * [Install Additional device dependencies](#install-additional-device-dependencies)
  * [Setup an Azure IoT Hub Instance](#setup-an-azure-iot-hub-instance)
  * [Register your IoT Edge Device](#register-your-iot-edge-device)
  * [Enable the NVIDIA logger](#enable-the-nvidia-logger)
* [Azure Module Client setup](#azure-module-client-setup)
  * [Setup the Azure IoT Edge runtime on the edge device](#setup-the-azure-iot-edge-runtime-on-the-edge-device)
  * [Build and deploy a Docker Image](#build-and-deploy-a-docker-image)
  * [Grant host access to the local X-server](grant-host-access-to-the-local-x-server)
  * [Deploy IoT Modules](#deploy-iot-modules)
  * [Next Steps and Useful Links](#next-steps-and-useful-links)
* [Trouble Shooting](#trouble-shooting)
  * [Failure installing azure-cli on Jetson](#failure-installing-azure-cli-on-jetson)
  * [Failure verifying azure-cli install on Jetson](#failure-verifying-azure-cli-install-on-jetson)
 
---

# Common Setup for both Protocol Adapters
## Install Additional device dependencies
#### For an x86 computer running Ubuntu:
```
sudo apt-get install -y libcurl3 libssl-dev uuid-dev libglib2.0 libglib2.0-dev libffi6 ibffi-dev
```
#### For Jetson:
```
sudo apt-get install -y libcurl4-openssl-dev libssl-dev uuid-dev libglib2.0 libglib2.0-dev libffi6 libffi-dev
```

Ensure that all `python` dependencies have been installed and that you are using the latest version of `pip`
```bash
sudo apt-get install -y python3-pip python-pip python3-setuptools
sudo pip3 install --upgrade pip
```

## Setup an Azure IoT Hub Instance
Follow the directions at https://docs.microsoft.com/en-us/azure/iot-hub/tutorial-connectivity#create-an-iot-hub.

## Register your IoT Edge Device
There are three methods to register your device as outlined at https://docs.microsoft.com/en-us/azure/iot-edge/how-to-provision-single-device-linux-symmetric?view=iotedge-2020-11&tabs=azure-portal%2Cubuntu#register-your-device - **Azure portal**, **Visual Studio Code**, and **Azure CLI** - with **Azure portal** being the simplest.

The following section details the installation process and steps to use the **Azure CLI**.

### Install the Azure CLI on your edge device
#### For an x86 computer running Ubuntu:
The Azure CLI team maintains a script to run all installation commands in one step. This script is downloaded via curl and piped directly to bash to install the CLI.
```bash
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

#### For Jetson:
Currently, the only way to use Azure CLI on ARM64 is to install from PyPI (https://pypi.org/project/azure-cli/):
```
sudo pip3 install azure-cli
```
If the install fails, see [Failure installing azure-cli on Jetson](#failure-installing-azure-cli-on-jetson) under [Trouble Shooting](#trouble-shooting).

***Important: once the installation is complete you will need to reboot the device!***
```bash
sudo reboot
```
Verify the installation with.
```bash
az --version
```
If the verification command fails, see [Failure verifying azure-cli install on Jetson](#failure-varifying-azure-cli-intall-on-jetson) under [Trouble Shooting](#trouble-shooting).

### Add the azure-iot extension.
``` bash
az extension add --name azure-iot
```

### Create an IoT Edge Device
Log into to your Azure IoT Hub on the cloud from your device terminal window. The following command will bring up the login screen in your browser.
```bash
az login
```

Create an IoT edge device from your device terminal with the following command.
```bash
az iot hub device-identity create --device-id <device-id> --hub-name <hub-name> --edge-enabled
```
where:
* `<device-id>` =  name (string) to identify the new device
* `<hub-name>` = the hub-name you used when you [Setup an Azure IoT Hub Instance](setup_an_azure_iot_hub_instance) above.

Verify the device creation with the following command.
```bash
az iot hub device-identity list --hub-name <hub-name>
```
Or check from your Azure IoT Hub instance on the cloud. From your hub dashboard, select the **`IoT Edge`** item in the left menu pane, you should then see your device listed in the main window.

![](/Images/new-azure-iot-edge-device.png)

Get the connection string for your new device
```bash
az iot hub device-identity connection-string show --device-id <device-id> --hub-name <hub-name>
```
Or copy the `Primary Connection String` from your Azure IoT Hub instance by selecting the device name on IoT Edge main page (see image above).

![](/Images/azure-iot-edge-device-details.png)

You will need the connection-string to use the [Message Sink](/docs/api-sink.md#dsl_sink_message_new) and [Message Broker API](/docs/api-msg-broker.md).

## Enable the NVIDIA logger
#### For Jetson and x86 computers running Ubuntu:
Run the settup script with the following commands.
```bash
sudo chmod u+x /opt/nvidia/deepstream/deepstream/sources/tools/nvds_logger/setup_nvds_logger.sh
sudo /opt/nvidia/deepstream/deepstream/sources/tools/nvds_logger/setup_nvds_logger.sh
```
Log messages will be written to `/tmp/nvds/ds.log`.

**Note:** when using the [Message Sink](/docs/api-sink.md#dsl_sink_message_new), setup errors will result in the Pipeline failing to play. The reason for the failure may be found in the nvds log file, for example
```
Mar 24 16:37:57 prominenceai1-desktop dsl-test-app.exe: DSLOG:NVDS_AZURE_PROTO: Error. Azure connection string not provided#012
Mar 24 16:37:57 prominenceai1-desktop dsl-test-app.exe: DSLOG:NVDS_AZURE_PROTO: nvds_msgapi_connect: Failure in fetching azure connection string
```

Your device setup is now sufficient to use the Device Client `libnvds_azure_proto.so` with the following examples.
* [ode_instance_trigger_message_server.py](/examples/python/ode_instance_trigger_message_server.py)
* [message_broker_azure_device_client.py](/examples/python/message_broker_azure_device_client.py)

---

# Azure Module Client setup

## Setup the Azure IoT Edge runtime on the edge device
#### For Jetson and x86 computers running Ubuntu:
Follow the below steps from the instructions here. https://docs.microsoft.com/en-us/azure/iot-edge/how-to-install-iot-edge-linux. Specifically:
* [Install IoT Edge](https://docs.microsoft.com/en-us/azure/iot-edge/how-to-provision-single-device-linux-symmetric?view=iotedge-2020-11&tabs=azure-portal%2Cubuntu#install-iot-edge).
* **Caution:** make sure to skip the section [Install a container engine](https://docs.microsoft.com/en-us/azure/iot-edge/how-to-provision-single-device-linux-symmetric?view=iotedge-2020-11&tabs=azure-portal%2Cubuntu#install-a-container-engine).
* [Install the IoT Edge runtime](https://docs.microsoft.com/en-us/azure/iot-edge/how-to-provision-single-device-linux-symmetric?view=iotedge-2020-11&tabs=azure-portal%2Cubuntu#install-the-iot-edge-runtime).
* [Provision the device with its cloud identity](https://docs.microsoft.com/en-us/azure/iot-edge/how-to-provision-single-device-linux-symmetric?view=iotedge-2020-11&tabs=azure-portal%2Cubuntu#provision-the-device-with-its-cloud-identity)
* [Verify successful configuration](https://docs.microsoft.com/en-us/azure/iot-edge/how-to-provision-single-device-linux-symmetric?view=iotedge-2020-11&tabs=azure-portal%2Cubuntu#verify-successful-configuration)

## Build and deploy a Docker Image
See the instructions and Docker file under the [deepstream-services-library-docker](https://github.com/prominenceai/deepstream-services-library-docker) GitHub repository.

## Grant host access to the local X-server
As a privileged user (root), append the following lines to file **`/etc/profile`** to allow access to the local X-server.

```bash
if [ "$DISPLAY" != "" ]
then
  xhost +
fi
```
Make the file executable
```bash
sudo chmod u+x /etc/profile
```
Then execute the file.
```bash
sudo /etc/profile
```
You should see confirmation similar to below.
```
access control disabled, clients can connect from any host.
```
**Note:** how to correctly setup specific or limited access is still under investigation.

## Deploy IoT Modules
The following instructions detail the steps to create a new *IoT Edge Custom Module* to run the Docker Image created in the previous section [Build and deploy a Docker Image](#build-and-deploy-a-docker-image).

From your Azure portal, select `IoT Edge` from the left menu pane, then select your device by its id. You should see the two existing *IoT Edge System Modules* that were deployed when you [Setup the Azure IoT Edge runtime on the edge device](#setup-the-azure-iot-edge-runtime-on-the-edge-device).

To setup a new *IoT Edge Custom Module* to run the [message_broker_azure_module_client.py](/examples/python/message_broker_azure_module_client.py) example, select the `Set modules` from the upper menu bar on the device page, as show below.

<br>

![](/Images/azure-iot-edge-device-set-modules.png)

<br>

Select the `+ Add` button under the **IoT Edge Modules** section on the `Set modules` page and select the first item `+ IoT Edge Module` as shown below.

<br>

![](/Images/azure-iot-edge-device-create-module.png)

<br>

Define the **`Module Settings`** by specifying the `IoT Edge Module Name` and `Image URI` to the Docker Image created in the previous section [Build and deploy a Docker Image](#build-and-deploy-a-docker-image) as shown in the image below.

![](/Images/azure-iot-edge-device-create-module-settings.png)

<br>

Select the **`Environment Variables`** tab and add the `DISPLAY` variable as shown below. Set the value to `:0` for local or `:1` for remote.

![](/Images/azure-iot-edge-device-create-environment-variables.png)

<br>

Select the **`Container Create Options`** tab

![](/Images/azure-iot-edge-device-create-container-create-options.png)
   
Then add the below JSON code and select **`Add`**
```json
{
    "Entrypoint": [
        "python3",
        "/opt/prominenceai/deepstream-services-library/examples/python/message_broker_azure_module_client.py"
    ],
    "HostConfig": {
        "runtime": "nvidia",
        "NetworkMode": "host",
        "Binds": [
            "/tmp/argus_socket:/tmp/argus_socket",
            "/tmp/.X11-unix/:/tmp/.X11-unix/",
            "/tmp/.dsl/:/tmp/.dsl/"
        ],
        "IpcMode": "host"
    },
    "NetworkingConfig": {
        "EndpointsConfig": {
            "host": {}
        }
    },
    "WorkingDir": "/opt/prominenceai/deepstream-services-library/examples/python/"
}
```

<br>

Once the code has been added, select the **`Review + create`** button from the `Set modules` main page as show below:

![](/Images/azure-iot-edge-device-create-module-review-and-create.png)

Make sure that the `validation passed` - as show in the upper left corner in the image below - and then select **`Create`**

<br>

![](/Images/azure-iot-edge-device-create-module-validate-success.png)

<br>

All three IoT Edge Modules should now be in a `running` state as shown below.

<br>

![](/Images/azure-iot-edge-device-set-modules-success.png)

The example script directs the DSL INFO logs and Python application logs to files located under `/tmp/.dsl/` which is viewable from outside of the Module container.

---

## Next Steps and Useful Links

### [Troubleshoot IoT Edge devices from the Azure portal](https://docs.microsoft.com/en-us/azure/iot-edge/troubleshoot-in-portal?view=iotedge-2020-11)
Use the trouble shooting page in the Azure portal to monitor IoT Edge devices and modules.

### [Quickstart: Create an Azure SQL Database](https://docs.microsoft.com/en-us/azure/azure-sql/database/single-database-create-quickstart?tabs=azure-portal)
Create a single database in Azure SQL Database using either the Azure portal, a PowerShell script, or an Azure CLI script. You then query the database using Query editor in the Azure portal.

---

# Trouble Shooting
## Failure installing azure-cli on Jetson.
If the command to install azure-cli using pip3 fails with the following module dependency errors
```
    No package 'libffi' found
    c/_cffi_backend.c:15:10: fatal error: ffi.h: No such file or directory
     #include <ffi.h>
              ^~~~~~~
    compilation terminated.
```
Install the dev suite of `libffi` as follows:
```bash
sudo apt-get install libffi6 libffi-dev
```

If the command to install fails with the following error
```
    Traceback (most recent call last):
      File "<string>", line 1, in <module>
      File "/tmp/pip-build-jshgucrb/cryptography/setup.py", line 14, in <module>
        from setuptools_rust import RustExtension
    ModuleNotFoundError: No module named 'setuptools_rust'
   
            =============================DEBUG ASSISTANCE==========================
            If you are seeing an error here please try the following to
            successfully install cryptography:
   
            Upgrade to the latest pip and try again. This will fix errors for most
            users. See: https://pip.pypa.io/en/stable/installing/#upgrading-pip
            =============================DEBUG ASSISTANCE==========================
```
Upgrade to the latest version of `setuptools` and `pip` with the following commands
```bash
sudo apt-get install python3-setuptools
sudo pip3 install --upgrade pip
```

### Failure verifying azure-cli install on Jetson
If the command `az --version ` fails with
```
/usr/bin/az: line 2: /opt/az/bin/python3: cannot execute binary file: Exec format error

```

You have an invalid Debian version installed which can be removed with.
```bash
sudo apt-get remove -y azure-cli
```
Open a new terminal and re-verify.
