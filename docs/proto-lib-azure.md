# Azure MQTT Protocol Adapter Libraries
NVIDIA provides two protocol libraries installed with DeepStream under `/opt/nvidia/deepstream/deepstream/sources/libs`
* `libnvds_azure_proto.so` - a device client protocol for sending messages from the device to an Azure IoT Hub instance. Requires minimal setup.
* `libnvds_azure_edge_proto.so` - a module client protocol for bidirectional device-server messaging. Applications linking with DSL must be run in a deployed Docker container. 

## Common Setup for both Protocol Adapters
### Install Additional device dependencies
#### For an x86 computer running Ubuntu:
```
sudo apt-get install -y libcurl3 libssl-dev uuid-dev libglib2.0 libglib2.0-dev
```
#### For Jetson:
```
sudo apt-get install -y libcurl4-openssl-dev libssl-dev uuid-dev libglib2.0 libglib2.0-dev
```

### Setup an Azure IoT Hub Instance
Follow the directions at https://docs.microsoft.com/en-us/azure/iot-hub/tutorial-connectivity#create-an-iot-hub.

### Install the Azure CLI on your edge device
#### For an x86 computer running Ubuntu:
The Azure CLI team maintains a script to run all installation commands in one step. This script is downloaded via curl and piped directly to bash to install the CLI.
```bash
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

#### For Jetson:
Currently, the only way to use Azure CLI on ARM64 is to install from PyPI (https://pypi.org/project/azure-cli/):
```
pip3 install azure-cli
```
If the install fails see [Failure installing azure-cli on Jetson](#failure-installing-azure-cli-on-jetson) under [Trouble Shooting](#trouble-shooting).

Once installation is complete you will need to reboot the device
```bash
sudo reboot
```
Verify the installation with.
```bash
az --version
```
If the verification command fails see [Failure varifying azure-cli install on Jetson](#failure-varifying-azure-cli-intall-on-jetson) under [Trouble Shooting](#trouble-shooting).

### Add the azure-iot extension.
``` bash
az extension add --name azure-iot
```

### Create an IoT Edge Device
Log into to your Azure IoT Hub on the cloud from your device terminal window. The following command will bring up the login screen in your browser.
```bash
az login
```

Create an IoT edge device from you device terminal with the following command.
```bash
az iot hub device-identity create --device-id <device-id> --hub-name <hub-name> --edge-enabled
```
where 
* `<device-id>` =  name (string) to identify the new device
* `<hub-name>` = the hub-name you used when you [Setup an Azure IoT Hub Instance](setup_an_azure_iot_hub_instance) above.

Verify the device creation with the following command. 
```bash
az iot hub device-identity list --hub-name <hub-name>
```
Or check from your Azure IoT Hub instance on the cloud. From your hub dashboard, select the **`IoT Edge`** item in the left menu pane, you should then seem your device listed in the main window.

Get the connection string for your new device
```bash
az iot hub device-identity connection-string show --device-id <device-id> --hub-name <hub-name>
```
You will need the connection-string to use the [Message Sink](/docs/api-sink#dsl_sink_message_new) and [Message Broker API](/docs/api-msg-broker.md).
Your device setup is now sufficient to use the Device Client `libnvds_azure_proto.so` with the following examples.
* [ode_instance_trigger_message_server.py](/examples/python/ode_instance_trigger_message_server.py)
* [message_broker_azure_device_client.py](/examples/python/message_broker_azure_device_client.py)

## Azure Module Client setup
***Still a work in progress (WIP)***

### Setup Azure IoT Edge runtime on the edge device
#### For an x86 computer running Ubuntu:
Follow the instructions here. https://docs.microsoft.com/en-us/azure/iot-edge/how-to-install-iot-edge-linux

#### For Jetson:
Enter the following commands.
```bash
sudo apt-get -y install libffi-dev jq python-pip
pip3 install iotedgedev
sudo mv ~/.local/bin/iotedgedev /usr/local/bin
```
Install curl 
```bash
sudo apt update
sudo apt install curl
```
Download and install the standard libiothsm implementation
```bash
curl -L https://github.com/Azure/azure-iotedge/releases/download/1.0.8-rc1/libiothsm-std_1.0.8.rc1-1_arm64.deb -o libiothsm-std.deb && sudo dpkg -i ./libiothsm-std.deb
```
Download and install the IoT Edge Security Daemon
```bash
curl -L https://github.com/Azure/azure-iotedge/releases/download/1.0.8-rc1/iotedge_1.0.8.rc1-1_arm64.deb -o iotedge.deb && sudo dpkg -i ./iotedge.deb
```
Run apt-get fix
```bash
sudo apt-get install -f
```

Update the `/etc/iotedge/config.yaml` file.

Find the `provisioning` section of the file and uncomment the manual provisioning mode. Update the value of device_connection_string with the connection string from your IoT Edge device.
```yaml
provisioning:
  source: "manual"
  device_connection_string: "<ADD DEVICE CONNECTION STRING HERE>"
```
Update the default IoT Edge `agent` configuration to pull the 1.0.8-rc1 version of the agent.
```yaml
agent:
  name: "edgeAgent"
  type: "docker"
  env: {}
  config:
    image: "mcr.microsoft.com/azureiotedge-agent:1.0.8-rc1"
    auth: {}
```

Restart the IoT Edge service
```bash
service iotedge restart
```

### Build and deploy a Docker Image
See the instructions and Docker file under the [deepstream-services-library-docker](https://github.com/prominenceai/deepstream-services-library-docker) GitHub repository.

### Deploy IoT Modules
There are two IoT Edge System Modules that must be deployed with every edge device. The modules can be pulled to the edge device using the Azure CLI and IoT extension. 

Clone the `azure-deployment-config` repository to your device.
```bash
git clone https://github.com/prominenceai/azure-deployment-config
```
Files:
* `deployment.template.json` - template file from which the deployment config is created.
* `.env` - environment file for your credentials
Update the `.env` file with your credentials and the address of your local Docker registry you created in the [Build and deploy a Docker Image] 
```yaml
CONTAINER_REGISTRY_USERNAME = "my-username"
CONTAINER_REGISTRY_PASSWORD = "my-password"
CONTAINER_REGISTRY_ADDRESS = "http://localhost:5000"
```

Generate the deployment manifest from the deployment.template.json with the following command.
```bash
iotedgedev genconfig -f deployment.template.json -P arm64v8
```
You should see the following confirmation
```
=======================================
======== ENVIRONMENT VARIABLES ========
=======================================

Environment Variables loaded from: .env (/home/prominenceai1/prominenceai/azure-deployment-config/.env)
Expanding image placeholders
Converting createOptions
Deleting template schema version
Expanding 'deployment.template.json' to 'config/deployment.arm64v8.json'
Validating generated deployment manifest config/deployment.arm64v8.json
Validating schema of deployment manifest.
Deployment manifest schema validation passed.
Start validating createOptions for all modules.
Validating createOptions for module edgeAgent
createOptions of module edgeAgent validation passed
Validating createOptions for module edgeHub
createOptions of module edgeHub validation passed
Validation for all createOptions passed.
```
Note the config file path above `config/deployment.arm64v8.json`

Then, deploy the modules with the following command using the config path as shown.
```bash
az iot edge set-modules --device-id <device id> --hub-name <hub name> --content ./config/deployment.arm64v8.json
```
where 
* `<device-id>` =  the device-id you used when you [Create an IoT Edge Device Identity](create_an_iot_edge_device_identity) above.
* `<hub-name>` = the hub-name you used when you [Setup an Azure IoT Hub Instance](setup_an_azure_iot_hub_instance) above.

Verify the module deployment with the following command.
```bash
iotedge list
```

## Trouble Shooting
### Failure installing azure-cli on Jetson.
If the command to intall azure-cli using pip3 fails with the following module dependency errors
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
Upgrade to the latest version of `setuptools` with 
```bash
sudo apt-get install python3-setuptools
```

### Failure varifying azure-cli install on Jetson
If the command `az --version ` fails with 
```
/usr/bin/az: line 2: /opt/az/bin/python3: cannot execute binary file: Exec format error

```

You have an invalid debian version installed which can be removed with.
```bash
sudo apt-get remove -y azure-cli
```
Open a new terminal and re-verify.
