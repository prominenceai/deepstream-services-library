# Azure MQTT Protocol Adapter Libraries
NVIDIA provides two protocol libraries installed with DeepStream under `/opt/nvidia/deepstream/deepstream/sources/libs`
* `libnvds_azure_proto.so` - a device client protocol for sending messages from the device to an Azure IoT Hub instance. Requires minimal setup.
* `libnvds_azure_edge_proto.so` - a module client protocol for bidirectional device-server messaging. Applications linking with DSL must by run in a Docker container. The Instructions below depend on Visual Studio Code for much of the module client setup.

## Common Setup for both Protocol Adapters
### Setup an Azure IoT Hub Instance
Follow the directions at https://docs.microsoft.com/en-us/azure/iot-hub/tutorial-connectivity#create-an-iot-hub.

### Install Additional device dependencies
#### For an x86 computer running Ubuntu:
```
sudo apt-get install -y libcurl3 libssl-dev uuid-dev libglib2.0 libglib2.0-dev
```
#### For Jetson:
```
sudo apt-get install -y libcurl4-openssl-dev libssl-dev uuid-dev libglib2.0 libglib2.0-dev
```

## Azure Module Client setup
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

### Install the Azure CLI
#### For an x86 computer running Ubuntu:
The Azure CLI team maintains a script to run all installation commands in one step. This script is downloaded via curl and piped directly to bash to install the CLI.
```bash
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

#### For Jetson:
Currently the only way to use Azure CLI on ARM64 is to install from PyPI (https://pypi.org/project/azure-cli/):
```
pip3 install azure-cli
```

### Verify Azure CLI Install
You can verify your installation with.
```bash
az --version
```

If the command fails with 
```
/usr/bin/az: line 2: /opt/az/bin/python3: cannot execute binary file: Exec format error

```

Remove the (invalid) debian installed version
```bash
sudo apt-get remove -y azure-cli
```
Open a new terminal and re-verify

### Login to your Azure subscription
The following command will bring up the login screen in your browser.
```bash
az login
```

### Add the azure-iot extension.
``` bash
az extension add --name azure-iot
```

### Deploy IoT Modules
Clone the `azure-deployment-config` repository to your device.
```bash
git clone https://github.com/prominenceai/azure-deployment-config
```
Files:
* `deployment.template.json` - template file from which the deployment config is created.
* `.env` - environment file for your credentials
Update the `.env` file and save.

Generate the deployment manifest from the deployment.template.json.
```bash
iotedgedev genconfig -f deployment.template.json -P arm64v8
```

### Build and deploy a Docker Image
See the instructions and Docker file under the [deepstream-services-library-docker](https://github.com/prominenceai/deepstream-services-library-docker) GitHub repository.


