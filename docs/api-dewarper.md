# Dewarper API Reference
The Dewarper component is used to dewarp a Source component's output-buffer stream. Built on NVIDIA's Gst-nvdewarper plugin, the DSL Dewarper component supports dewarping of three projection types:
1. PushBroom
2. VertRadCyl
3. PerspectivePerspective. 

The first two are used for dewarping 360° camera input. See the [NVIDIA Gst-nvdewarper](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvdewarper.html#gst-nvdewarper) documentation for more information.

NVIDA provides two sample configuration files located under `/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-dewarper-test/`.

Dewarpers are derived from the "Component" class, therefore all [component methods](/docs/api-component.md) can be called with any Dewarper.

#### Hierarchy
[`component`](/docs/api-component.md)<br>
&emsp;╰── `dewarper`

## Construction and Destruction
A Dewarper component is created by calling [`dsl_dewarper_new`](#dsl_dewarper_new) with a type specific configuration file. 

Dewarper components are deleted by calling [`dsl_component_delete`](api-component.md#dsl_component_delete), [`dsl_component_delete_many`](api-component.md#dsl_component_delete_many), or [`dsl_component_delete_all`](api-component.md#dsl_component_delete_all). Calling a delete service on a Dewarper `in-use` by a Source will fail.

## Adding and Removing
The relationship between Sources and Dewarpers is one-to-one. Once added to a Source, a Dewarper must be removed before it can be used with another.

A Dewarper is added to a Source by calling [`dsl_source_dewarper_add`](/docs/api-source.md#dsl_source_dewarper_add) and removed with [`dsl_source_dewarper_remove`](/docs/api-source.md#dsl_source_dewarper_remove)

## Relavent Examples:
There are two simple examples that use the configuration files and sample videos provided by NVIDIA.
**Python*
* [video_dewarper_360.py](/examples/python/video_dewarper_360.py)
* [video_dewarper_perspective.py](/examples/python/video_dewarper_perspective.py)

## Dewarper API
**Constructors:**
* [`dsl_dewarper_new`](#dsl_dewarper_new)

**Methods:**
* [`dsl_dewarper_config_file_get`](#dsl_dewarper_config_file_get)
* [`dsl_dewarper_config_file_set`](#dsl_dewarper_config_file_set)
* [`dsl_dewarper_camera_id_get`](#dsl_dewarper_camera_id_get)
* [`dsl_dewarper_camera_id_set`](#dsl_dewarper_camera_id_set)
* [`dsl_dewarper_num_batch_buffers_get`](#dsl_dewarper_num_batch_buffers_get)
* [`dsl_dewarper_num_batch_buffers_set`](#dsl_dewarper_num_batch_buffers_set)

<br>

## Return Values
The following return codes are used specifically by the Dewarper API
```C
#define DSL_RESULT_DEWARPER_RESULT                                  0x00090000
#define DSL_RESULT_DEWARPER_NAME_NOT_UNIQUE                         0x00090001
#define DSL_RESULT_DEWARPER_NAME_NOT_FOUND                          0x00090002
#define DSL_RESULT_DEWARPER_NAME_BAD_FORMAT                         0x00090003
#define DSL_RESULT_DEWARPER_THREW_EXCEPTION                         0x00090004
#define DSL_RESULT_DEWARPER_CONFIG_FILE_NOT_FOUND                   0x00090005
#define DSL_RESULT_DEWARPER_SET_FAILED                              0x00090006
```

## Constructors
### *dsl_dewarper_new*

```C
DslReturnType dsl_dewarper_new(const wchar_t* name, 
    const wchar_t* config_file, uint camera_id);
```
This service creates a uniquely named Dewarper. Construction will fail if the name is currently in use. 

#### Hierarchy
[`component`](/docs/api-component.md)<br>
&emsp;╰── `dewarper`

**Parameters**
* `name` - [in] unique name for the Dewarper component to create.
* `config_file` - [in] relative or absolute pathspec to a valid config text file.
* `camera_id` - [in] index into the first column of the CSV files (i.e. csv_files/nvaisle_2M.csv & csv_files/nvspot_2M.csv). The dewarping parameters for the given camera are read from CSV files and are used to generate dewarp surfaces (i.e. multiple aisle and spot surface) from a 360d input video stream.

**Note:** The `camera_id` parameter is NOT used if the config file specifies `projection-type = 3` (3 = PerspectivePerspective). In this case, all dewarping parameters must be defined in the config-file.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_dewarper_new('my-dewarper', path_to_config_file, 6)
```

<br>

## Methods
### *dsl_dewarper_config_file_get*
```C
DslReturnType dsl_dewarper_config_file_get(const wchar_t* name, 
    const wchar_t** config_file);
```
This service returns the path to the config-file in use by the named Dewarper.

**Parameters**
* `name` - [in] unique name of the Dewarper to query.
* `config_file` - [out] path specification to the config-file in use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, config_file = dsl_dewarper_config_file_get('my-dewarper')
```

<br>

### *dsl_dewarper_config_file_set*
```C
DslReturnType dsl_dewarper_config_file_set(const wchar_t* name, 
    const wchar_t* config_file);
```
This service updates the named Dewarper component with a new config-file to use.

**Parameters**
* `name` - [in] unique name of Dewarper to update.
* `config_file` - [in] absolute or relative path specification to the new confif-file to use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_dewarper_config_file_set('my-dewarper', path_to_config_file)
```

<br>

### *dsl_dewarper_camera_id_get*
```C
DslReturnType dsl_dewarper_camera_id_get(const wchar_t* name, uint* camera_id);
```
This service returns the `camera_id` in use by the named Dewarper component. The camera-id is the index into the first column of the CSV files (i.e. csv_files/nvaisle_2M.csv & csv_files/nvspot_2M.csv). The dewarping parameters for the given camera are read from CSV files and are used to generate dewarp surfaces (i.e. multiple aisle and spot surface) from a 360d input video stream.

**Note:** The `camera_id` parameter is NOT used if the config file specifies `projection-type = 3` (3 = PerspectivePerspective). In this case, all dewarping parameters must be defined in the config-file.

**Parameters**
* `name` - [in] unique name of the Dewarper to query.
* `camera_id` - [out] current camera_id in use. 

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, camera_id = dsl_dewarper_camera_id_get('my-dewarper')
```

<br>

### *dsl_dewarper_camera_id_set*
```C
DslReturnType dsl_dewarper_camera_id_set(const wchar_t* name, uint camera_id);
```
This service updates the named Dewarper component with a new camera-id to use. The camera-id is the index into the first column of the CSV files (i.e. csv_files/nvaisle_2M.csv & csv_files/nvspot_2M.csv). The dewarping parameters for the given camera are read from CSV files and are used to generate dewarp surfaces (i.e. multiple aisle and spot surface) from a 360d input video stream.

**Note:** The `camera_id` parameter is NOT used if the config file specifies `projection-type = 3` (3 = PerspectivePerspective). In this case, all dewarping parameters must be defined in the config-file.

**Parameters**
* `name` - [in] unique name of the Dewarper to update.
* `camera_id` - [in] new camera-id for the Dewarper to use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_dewarper_camera_id_set('my-dewarper', 4)
```

<br>

### *dsl_dewarper_num_batch_buffers_get*
```C
DslReturnType dsl_dewarper_num_batch_buffers_get(const wchar_t* name, uint* num);
```
This service returns the num-batch-buffers setting in use by the named Dewarper. Refers to the number of dewapred output surfaces per frame buffer, i.e. the batch-size of the buffer.

**Note:** this property can be defined in the configuration file.

**Parameters**
* `name` - [in] unique name of the Dewarper to query.
* `num` - [out] number of dewarped output surfaces per buffer [1..4] 

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, num = dsl_dewarper_num_batch_buffers_get('my-dewarper')
```

<br>

### *dsl_dewarper_num_batch_buffers_set*
```C
DslReturnType dsl_dewarper_num_batch_buffers_set(const wchar_t* name, uint num);
```
This service updates the named Dewarper's num-batch-buffers setting. Refers to the number of dewapred output surfaces per frame buffer, i.e. the batch-size of the buffer.
**Note:** this property can be defined in the configuration file.

**Parameters**
* `name` - [in] unique name of the Dewarper to update.
* `num` - [in] new num-batch-buffers value to use [1..4].

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_dewarper_num_batch_buffers_set('my-dewarper', 4)
```

<br>

---

## API Reference
* [List of all Services](/docs/api-reference-list.md)
* [Pipeline](/docs/api-pipeline.md)
* [Player](/docs/api-player.md)
* [Source](/docs/api-source.md)
* [Tap](/docs/api-tap.md)
* **Dewarper**
* [Preprocessor](/docs/api-preproc.md)
* [Inference Engine and Server](/docs/api-infer.md)
* [Tracker](/docs/api-tracker.md)
* [Segmentation Visualizer](/docs/api-segvisual.md)
* [Tiler](/docs/api-tiler.md)
* [Demuxer, Remxer, and Splitter Tees](/docs/api-tee.md)
* [On-Screen Display](/docs/api-osd.md)
* [Sink](/docs/api-sink.md)
* [Pad Probe Handler](/docs/api-pph.md)
* [ODE Trigger](/docs/api-ode-trigger.md)
* [ODE Accumulator](/docs/api-ode-accumulator.md)
* [ODE Acton](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* [ODE Heat-Mapper](/docs/api-ode-heat-mapper.md)
* [Display Type](/docs/api-display-type.md)
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
* [Mailer](/docs/api-mailer.md)
* [WebSocket Server](/docs/api-ws-server.md)
* [Message Broker](/docs/api-msg-broker.md)
* [Info API](/docs/api-info.md)
