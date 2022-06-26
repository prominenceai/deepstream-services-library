# Segmentation Visualizer API Reference
The Segmentation Visualizer visualizes segmentation results produced from either a Primary Gst Inference Engine (PGIE) or Primary Triton Inference Server (TIS).  

From Nivida's [Gst-nvsegvisual](https://docs.nvidia.com/metropolis/deepstream/5.0DP/plugin-manual/DeepStream%20Plugins%20Development%20Guide/deepstream_plugin_details.3.11.html#) plugin documentation. *"Segmentation is based on image recognition, except that the classifications occur at the pixel level as opposed to the image level as with image recognition."*
Please refer to the plugin manual for more information.

#### Segmentation Visualizer Construction and Destruction
Segmentation Visualizers are created by calling the constructor [dsl_segvisual_new](#dsl_segvisual_new). Segmentation Visualizers are deleted by calling [dsl_component_delete](api-component.md#dsl_component_delete), [dsl_component_delete_many](api-component.md#dsl_component_delete_many), or [dsl_component_delete_all](api-component.md#dsl_component_delete_all)

#### Adding to a Pipeline
The relationship between Pipelines/Branches and Segmentation Visualizers is one-to-one. Once added to a Pipeline or Branch, a Segmentation Visualizer must be removed before it can used with another.

Segmentation Visualizers are added to a Pipeline by calling [dsl_pipeline_component_add](api-pipeline.md#dsl_pipeline_component_add) or [dsl_pipeline_component_add_many](api-pipeline.md#dsl_pipeline_component_add_many) (when adding with other components) and removed with [dsl_pipeline_component_remove](api-pipeline.md#dsl_pipeline_component_remove), [dsl_pipeline_component_remove_many](api-pipeline.md#dsl_pipeline_component_remove_many), or [dsl_pipeline_component_remove_all](api-pipeline.md#dsl_pipeline_component_remove_all).

#### Adding to a Branch
Segmentation Visualizers are added to Branches by calling [dsl_branch_component_add](api-branch.md#dsl_branch_component_add) or [dsl_branch_component_add_many](api-branch.md#dsl_branch_component_add_many) (when adding with other components) and removed with [dsl_branch_component_remove](api-branch.md#dsl_branch_component_remove), [dsl_branch_component_remove_many](api-branch.md#dsl_branch_component_remove_many), or [dsl_branch_component_remove_all](api-branch.md#dsl_branch_component_remove_all).

#### Adding/Removing Pad-Probe-handlers
Multiple Source [Pad-Probe Handlers](/docs/api-pph/md) can be added to a Segmentation Visualizer by calling [dsl_segvisual_pph_add](#dsl_segvisual_pph_add) and removed with [dsl_segvisual_pph_remove](#dsl_segvisual_pph_remove).

## Relevant Examples
* [industrial_segmentation.py](/examples/python/industrial_segmentation.py)
* [semantic_segmentation.py](/examples/python/semantic_segmentation.py)

## Segmentation Visualizer API
**Constructors**
* [dsl_segvisual_new](#dsl_segvisual_new)

**Methods**
* [dsl_segvisual_dimensions_get](#dsl_segvisual_dimensions_get)
* [dsl_segvisual_dimensions_set](#dsl_segvisual_dimensions_set)
* [dsl_segvisual_pph_add](#dsl_segvisual_pph_add).
* [dsl_segvisual_pph_remove](#dsl_segvisual_pph_remove).

## Return Values
The following return codes are used by the Segmentation Visualizer API
```C
#define DSL_RESULT_SEGVISUAL_RESULT                                 0x00600000
#define DSL_RESULT_SEGVISUAL_NAME_NOT_UNIQUE                        0x00600001
#define DSL_RESULT_SEGVISUAL_NAME_NOT_FOUND                         0x00600002
#define DSL_RESULT_SEGVISUAL_THREW_EXCEPTION                        0x00600003
#define DSL_RESULT_SEGVISUAL_IN_USE                                 0x00600004
#define DSL_RESULT_SEGVISUAL_SET_FAILED                             0x00600005
#define DSL_RESULT_SEGVISUAL_PARAMETER_INVALID                      0x00600006
#define DSL_RESULT_SEGVISUAL_HANDLER_ADD_FAILED                     0x00600007
#define DSL_RESULT_SEGVISUAL_HANDLER_REMOVE_FAILED                  0x00600008
```

## Constructors
### *dsl_segvisual_new*
```C++
DslReturnType dsl_segvisual_new(const wchar_t* name, uint width, uint height);
```
The constructor creates a uniquely named Segmentation Visualizer with given output dimensions. Construction will fail if the name is currently in use. The Segmentation Visualizer requires either a [Primary Gst Inference Engine (GIE)](/docs/api-infer.md#dsl_infer_primary_gie_new) or [Primary Triton Inference Server (TIS)](/docs/api-infer.md#dsl_infer_primary_gie_new) configured with an industrial or semantic segmentation configuration file.

**Parameters**
* `name` - [in] unique name for the Segmentation Visualizer to create.
* `width` - [in] output width of the Visualizer in pixels
* `height` - [in] output height of the Visualizer in pixels

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_segvisual_new('my-segvisual', 512, 512)
```

<br>

## Methods
### *dsl_segvisual_dimensions_get*
```C++
DslReturnType dsl_segvisual_dimensions_get(const wchar_t* name,
    uint* width, uint* height);
```
This service returns the current width and height of the named Segmentation Visualizer.

**Parameters**
* `name` - [in] unique name for the Segmentation Visualizer to query.
* `width` - [out] width of the Visualizer in pixels.
* `height` - [out] height of the Visualizer in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, width, height = dsl_segvisual_dimensions_get('my-segvisual')
```

<br>

### *dsl_segvisual_dimensions_set*
```C++
DslReturnType dsl_segvisual_dimensions_set(const wchar_t* name,
    uint width, uint height);
```
This service sets the width and height of the named Segmentation Visualizer.

**Parameters**
* `name` - [in] unique name for the Segmentation Visualizer to update.
* `width` - [in] new output width for the Visualizer in pixels.
* `height` - [in] new output height for the Visualizer in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_segvisual_dimensions_set('my-segvisual', 512, 512)
```

<br>

### *dsl_segvisual_pph_add*
```C++
DslReturnType dsl_segvisual_pph_add(const wchar_t* name, const wchar_t* handler);
```
This service adds a [Pad Probe Handler](/docs/api-pph.md) to the Source pad (only) of the named Segmentation Visualizer.

**Parameters**
* `name` - [in] unique name of the Segmentation Visualizer to update.
* `handler` - [in] unique name of Pad Probe Handler to add

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**

```Python
retval = dsl_segvisual_pph_add('my-segvisual', 'my-pph-handler')
```

<br>

### *dsl_segvisual_pph_remove*
```C++
DslReturnType dsl_segvisual_pph_remove(const wchar_t* name, const wchar_t* handler);
```
This service removes a [Pad Probe Handler](/docs/api-pph.md) from the Source pad of the named Segmentation Visualizer. The service will fail if the named handler is not owned by the Segmentation Visualizer

**Parameters**
* `name` - [in] unique name of the Segmentation Visualizer to update.
* `handler` - [in] unique name of Pad Probe Handler to remove

**Returns**
* `DSL_RESULT_SUCCESS` on successful remove. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_segvisual_pph_remove('my-segvisual', 'my-pph-handler')
```

<br>

---

## API Reference
* [List of all Services](/docs/api-reference-list.md)
* [Pipeline](/docs/api-pipeline.md)
* [Source](/docs/api-source.md)
* [Tap](/docs/api-tap.md)
* [Dewarper](/docs/api-dewarper.md)
* [Preprocessor](/docs/api-preproc.md)
* [Inference Engine and Server](/docs/api-infer.md)
* [Tracker](/docs/api-tracker.md)
* **Segmentation Visualizer**
* [Tiler](/docs/api-tiler.md)
* [On-Screen Display](/docs/api-osd.md)
* [Demuxer and Splitter](/docs/api-tee.md)
* [Sink](/docs/api-sink.md)
* [Pad Probe Handler](/docs/api-pph.md)
* [ODE Trigger](/docs/api-ode-trigger.md)
* [ODE Accumulator](/docs/api-ode-accumulator.md)
* [ODE Acton](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* [ODE Heat-Mapper](/docs/api-ode-heat-mapper.md)
* [Display Type](/docs/api-display-types.md)
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
* [Mailer](/docs/api-mailer.md)
* [WebSocket Server](/docs/api-ws-server.md)
* [Message Broker](/docs/api-msg-broker.md)
* [Info API](/docs/api-info.md)
