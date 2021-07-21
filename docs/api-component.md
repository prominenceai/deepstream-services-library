# Component API Reference
The Pipeline Component API provides the common services that apply to all Pipeline Component types.
* [Sources](/docs/api-source.md)
* [Taps](/docs/api-tap.md)
* [Inference Engines and Servers](/docs/api-infer.md)
* [Trackers](/docs/api-tracker.md)
* [Segmentation Visualizers](/dos/api-segvisual.md)
* [Splitters and Demuxers](/docs/api-tee.md)
* [2D-Tilers](/docs/api-tiler.md)
* [On Sreen Displays](/docs/api-osd.md)
* [Sinks](/docs/api-sink.md)

##
* [dsl_component_delete](#dsl_component_delete)
* [dsl_component_delete_many](#dsl_component_delete_many)
* [dsl_component_delete_all](#dsl_component_delete_all)
* [dsl_component_list_size](#dsl_component_list_size)
* [dsl_component_gpuid_get](#dsl_component_gpuid_get)
* [dsl_component_gpuid_set](#dsl_component_gpuid_set)
* [dsl_component_gpuid_set_many](#dsl_component_gpuid_set_many)

## Return Values
The following return codes are used by the Component API
```C++
#define DSL_RESULT_SUCCESS                                          0x00000000

#define DSL_RESULT_COMPONENT_NAME_NOT_UNIQUE                        0x00010001
#define DSL_RESULT_COMPONENT_NAME_NOT_FOUND                         0x00010002
#define DSL_RESULT_COMPONENT_NAME_BAD_FORMAT                        0x00010003
#define DSL_RESULT_COMPONENT_IN_USE                                 0x00010004
#define DSL_RESULT_COMPONENT_NOT_USED_BY_PIPELINE                   0x00010005
#define DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE                   0x00010006
#define DSL_RESULT_COMPONENT_SET_GPUID_FAILED                       0x00010007
```

## Destructors
### *dsl_component_delete*
```c++
DslReturnType dsl_component_delete(const wchar_t* component);
```
This service deletes a single Pipeline Component of any type. The call will fail if the Component is currently `in-use` by a Pipeline.

**Parameters**
* `component` - [in] unique name of the component to delete.

**Returns**
* `DSL_RESULT_SUCCESS` on successful delete. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_component_delete('my-uri-source')
```

<br>

### *dsl_component_delete_many*
```c++
DslReturnType dsl_component_delete_many(const wchar_t** component);
```
This service deletes a Null terminated list of named Components of any type. The call will fail if any Component is currently `in-use` by a Pipeline.

**Parameters**
* `components` - [in] Null terminated list of unique component names to delete.

**Returns**
* `DSL_RESULT_SUCCESS` on successful delete. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_component_delete_many(['my-uri-source', 'my-primary-gie', 'my-osd', 'my-window-sink', None])
```

<br>

### *dsl_component_delete_all*
```c++
DslReturnType dsl_component_delete_all();
```
This service deletes all Components in memory. The call will fail if any Component is currently `in-use` by a Pipeline.

**Returns**
* `DSL_RESULT_SUCCESS` on successful delete. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_component_all()
```

<br>

## Methods
### *dsl_component_list_size*
```c++
uint dsl_component_list_size();
```
This service returns the current number of Components (all types) in memory. The number does not include Pipelines.

**Returns**
* The number of Components in memory

**Python Example**
```Python
number_of_components = dsl_component_list_size()
```

<br>

### *dsl_component_gpuid_get*
```c++
DslReturnType dsl_component_gpuid_get(const wchar_t* component, uint* gpuid);
```
This service returns the current GPU ID for the named Component. The default setting for all components is GPU ID = 0.

**Parameters**
* `component` - [in] unique name of the Component to query.
* `gpuid` - [out] current GPU ID in use by the Component.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval, gpuid = dsl_component_gpuid_get('my-primary-gie')
```

<br>

### *dsl_component_gpuid_set*
```c++
DslReturnType dsl_component_gpuid_set(const wchar_t* component, uint gpuid);
```
This service sets the current GPU ID for the named Component to use. The call will fail if the Component is currently `in-use` by a Pipline.

**Parameters**
* `component` - [in] unique name of the Component to query.
* `gpuid` - [in] new GPU ID to use by the Component.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval = dsl_component_gpuid_set('my-primary-gie', 1)
```

<br>

### *dsl_component_gpuid_set_many*
```c++
DslReturnType dsl_component_gpuid_set_many(const wchar_t** component, uint gpuid);
```
This service sets the GPU ID for a Null terminated list of named components. The call will fail if any Component is currently `in-use` by a Pipeline, on first exception.

**Parameters**
* `components` - [in] Null terminated list of unique Component names to delete.
* `gpuid` - [in] new GPU ID to use by all named Components.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_component_gpuid_set_many(['my-uri-source', 'my-primary-gie', 'my-osd', 'my-window-sink', None], 1)
```

<br>

---

## API Reference
* [List of all Services](/docs/api-reference-list.md)
* [Pipeline](/docs/api-pipeline.md)
* [Player](/docs/api-player.md)
* [Source](/docs/api-source.md)
* [Tap](/docs/api-tap.md)
* [Dewarper](/docs/api-dewarper.md)
* [Inference Engine and Server](/docs/api-infer.md)
* [Tracker](/docs/api-tracker.md)
* [Segmentation Visualizer](/docs/api-segvisual.md)
* [Tiler](/docs/api-tiler.md)
* [Demuxer and Splitter](/docs/api-tee.md)
* [On-Screen Display](/docs/api-osd.md)
* [Sink](/docs/api-sink.md)
* [Pad Probe Handler](/docs/api-pph.md)
* [ODE Trigger](/docs/api-ode-trigger.md)
* [ODE Acton](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* [Display Type](/docs/api-display-type.md)
* [Branch](/docs/api-branch.md)
* **Component**
* [Mailer](/docs/api-mailer.md)
