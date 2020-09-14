# Pipeline API Reference
Pipelines are the top level component in DSL. They manage and synchronize Child components when transitioning to states of `ready`, `paused`, and `playing`. There is no practical limit to the number of Pipelines that can be created, only the number of Sources, Secondary GIE's and Sinks in-use by one or more Pipelines at any one time; counts constrained by the Jetson Hardware in use.

#### Pipeline Construction and Destruction
Pipelines are constructed by calling [dsl_pipeline_new](#dsl_pipeline_new) or [dsl_pipeline_new_many](#dsl_pipeline_new_many). The current number of Pipelines in memory can be obtained by calling [dsl_pipeline_list_size](#dsl_pipeline_list_size).

Pipelines are destructed by calling [dsl_pipeline_delete](#dsl_pipeline_delete), [dsl_pipeline_delete_many](#dsl_pipeline_delete_many), or [dsl_pipeline_delete_all](#dsl_pipeline_delete_all). Deleting a pipeline will not delete its child component, but will unlink then and return to a state of `not-in-use`. The client application is responsible for deleting all child components by calling [dsl_component_delete](/docs/api-component.md#dsl_component_delete), [dsl_component_delete_many](/docs/api-component.md#dsl_component_delete_many), or [dsl_component_delete_all](/docs/api-component.md#dsl_component_delete_all).

#### Adding and Removing Components
Child components -- Sources, Inference Engines, Trackers, Tiled-Displays, On Screen-Display, and Sinks -- are added to a Pipeline by calling [dsl_pipeline_component_add](#dsl_pipeline_component_add) and [dsl_pipeline_component_add_many](#dsl_pipeline_component_add_many). A Pipeline's current number of child components can be obtained by calling [dsl_pipeline_component_list_size](#dsl_pipeline_component_list_size)

Child components can be removed from their Parent Pipeline by calling [dsl_pipeline_component_remove](#dsl_pipeline_componet_remove), [dsl_pipeline_component_remove_many](#dsl_pipeline_componet_remove_many), and [dsl_pipeline_component_remove_all](#dsl_pipeline_component_remove_all)
#### Playing, Pausing and Stopping a Pipeline

Pipelines - with a minimum required set of components - can be `played` by calling [dsl_pipeline_play](#dsl_pipeline_play), `paused` by calling [dsl_pipeline_pause](#dsl_pipeline_pause) and `stopped` by calling [dsl_pipeline_stop](#dsl_pipeline_stop).

#### Pipeline Client-Listener Notifications
Clients can be notified of Pipeline events by registering/deregistering one or more callback functions with the following services.
* Change of State `(COS)` events -[dsl_pipeline_state_change_listener_add](#dsl_pipeline_state_change_listener_add) / [dsl_pipeline_state_change_listener_remove](#dsl_pipeline_state_change_listener_remove). 
* End of Stream `(EOS)` events - with [dsl_pipeline_eos_listener_add](#dsl_pipeline_eos_listener_add) / [dsl_pipeline_eos_listener_remove](#dsl_pipeline_eos_listener_remove).
* Quality of Service `(QOS)` events - with [dsl_pipeline_qos_listener_add](#dsl_pipeline_qos_listener_add) / [dsl_pipeline_qos_listener_remove](#dsl_pipeline_qos_listener_remove).

#### Pipeline XWindow Support
Pipelines - that have at least one Window-Sink - will create an XWindow by default, unless one is provided. Clients can obtain a handle to this window by calling [dsl_pipeline_xwindow_handle_get](#dsl_pipeline_xwindow_handle_get). The Client can provide the Pipeline with the XWindow handle to use by calling [dsl_pipeline_xwindow_handle_set](#dsl_pipeline_display_xwindow_handle_set). A multi-Pipeline Application can have one Pipeline create the XWindow and then sharing with others, all with Window Sinks using difference offsets within the XWindow.

In the case that the Pipeline creates the XWindow, Clients can be notified of XWindow `KeyRelease` events by registering one or more callback functions with [dsl_pipeline_xwindow_key_event_handler_add](#dsl_pipeline_xwindow_key_event_handler_add). Notifications are stopped by calling [dsl_pipeline_xwindow_key_event_handler_remove](#dsl_pipeline_xwindow_key_event_handler_remove). Notifications of XWindow `ButtonPress` events can be enabled and stopped by calling [dsl_pipeline_xwindow_button_event_handler_add](#dsl_pipeline_xwindow_button_event_handler_add) and [dsl_pipeline_xwindow_button_event_handler_remove](#dsl_pipeline_xwindow_button_event_handler_remove) respectively.

---
## Pipeline API
**Client CallBack Typdefs**
* [dsl_state_change_listener_cb](#dsl_state_change_listener_cb)
* [dsl_eos_listener_cb](#dsl_eos_listener_cb)
* [dsl_qos_listener_cb](#dsl_qos_listener_cb)
* [dsl_xwindow_key_event_handler_cb](#dsl_xwindow_key_event_handler_cb)
* [dsl_xwindow_button_event_handler_cb](#dsl_xwindow_button_event_handler_cb)
* [dsl_xwindow_delete_event_handler_cb](#dsl_xwindow_delete_event_handler_cb)

**Constructors**
* [dsl_pipeline_new](#dsl_pipeline_new)
* [dsl_pipeline_new_many](#dsl_pipeline_new_many)
* [dsl_pipeline_new_component_add_many](#dsl_pipeline_new_component_add_many)

**Destructors**
* [dsl_pipeline_delete](#dsl_pipeline_delete)
* [dsl_pipeline_delete_many](#dsl_pipeline_delete_many)
* [dsl_pipeline_delete_all](#dsl_pipeline_delete_all)

**Methods**
* [dsl_pipeline_component_add](#dsl_pipeline_component_add)
* [dsl_pipeline_component_add_many](#dsl_pipeline_component_add_many)
* [dsl_pipeline_component_list_size](#dsl_pipeline_component_list_size)
* [dsl_pipeline_component_remove](#dsl_pipeline_component_remove)
* [dsl_pipeline_component_remove_many](#dsl_pipeline_component_remove_many)
* [dsl_pipeline_component_remove_all](#dsl_pipeline_component_remove_all)
* [dsl_pipeline_streammux_batch_properties_get](#dsl_pipeline_streammux_batch_properties_get)
* [dsl_pipeline_streammux_dimensions_get](#dsl_pipeline_streammux_dimensions_get)
* [dsl_pipeline_streammux_dimensions_set](#dsl_pipeline_streammux_dimensions_set)
* [dsl_pipeline_xwindow_handle_get](/docs/api-pipeline.md#dsl_pipeline_xwindow_handle_get)
* [dsl_pipeline_xwindow_handle_set](/docs/api-pipeline.md#dsl_pipeline_xwindow_handle_set)
* [dsl_pipeline_xwindow_dimensions_get](#dsl_pipeline_xwindow_dimensions_get)
* [dsl_pipeline_xwindow_dimensions_set](#dsl_pipeline_xwindow_dimensions_set)
* [dsl_pipeline_xwindow_handle_get](#dsl_pipeline_xwindow_handle_get)
* [dsl_pipeline_xwindow_handle_set](#dsl_pipeline_xwindow_handle_set)
* [dsl_pipeline_xwindow_key_event_handler_add](#dsl_pipeline_xwindow_key_event_handler_add)
* [dsl_pipeline_xwindow_key_event_handler_remove](#dsl_pipeline_xwindow_key_event_handler_remove)
* [dsl_pipeline_xwindow_button_event_handler_add](#dsl_pipeline_xwindow_button_event_handler_add)
* [dsl_pipeline_xwindow_button_event_handler_remove](#dsl_pipeline_xwindow_button_event_handler_remove)
* [dsl_pipeline_xwindow_delete_event_handler_add](#dsl_pipeline_xwindow_delete_event_handler_add)
* [dsl_pipeline_xwindow_delete_event_handler_remove](#dsl_pipeline_xwindow_delete_event_handler_remove)
* [dsl_pipeline_state_get](#dsl_pipeline_state_get)
* [dsl_pipeline_state_change_listener_add](#dsl_pipeline_state_change_listener_add)
* [dsl_pipeline_state_change_listener_remove](#dsl_pipeline_state_change_listener_remove)
* [dsl_pipeline_eos_listener_add](#dsl_pipeline_eos_listener_add)
* [dsl_pipeline_eos_listener_remove](#dsl_pipeline_eos_listener_remove)
* [dsl_pipeline_qos_listener_add](#dsl_pipeline_qos_listener_add)
* [dsl_pipeline_qos_listener_remove](#dsl_pipeline_qos_listener_remove)
* [dsl_pipeline_play](#dsl_pipeline_play)
* [dsl_pipeline_pause](#dsl_pipeline_pause)
* [dsl_pipeline_stop](#dsl_pipeline_stop)
* [dsl_pipeline_list_size](#dsl_pipeline_list_size)
* [dsl_pipeline_dump_to_dot](#dsl_pipeline_dump_to_dot)
* [dsl_pipeline_dump_to_dot_with_ts](#dsl_pipeline_dump_to_dot_with_ts)

---
## Return Values
The following return codes are used by the Pipeline API
```C++
#define DSL_RESULT_PIPELINE_NAME_NOT_UNIQUE                         0x00080001
#define DSL_RESULT_PIPELINE_NAME_NOT_FOUND                          0x00080002
#define DSL_RESULT_PIPELINE_NAME_BAD_FORMAT                         0x00080003
#define DSL_RESULT_PIPELINE_STATE_PAUSED                            0x00080004
#define DSL_RESULT_PIPELINE_STATE_RUNNING                           0x00080005
#define DSL_RESULT_PIPELINE_THREW_EXCEPTION                         0x00080006
#define DSL_RESULT_PIPELINE_COMPONENT_ADD_FAILED                    0x00080007
#define DSL_RESULT_PIPELINE_COMPONENT_REMOVE_FAILED                 0x00080008
#define DSL_RESULT_PIPELINE_STREAMMUX_GET_FAILED                    0x00080009
#define DSL_RESULT_PIPELINE_STREAMMUX_SET_FAILED                    0x0008000A
#define DSL_RESULT_PIPELINE_XWINDOW_GET_FAILED                      0x0008000B
#define DSL_RESULT_PIPELINE_XWINDOW_SET_FAILED                      0x0008000C
#define DSL_RESULT_PIPELINE_CALLBACK_ADD_FAILED                     0x0008000D
#define DSL_RESULT_PIPELINE_CALLBACK_REMOVE_FAILED                  0x0008000E
#define DSL_RESULT_PIPELINE_FAILED_TO_PLAY                          0x0008000F
#define DSL_RESULT_PIPELINE_FAILED_TO_PAUSE                         0x00080010
#define DSL_RESULT_PIPELINE_FAILED_TO_STOP                          0x00080011
#define DSL_RESULT_PIPELINE_SOURCE_MAX_IN_USE_REACED                0x00080012
#define DSL_RESULT_PIPELINE_SINK_MAX_IN_USE_REACED                  0x00080013
```

## Pipeline States
```C++
#define DSL_STATE_NULL                                              1
#define DSL_STATE_READY                                             2
#define DSL_STATE_PAUSED                                            3
#define DSL_STATE_PLAYING                                           4
#define DSL_STATE_IN_TRANSITION                                     5
```
<br>

---

## Client Callback Typedefs
### *dsl_state_change_listener_cb*
```C++
typedef void (*dsl_state_change_listener_cb)(uint old_state, uint new_state, void* user_data);
```
Callback typedef for a client state-change listener. Functions of this type are added to a Pipeline by calling [dsl_pipeline_state_change_listener_add](#dsl_pipeline_state_change_listener_add). Once added, the function will be called on every change of Pipeline state until the client removes the listener by calling [dsl_pipeline_state_change_listener_remove](#dsl_pipeline_state_change_listener_remove).

**Parameters**
* `old_state` - [in] one of [DSL_PIPELINE_STATE](#DSL_PIPELINE_STATE) constants for the old (previous) pipeline state
* `new_state` - [in] one of [DSL_PIPELINE_STATE](#DSL_PIPELINE_STATE) constants for the new pipeline state
* `user_data` - [in] opaque pointer to client's user data, passed into the pipeline on callback add

<br>

### *dsl_eos_listener_cb*
```C++
typedef void (*dsl_eos_listener_cb)(void* user_data);
```
Callback typedef for a client EOS listener function. Functions of this type are added to a Pipeline by calling [dsl_pipeline_eos_listener_add](#dsl_pipeline_eos_listener_add). Once added, the function will be called on the event of Pipeline end-of-stream (EOS). The listener function is removed by calling [dsl_pipeline_eos_listener_remove](#dsl_pipeline_eos_listener_remove) . 

**Parameters**
* `user_data` - [in] opaque pointer to client's user data, passed into the pipeline on callback add

<br>

### *dsl_qos_listener_cb*
```C++
typedef void (*dsl_eos_listener_cb)(void* user_data);
```
Callback typedef for a client QOS listener function. Functions of this type are added to a Pipeline be calling [dsl_pipeline_qos_listener_add](#dsl_pipeline_eos_listener_add). Once added, the function will be called on the event that one or more of the Pipeline's components has detected a degradation in the Quality-of-Service (QOS). The listener function is removed by calling [dsl_pipeline_qos_listener_remove](#dsl_pipeline_eos_listener_remove). 

**Parameters**
* `user_data` - [in] opaque pointer to client's user data, passed into the pipeline on callback add

<br>

### *dsl_xwindow_key_event_handler_cb*
```C++
typedef void (*dsl_xwindow_key_event_handler_cb)(const wchar_t* key, void* user_data);
```
Callback typedef for a client XWindow `KeyRelease` event hander function. Functions of this type are added to a Pipeline be calling [dsl_pipeline_xwindow_key_event_handler_add](#dsl_pipeline_xwindow_key_event_handler_add). Once added, the function will be called on every XWindow `KeyRelease` event, as long as the Pipeline has at least one [Window Sink](#). The handler function is removed by calling  [dsl_pipeline_xwindow_key_event_handler_remove](#dsl_pipeline_xwindow_key_event_handler_remove).

**Parameters**
* `key` - [in] UNICODE key string for the key pressed
* `user_data` - [in] opaque pointer to client's user data, passed into the pipeline on callback add

<br>

### *dsl_xwindow_button_event_handler_cb*
```C++
typedef void (*dsl_xwindow_button_event_handler_cb)(uint button, uint xpos, uint ypos, void* user_data);
```
Callback typedef for a client XWindow `ButtonPress` event hander function. Functions of this type are added to a Pipeline be calling [dsl_pipeline_xwindow_button_event_handler_add](#dsl_pipeline_xwindow_button_event_handler_add). Once added, the function will be called on every XWindow `ButtonPress` event, as long as the Pipeline has at least one [Window Sink](#). The handler function is removed by calling [dsl_pipeline_xwindow_button_event_handler_remove](#dsl_pipeline_xwindow_button_event_handler_remove).

**Parameters**
* `button` - [in] one of [DSL_BUTTON_ID](#) indicating which mouse button was pressed
* `xpos` - [in] possitional X-offset from the XWindow's upper left corner in pixels
* `ypos` - [in] possitional Y-offset from the XWindow's upper left corner in pixels
* `user_data` - [in] opaque pointer to client's user data, passed into the pipeline on callback add

<br>

---
## Constructors
### *dsl_pipeline_new*
```C++
DslReturnType dsl_pipeline_new(const wchar_t* pipeline);
```
The constructor creates a uniquely named Pipeline. Construction will fail
if the name is currently in use.

**Parameters**
* `pipeline` - [in] unique name for the Pipeline to create.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_pipeline_new('my-pipeline')
```

<br>

### *dsl_pipeline_new_many*
```C++
DslReturnType dsl_pipeline_new_many(const wchar_t** pipelines);
```
The constructor creates multiple uniquely named Pipelines at once. All names are checked for uniqueness, with the call returning `DSL_RESULT_PIPELINE_NAME_NOT_UNIQUE` on first occurrence of a duplicate.

**Parameters**
* `pipelines` - [in] a NULL terminated array of unique names for the Pipelines to create.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation of all Pipelines. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_pipeline_new_many(['my-pipeline-a', 'my-pipeline-b', 'my-pipeline-c', None])
```

<br>

### *dsl_pipeline_new_component_add_many*
```C++
DslReturnType dsl_pipeline_new_component_add_many(const wchar_t* pipeline, const wchar_t** components);
```
Creates a new Pipeline with a given list of named Components. The service will fail if any of components are currently `in-use` by any Pipeline. All of the component's `in-use` states will be set to true on successful add. 

**Parameters**
* `pipeline` - [in] unique name for the Pipeline to create.
* `components` - [in] a NULL terminated array of uniquely named Components to add.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation and addion. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_pipeline_new_component_add_many('my-pipeline', 
    ['my-camera-source', 'my-pgie`, 'my-osd' 'my-tiled-display', 'my-sink', None])
```

<br>

---
## Destructors
### *dsl_pipeline_delete*
```C++
DslReturnType dsl_pipeline_delete(const wchar_t* pipeline);
```
This destructor deletes a single, uniquely named Pipeline. 
All components owned by the pipeline move to a state of `not-in-use`.

**Parameters**
* `pipelines` - [in] unique name for the Pipeline to delete

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_pipeline_delete('my-pipeline')
```

<br>

### *dsl_pipeline_delete_many*
```C++
DslReturnType dsl_pipeline_delete_many(const wchar_t** pipelines);
```
This destructor deletes multiple uniquely named Pipelines. Each name is checked for existence, with the function returning `DSL_RESULT_PIPELINE_NAME_NOT_FOUND` on first occurrence of failure. 
All components owned by the Pipelines move to a state of `not-in-use`

**Parameters**
* `pipelines` - [in] a NULL terminated array of uniquely named Pipelines to delete.

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_pipeline_delete_many(['my-pipeline-a', 'my-pipeline-b', 'my-pipeline-c', None])
```

<br>

### *dsl_pipeline_delete_all*
```C++
DslReturnType dsl_pipeline_delete_all();
```
This destructor deletes all Pipelines currently in memory  All components owned by the pipelines move to a state of `not-in-use`

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_pipeline_delete_all()
```

<br>

---
## Methods

### *dsl_pipeline_component_add*
```C++
DslReturnType dsl_pipeline_component_add(const wchar_t* pipeline, const wchar_t* component);
```
Adds a single named Component to a named Pipeline. The add service will fail if the component is currently `in-use` by any Pipeline. The add service will also fail if adding a `one-only` type of Component, such as a Tiled-Display, for which the Pipeline already has. The Component's `in-use` state will be set to `true` on successful add. 

If a Pipeline is in a `playing` or `paused` state, the service will attempt a dynamic update if possible, returning from the call with the Pipeline in the same state.

**Parameters**
* `pipeline` - [in] unique name for the Pipeline to update.
* `component` - [in] unique name of the Component to add.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_source_csi_new('my-camera-source', 1280, 720, 30, 1)
retval = dsl_pipeline_new('my-pipeline')

retval = dsl_pipeline_component_add('my-pipeline', 'my-camera-source')
```

<br>

### *dsl_pipeline_component_add_many*
```C++
DslReturnType dsl_pipeline_component_add_many(const wchar_t* pipeline, const wchar_t** components);
```
Adds a list of named Component to a named Pipeline. The add service will fail if any of components are currently `in-use` by any Pipeline. The add service will fail if any of the components to add are a `one-only` type of component for which the Pipeline already has. All of the component's `in-use` states will be set to true on successful add. 

If a Pipeline is in a `playing` or `paused` state, the service will attempt a dynamic update if possible, returning from the call with the Pipeline in the same state.

**Parameters**
* `pipeline` - [in] unique name for the Pipeline to update.
* `components` - [in] a NULL terminated array of uniquely named Components to add.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_csi_new('my-camera-source', 1280, 720, 30, 1)
retval = dsl_display_new('my-tiled-display', 1280, 720)
retval = dsl_pipeline_new('my-pipeline')

retval = dsl_pipeline_component_add_many('my-pipeline', ['my-camera-source', 'my-tiled-display', None])
```

<br>

### *dsl_pipeline_component_list_size*
```C++
uint dsl_pipeline_list_size(wchar_t* pipeline);
```
This method returns the size of the current list of Components `in-use` by the named Pipeline

**Parameters**
* `pipeline` - [in] unique name for the Pipeline to query.

**Returns** 
* The size of the list of Components currently in use

<br>

### *dsl_pipeline_component_remove*
```C++
DslReturnType dsl_pipeline_component_remove(const wchar_t* pipeline, const wchar_t* component);
```
Removes a single named Component from a named Pipeline. The remove service will fail if the Component is not currently `in-use` by the Pipeline. The Component's `in-use` state will be set to `false` on successful removal. 

If a Pipeline is in a `playing` or `paused` state, the service will attempt a dynamic update if possible, returning from the call with the Pipeline in the same state.

**Parameters**
* `pipeline` - [in] unique name for the Pipeline to update.
* `component` - [in] unique name of the Component to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_pipeline_component_remove('my-pipeline', 'my-camera-source')
```
<br>

### *dsl_pipeline_component_remove_many*
```C++
DslReturnType dsl_pipeline_component_remove_many(const wchar_t* pipeline, const wchar_t** components);
```
Removes a list of named components from a named Pipeline. The remove service will fail if any of components are currently `not-in-use` by the named Pipeline.  All of the removed component's `in-use` state will be set to `false` on successful removal. 

If a Pipeline is in a `playing` or `paused` state, the service will attempt a dynamic update if possible, returning from the call with the Pipeline in the same state.

**Parameters**
* `pipeline` - [in] unique name for the Pipeline to update.
* `components` - [in] a NULL terminated array of uniquely named Components to add.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_pipeline_component_remove_many('my-pipeline', ['my-camera-source', 'my-tiled-display', None])
```

<br>

### *dsl_pipeline_component_remove_all*
```C++
DslReturnType dsl_pipeline_component_add_(const wchar_t* pipeline);
```
Removes all child components from a named Pipeline. The add service will fail if any of components are currently `not-in-use` by the named Pipeline.  All of the removed component's `in-use` state will be set to `false` on successful removal. 

**Parameters**
* `pipeline` - [in] unique name for the Pipeline to update.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_pipeline_component_remove_all('my-pipeline')
```

<br>

### *dsl_pipeline_source_name_get*
```C++
DslReturnType dsl_pipeline_source_name_get(const wchar_t name uint source_id, const wchar_t** source);
```
This service returns the name of a Source component from a unqiue Source Id. The service will only return a Source that is currently `in-use` by a Pipeline in a Playing state.

**Parameters**
* `source_id` - [in] 
* `name` - [out] unique name of the Source for the given Id. Name will be equal to Null if the source id is invalid.

**Returns**
* `DSL_RESULT_SUCCESS` on successful transition. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, source = dsl_pipeline_source_name_get('my-source', 3)
```

<br>

### *dsl_pipeline_streammux_batch_properties_get*
```C++
DslReturnType dsl_pipeline_streammux_batch_properties_get(const wchar_t* pipeline, 
    uint* batch_size, uint* batch_timeout);
```
This service returns the current `batch_size` and `batch_timeout` for the named Pipeline
**Parameters**
* `pipeline` - [in] unique name for the Pipeline to query.
* `batch_size` - [out] the current batch size, set by the Pipeline according to the current number of child Source components.
* `batch_timeout` - [out] timeout in milliseconds before a batch meta push is forced. The property is set by the Pipeline relative to the child Source component with the minimum frame-rate.

**Returns**
* `DSL_RESULT_SUCCESS` on success. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, batch_size, batch_timeout = dsl_pipeline_streammux_batch_properties_get('my-pipeline')
```
<br>

### *dsl_pipeline_streammux_dimensions_get*
```C++
DslReturnType dsl_pipeline_streammux_dimensions_get(const wchar_t* pipeline, 
    uint* width, uint* height);
```
This service returns the current Stream-Muxer output dimensions for the uniquely named Pipeline. The default dimensions, defined in `DslApi.h`, are assigned during Pipeline creation. The values can be changed after creation by calling [dsl_pipeline_streammux_dimensions_set](#dsl_pipeline_streammux_dimensions_set)

**Parameters**
* `pipeline` - [in] unique name for the Pipeline to query.
* `width` - [out] width of the Stream Muxer output in pixels.
* `height` - [out] height of the Stream Muxer output in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, width, height = dsl_pipeline_streammux_dimensions_get('my-pipeline')
```
<br>

### *dsl_pipeline_streammux_dimensions_set*
```C++
DslReturnType dsl_pipeline_streammux_dimensions_set(const wchar_t* pipeline, 
    uint width, uint height);
```
This service sets the current Stream-Muxer output dimensions for the uniquely named Pipeline. The dimensions cannot be updated while the Pipeline is in a state of `passed` or `playing`.

**Parameters**
* `pipeline` - [in] unique name for the Pipeline to update.
* `width` - [in] new width for the Stream Muxer output in pixels.
* `height` - [in] new height for the Stream Muxer output in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_pipeline_streammux_dimensions_set('my-pipeline', 1280, 720)
```
<br>

### *dsl_pipeline_xwindow_handle_get*
```C++
DslReturnType dsl_pipeline_xwindow_handle_get(const wchar_t* pipeline, Window* handle);
```
This service returns the current XWindow handle in use by the named Pipeline. The handle is set to `Null` on Pipeline creation and will remain `Null` until,
1. The Pipeline creates an internal XWindow synchronized with one or more Window-Sinks on Transition to a state of playing, or
2. The Client Application passes an XWindow handle into the Pipeline by calling [dsl_pipeline_xwindow_handle_set](#dsl_pipeline_xwindow_handle_set).

**Parameters**
* `pipeline` - [in] unique name for the Pipeline to query.
* `handle` - [out] width of the XWindow in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, x_window = dsl_pipeline_xwindow_handle_get('my-pipeline')
```
<br>

### *dsl_pipeline_xwindow_handle_set*
```C++
DslReturnType dsl_pipeline_xwindow_handle_set(const wchar_t* pipeline, Window handle);
```
This service sets the the XWindow for the named Pipeline to use. This service will fail if the Pipeline has an existing XWindow handle. 

**Parameters**
* `pipeline` - [in] unique name for the Pipeline to update.
* `handle` - [in] XWindow handle to use by all Child Window-Sink components of this Pipeline.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_pipeline_xwindow_handle_set('my-pipeline', x_window)
```
<br>

### *dsl_pipeline_xwindow_dimensions_get*
```C++
DslReturnType dsl_pipeline_xwindow_dimensions_get(const wchar_t* pipeline, 
    uint* width, uint* height);
```
This service returns the current XWindow dimensions in use on XWindow creation for the uniquely named Pipeline.

**Parameters**
* `pipeline` - [in] unique name for the Pipeline to query.
* `width` - [out] width of the XWindow in pixels.
* `height` - [out] height of the XWindow output in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, width, height = dsl_pipeline_xwindow_dimensions_get('my-pipeline')
```

<br>

### *dsl_pipeline_xwindow_dimensions_set*
```C++
DslReturnType dsl_pipeline_xwindow_dimensions_set(const wchar_t* pipeline, 
    uint width, uint height);
```
This service updates the dimensions to use on XWindow creation. This service will fail if the Pipeline has an existing XWindow handle. 

**Parameters**
* `pipeline` - [in] unique name for the Pipeline to update.
* `width` - [in] new width setting to use on XWindow creation, in pixels.
* `height` - [in] new height setting to use on XWindow creation in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_pipeline_xwindow_dimensions_set('my-pipeline', 1280, 720)
```

<br>

### *dsl_pipeline_xwindow_key_event_handler_add*
```C++
DslReturnType dsl_pipeline_xwindow_key_event_handler_add(const wchar_t* pipeline, 
    dsl_xwindow_key_handler_cb handler, void* user_data);
```
This service adds a callback function of type [dsl_xwindow_key_event_handler_cb](#dsl_xwindow_key_event_handler_cb) to a
pipeline identified by it's unique name. The function will be called on every Pipeline XWindow `KeyReleased` event with Key string and the client provided `user_data`. Multiple callback functions can be registered with one Pipeline, and one callback function can be registered with multiple Pipelines.

**Note** Client XWindow Callback functions will only be called if the Pipeline creates the XWindow, which requires a minimum of one Window-Sink component.

**Parameters**
* `pipeline` - [in] unique name of the Pipeline to update.
* `handler` - [in] XWindow event handler callback function to add.
* `user_data` - [in] opaque pointer to user data returned to the handler when called back

**Returns** 
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
def key_event_handler(key_string, user_data):
    print('key pressed = ', key_string)
    
retval = dsl_pipeline_xwindow_key_event_handler_add('my-pipeline', key_event_handler, None)
```

<br>

### *dsl_pipeline_xwindow_key_event_handler_remove*
```C++
DslReturnType dsl_pipeline_xwindow_key_event_handler_remove(const char* pipeline, 
    dsl_xwindow_key_event_handler_cb handler);
```
This service removes a Client XWindow key event handler callback that was added previously with [dsl_pipeline_xwindow_key_event_handler_add](#dsl_pipeline_xwindow_key_event_handler_add)

**Parameters**
* `pipeline` - [in] unique name of the Pipeline to update
* `handler` - [in] XWindow event handler callback function to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful remove. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_pipeline_xwindow_key_event_handler_remove('my-pipeline', key_event_handler)
```

<br>

### *dsl_pipeline_xwindow_button_event_handler_add*
```C++
DslReturnType dsl_pipeline_xwindow_button_event_handler_add(const wchar_t* pipeline, 
    dsl_xwindow_button_handler_cb handler, void* user_data);
```
This service adds a callback function of type [dsl_xwindow_button_event_handler_cb](#dsl_xwindow_button_event_handler_cb) to a
pipeline identified by it's unique name. The function will be called on every Pipeline XWindow `ButtonPressed` event with Button ID, X and Y positional offsets, and the client provided `user_data`. Multiple callback functions can be registered with one Pipeline, and one callback function can be registered with multiple Pipelines.

**Note** Client XWindow Callback functions will only be called if the Pipeline has created an XWindow, which requires a minimum of one Window-Sink component.

**Parameters**
* `pipeline` - [in] unique name of the Pipeline to update.
* `handler` - [in] XWindow event handler callback function to add.
* `user_data` - [in] opaque pointer to user data returned to the handler when called back

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
def button_event_handler(button, xpos, ypos, user_data):
    print('button = ', button)
    print('xpos = ', xpos)
    print('ypos = ', ypos)
    
retval = dsl_pipeline_xwindow_button_event_handler_add('my-pipeline', button_event_handler, None)
```

<br>

### *dsl_pipeline_xwindow_button_event_handler_remove*
```C++
DslReturnType dsl_pipeline_xwindow_button_event_handler_remove(const char* pipeline, 
    dsl_xwindow_button_event_handler_cb handler);
```
This service removes a Client XWindow button event handler callback that was added previously with [dsl_pipeline_xwindow_button_event_handler_add](#dsl_pipeline_xwindow_button_event_handler_add)

**Parameters**
* `pipeline` - [in] unique name of the Pipeline to update
* `handler` - [in] XWindow event handler callback function to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful remove. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_pipeline_xwindow_button_event_handler_remove('my-pipeline', button_event_handler)
```

<br>

### *dsl_pipeline_xwindow_delete_event_handler_add*
```C++
DslReturnType dsl_pipeline_xwindow_delete_event_handler_add(const wchar_t* pipeline, 
    dsl_xwindow_delete_handler_cb handler, void* user_data);
```
This service adds a callback function of type [dsl_xwindow_delete_event_handler_cb](#dsl_xwindow_delete_event_handler_cb) to a
pipeline identified by it's unique name. The function will be called on when the XWindow is closed/deleted. Multiple callback functions can be registered with one Pipeline, and one callback function can be registered with multiple Pipelines.

**Note** Client XWindow Callback functions will only be called if the Pipeline has created an XWindow, which requires a minimum of one Window-Sink component.

**Parameters**
* `pipeline` - [in] unique name of the Pipeline to update.
* `handler` - [in] XWindow event handler callback function to add.
* `user_data` - [in] opaque pointer to user data returned to the handler when called back

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
def xwindow_delete_event_handler(client_data):
    dsl_main_loop_quit()    

retval = dsl_pipeline_xwindow_delete_event_handler_add('my-pipeline', xwindow_delete_event_handler, None)
```

<br>

### *dsl_pipeline_xwindow_delete_event_handler_remove*
```C++
DslReturnType dsl_pipeline_xwindow_delete_event_handler_remove(const char* pipeline, 
    dsl_xwindow_delete_handler_cb handler);
```
This service removes a Client XWindow delete event handler callback that was added previously with [dsl_pipeline_xwindow_delete_event_handler_add](#dsl_pipeline_xwindow_delete_event_handler_add)

**Parameters**
* `pipeline` - [in] unique name of the Pipeline to update
* `handler` - [in] XWindow event handler callback function to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful remove. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_pipeline_xwindow_delete_event_handler_remove('my-pipeline', xwindow_delete_event_handler)
```

<br>

### *dsl_pipeline_state_change_listener_add*
```C++
DslReturnType dsl_pipeline_state_change_listener_add(const wchar_t* pipeline, 
    state_change_listener_cb listener, void* user_data);
```
This service adds a callback function of type [dsl_state_change_listener_cb](#dsl_state_change_listener_cb) to a
pipeline identified by it's unique name. The function will be called on every Pipeline change-of-state with `old_state`, `new_state`, and the client provided `user_data`. Multiple callback functions can be registered with one Pipeline, and one callback function can be registered with multiple Pipelines. 

**Parameters**
* `pipeline` - [in] unique name of the Pipeline to update.
* `listener` - [in] state change listener callback function to add.
* `user_data` - [in] opaque pointer to user data returned to the listner is called back

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
def state_change_listener(old_state, new_state, user_data, client_data):
    print('old_state = ', old_state)
    print('new_state = ', new_state)
    
retval = dsl_pipeline_state_change_listener_add('my-pipeline', state_change_listener, None)
```

<br>

### *dsl_pipeline_state_change_listener_remove*
```C++
DslReturnType dsl_pipeline_state_change_listener_remove(const wchar_t* pipeline, 
    state_change_listener_cb listener);
```
This service removes a callback function of type [state_change_listener_cb](#state_change_listener_cb) from a
pipeline identified by it's unique name.

**Parameters**
* `pipeline` - [in] unique name of the Pipeline to update.
* `listener` - [in] state change listener callback function to remove.

**Returns**  
* `DSL_RESULT_SUCCESS` on successful removal. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_pipeline_state_change_listener_remove('my-pipeline', state_change_listener)
```

<br>

### *dsl_pipeline_eos_listener_add*
```C++
DslReturnType dsl_pipeline_eos_listener_add(const wchar_t* pipeline, 
    eos_listener_cb listener, void* user_data);
```
This service adds a callback function of type [dsl_eos_listener_cb](#dsl_eos_listener_cb) to a pipeline identified by it's unique name. The function will be called on a Pipeline `EOS` event. Multiple callback functions can be registered with one Pipeline, and one callback function can be registered with multiple Pipelines.

**Parameters**
* `pipeline` - [in] unique name of the Pipeline to update.
* `listener` - [in] state change listener callback function to add.
* `user_data` - [in] opaque pointer to user data returned to the listener is called back

**Returns**  `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
def eos_listener(user_data):
    print('EOS event received')
    
retval = dsl_pipeline_eos_listener_add('my-pipeline', eos_listener, None)
```

<br>

### *dsl_pipeline_eos_listener_remove*
```C++
DslReturnType dsl_pipeline_eos_listener_remove(const wchar_t* pipeline, 
    dsl_eos_listener_cb listener);
```
This service removes a callback function of type [dsl_eos_listener_cb](#dsl_eos_listener_cb) from a
pipeline identified by it's unique name.

**Parameters**
* `pipeline` - [in] unique name of the Pipeline to update.
* `listener` - [in] state change listener callback function to remove.

**Returns**  
* `DSL_RESULT_SUCCESS` on successful removal. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_pipeline_eos_listener_remove('my-pipeline', eos_listener)
```

<br>

### *dsl_pipeline_qos_listener_add*
```C++
DslReturnType dsl_pipeline_qos_listener_add(const wchar_t* pipeline, 
    dsl_qos_listener_cb listener, void* user_data);
```
This service adds a callback function of type [dsl_qos_listener_cb](#dsl_qos_listener_cb) to a pipeline identified by it's unique name. The function will be called on every Pipeline `QOS` event. Multiple calback functions can be registered with one Pipeline, and one callback function can be registered with multiple Pipelines.

**Parameters**
* `pipeline` - [in] unique name of the Pipeline to update.
* `listener` - [in] state change listener callback function to add.
* `user_data` - [in] opaque pointer to user data returned to the listener is called back

**Returns**  `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
def qos_listener(user_data):
    print('QOS event received')
    
retval = dsl_pipeline_qos_listener_add('my-pipeline', qos_listener, None)
```

<br>

### *dsl_pipeline_qos_listener_remove*
```C++
DslReturnType dsl_pipeline_qos_listener_remove(const wchar_t* pipeline, 
    qos_listener_cb listener);
```
This service removes a callback function of type [dsl_qos_listener_cb](#dsl_qos_listener_cb) from a pipeline identified by it's unique name.

**Parameters**
* `pipeline` - [in] unique name of the Pipeline to update.
* `listener` - [in] state change listener callback function to remove.

**Returns**  
* `DSL_RESULT_SUCCESS` on successful removal. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_pipeline_qos_listener_remove('my-pipeline', qos_listener)
```

<br>

### *dsl_pipeline_play*
```C++
DslReturnType dsl_pipeline_play(wchar_t* pipeline);
```
This service is used to play a named Pipeline. The service will fail if the Pipeline's list of components are insufficient for the Pipeline to play. The service will also fail if one of the Pipeline's components fails to transition to a state of `playing`.  

**Parameters**
* `pipeline` - [in] unique name for the Pipeline to play.

**Returns** 
* `DSL_RESULT_SUCCESS` if the named Pipeline is able to successfully transition to a state of `playing`, one of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_pipeline_play('my-pipeline')
```

<br>

### *dsl_pipeline_pause*
```C++
DslReturnType dsl_pipeline_pause(wchar_t* pipeline);
```
This service is used to pause a named Pipeline. The service will fail if the Pipeline is not currently in a `playing` state. The service will also fail if one of the Pipeline's components fails to transition to a state of `paused`.  

**Parameters**
* `pipeline` - [in] unique name for the Pipeline to pause.

**Returns** 
* `DSL_RESULT_SUCCESS` if the named Pipeline is able to successfully transition to a state of `paused`, one of the [Return Values](#return-values) defined above on failure.

<br>

### *dsl_pipeline_stop*
```C++
DslReturnType  dsl_pipeline_stop(wchar_t* pipeline);
```
This service is used to stop a named Pipeline and return it to a state of `read`. The service will fail if the Pipeline is not currently in a `playing` or `paused` state. The service will also fail if one of the Pipeline's components fails to transition to a state of `ready`.  

**Parameters**
* `pipeline` - [in] unique name for the Pipeline to stop.

**Returns**
* `DSL_RESULT_SUCCESS` if the named Pipeline is able to succesfully transition to a state of `stopped`, one of the [Return Values](#return-values) defined above on failure.

<br>

### *dsl_pipeline_state_get*
```C++
DslReturnType dsl_pipeline_state_get(wchar_t* pipeline, uint* state);
```
This service returns the current [state]() of the named Pipeline The service fails if the named Pipeline was not found.  

**Parameters**
* `pipeline` - [in] unique name for the Pipeline to query.
* `state` - [out] the current [Pipeline State](#pipeline-states) of the named Pipeline

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, state = dsl_pipeline_state_get('my-pipeline')
```

<br>


### *dsl_pipeline_list_size*
```C++
uint dsl_pipeline_list_size();
```
This service returns the size of the current list of Pipelines in memory

**Returns** 
* The number of Pipelines currently in memory

**Python Example**
```Python
retval = dsl_pipeline_delete_all()
```

<br>

### *dsl_pipeline_dump_to_dot*
```C++
DslReturnType dsl_pipeline_dump_to_dot(const char* pipeline, char* filename);
```
This method dumps a Pipeline's graph to dot file. The GStreamer Pipeline will a create 
topology graph on each change of state to ready, playing and paused if the debug 
environment variable `GST_DEBUG_DUMP_DOT_DIR` is set.

GStreamer will add the `.dot` suffix and write the file to the directory specified by
the environment variable. The caller of this service is responsible for providing a 
correctly formatted filename. 

**Parameters**
* `pipeline` - [in] unique name of the Pipeline to dump
* `filename` - [in] name of the file without extension.

**Returns**  `DSL_RESULT_SUCCESS` on successful file dump. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_pipeline_play('my-pipeline`)
if retval != DSL_RESULT_SUCCESS:
    # handle error
    
retval = dsl_pipeline_dump_to_dot('my-pipeline', 'my-pipeline-on-playing')
```

<br>

### *dsl_pipeline_dump_to_dot_with_ts*
```C++
DslReturnType dsl_pipeline_dump_to_dot_with_ts(const char* pipeline, char* filename);
```
This method dumps a Pipeline's graph to dot file prefixed with the current timestamp. 
Except for the prefix, this method performs the identical service as 
[dsl_pipeline_dump_to_dot](#dsl_pipeline_dump_to_dot).

**Parameters**
* `pipeline` - [in] unique name of the Pipeline to dump
* `filename` - [in] name of the file without extension.

**Returns**  `DSL_RESULT_SUCCESS` on successful file dump. One of the [Return Values](#return-values) defined above on failure.
<br>

---

## API Reference
* [List of all Services](/docs/api-reference-list.md)
* **Pipeline**
* [Source](/docs/api-source.md)
* [Dewarper](/docs/api-dewarper.md)
* [Primary and Secondary GIE](/docs/api-gie.md)
* [Tracker](/docs/api-tracker.md)
* [ODE Handler](/docs/api-ode-handler.md)
* [ODE Trigger](/docs/api-ode-trigger.md)
* [ODE Acton](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* [On-Screen Display](/docs/api-osd.md)
* [Tiler](/docs/api-tiler.md)
* [Demuxer and Splitter](/docs/api-tee.md)
* [Sink](/docs/api-sink.md)
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
