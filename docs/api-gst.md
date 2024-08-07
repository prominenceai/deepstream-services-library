# GStreamer (GST) Element API Reference
The GST API is used to create *Custom GStreamer (GST) Elements*. Once created, GST Elements can be added to either a [Custom Source](/docs/api-source.md#custom-video-sources) [Custom Component](/docs/api-component.md#custom-components), or [Custom Sink](/docs/api-sink.md#custom-video-sinks)).


There are restrictions imposed on the type of Elements that can be added to one of the Custom Types. The Element must have at most a single static sink (input) pad and at most a single static source (output) pad. Therefore, no tees, muxers, aggregators, or demuxers (this list may not be complete).


The first element added to a Custom Source
* Must be a Source element and without a sink pad.


The last element added to a Custom Sink
* must be a Sink element and without a source pad.


## Element Construction and Destruction
GST Elements are created by calling [`dsl_gst_element_new`](#dsl_gst_element_new) and deleted by calling [`dsl_gst_element_delete`](#dsl_gst_element_delete), [`dsl_gst_element_delete_many`](#dsl_gst_element_delete_many), or [`dsl_gst_element_delete_all`](#dsl_gst_element_delete_all).


## Element Properties
Elements have properties that define and control there behavior. DSL provides services to read and write properties of type:
* boolean
* float
* int
* uint
* int64
* uint64
* string
* caps


All elements have at least one property which is the `name` property of type string.


## GST Caps Objects
GStreamer Caps Objects define media types and are used to query and set Element properties of type `caps`.
GST Caps Objects are created by calling [`dsl_gst_caps_new`](#dsl_gst_caps_new) and deleted by calling [`dsl_gst_caps_delete`](#dsl_gst_caps_delete), [`dsl_gst_caps_delete_many`](#dsl_gst_caps_delete_many), or [`dsl_gst_caps_delete_all`](#dsl_gst_caps_delete_all).


Using Python for example.
```Python
# Create the new Caps Object from a string representation of the desired caps.
retval = dsl_gst_caps_new('my-video-caps',
  'video/x-raw, framerate=15/1, width=1280, height=720')


# Set the Element's caps property used the Caps Object  
retval = dsl_gst_element_property_caps_set('my-element',
  'caps', 'my-caps-object')


# Delete the Caps Object when done.
retval = dsl_gst_caps_delele('my-caps-object')  
```


## Adding/Removing Pad-Probe-handlers
Multiple sink (input) and/or source (output) [Pad-Probe Handlers](/docs/api-pph.md) can be added to any Element by calling [`dsl_gst_element_pph_add`](#dsl_gst_element_pph_add) and removed with [`dsl_gst_element_pph_remove`](#dsl_gst_element_pph_remove).




---




## GST API
**Constructors**
* [`dsl_gst_caps_new`](#dsl_gst_caps_new)
* [`dsl_gst_element_new`](#dsl_gst_element_new)


**Destructors**
* [`dsl_gst_caps_delete`](#dsl_gst_caps_delete)
* [`dsl_gst_caps_delete_many`](#dsl_gst_caps_delete_many)
* [`dsl_gst_caps_delete_all`](#dsl_gst_caps_delete_all)
* [`dsl_gst_element_delete`](#dsl_gst_element_delete)
* [`dsl_gst_element_delete_many`](#dsl_gst_element_delete_many)
* [`dsl_gst_element_delete_all`](#dsl_gst_element_delete_all)


**Caps Object Methods**
* [`dsl_gst_caps_string_get`](#dsl_gst_caps_string_get)


**Element Methods**
* [`dsl_gst_element_property_boolean_get`](#dsl_gst_element_property_boolean_get)
* [`dsl_gst_element_property_boolean_set`](#dsl_gst_element_property_boolean_set)
* [`dsl_gst_element_property_float_get`](#dsl_gst_element_property_float_get)
* [`dsl_gst_element_property_float_set`](#dsl_gst_element_property_float_set)
* [`dsl_gst_element_property_uint_get`](#dsl_gst_element_property_uint_get)
* [`dsl_gst_element_property_uint_set`](#dsl_gst_element_property_uint_set)
* [`dsl_gst_element_property_int_get`](#dsl_gst_element_property_int_get)
* [`dsl_gst_element_property_int_set`](#dsl_gst_element_property_int_set)
* [`dsl_gst_element_property_uint64_get`](#dsl_gst_element_property_uint64_get)
* [`dsl_gst_element_property_uint64_set`](#dsl_gst_element_property_uint64_set)
* [`dsl_gst_element_property_int64_get`](#dsl_gst_element_property_int64_get)
* [`dsl_gst_element_property_int64_set`](#dsl_gst_element_property_int64_set)
* [`dsl_gst_element_property_string_get`](#dsl_gst_element_property_string_get)
* [`dsl_gst_element_property_string_set`](#dsl_gst_element_property_string_set)
* [`dsl_gst_element_property_caps_get`](#dsl_gst_element_property_caps_get)
* [`dsl_gst_element_property_caps_set`](#dsl_gst_element_property_caps_set)
* [`dsl_gst_element_pph_add`](#dsl_gst_element_pph_add)
* [`dsl_gst_element_pph_remove`](#dsl_gst_element_pph_remove)


---
## Return Values
The following return codes are used by the GStreamer Element API


```C
#define DSL_RESULT_GST_ELEMENT_RESULT                               0x00D00000
#define DSL_RESULT_GST_ELEMENT_NAME_NOT_UNIQUE                      0x00D00001
#define DSL_RESULT_GST_ELEMENT_NAME_NOT_FOUND                       0x00D00002
#define DSL_RESULT_GST_ELEMENT_THREW_EXCEPTION                      0x00D00003
#define DSL_RESULT_GST_ELEMENT_IN_USE                               0x00D00004
#define DSL_RESULT_GST_ELEMENT_SET_FAILED                           0x00D00005
#define DSL_RESULT_GST_ELEMENT_HANDLER_ADD_FAILED                   0x00D00006
#define DSL_RESULT_GST_ELEMENT_HANDLER_REMOVE_FAILED                0x00D00007
#define DSL_RESULT_GST_ELEMENT_PAD_TYPE_INVALID                     0x00D00008
```


---


## Constructors


### *dsl_gst_caps_new*
```C++
DslReturnType dsl_gst_caps_new(const wchar_t* name, const wchar_t* caps);
```
This constructor creates a uniquely named GST Caps Object from a string representation..


**Parameters**
* `name` - [in] unique name for the GStreamer Caps Object to create.
* `caps` - [in] a string defining the caps to create.


**Returns**
`DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure


**Python Example**
```Python
retval = dsl_gst_caps_new('my-video-caps',
  'video/x-raw, framerate=15/1, width=1280, height=720')
```


### *dsl_gst_element_new*
```C++
DslReturnType dsl_gst_element_new(const wchar_t* name, const wchar_t* factory_name);
```
This constructor creates a uniquely named GStreamer Element from a plugin factory name. Construction will fail if the name is currently in use.


**Parameters**
* `name` - [in] unique name for the GStreamer Element to create.
* `factory_name` - [in] factory (plugin) name for the Element to create.


**Returns**
`DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure


**Python Example**
```Python
retval = dsl_gst_element_new('my-element', 'my-plugin)
```


---


## Destructors
### *dsl_gst_caps_delete*
```C++
DslReturnType dsl_gst_caps_delete(const wchar_t* name);
```
This destructor deletes a uniquely named GST Caps Object.


**Parameters**
* `name` - [in] unique name of the GST Caps Object to delete.


**Returns**
`DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure


**Python Example**
```Python
retval = dsl_gst_caps_delete('my-video-caps')
```


<br>


### *dsl_gst_caps_delete_many*
```C++
DslReturnType dsl_gst_caps_delete_many(const wchar_t** names);
```
This destructor deletes a NULL terminated list of GST Caps Objects. This service will return with an error if any of the Caps Objects are currently not found.


**Parameters**
* `names` - [in] NULL terminated list of GStreamer Elements to delete.


**Returns**
`DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure




**Python Example**
```Python
retval = dsl_gst_caps_delete_many('my-caps-1',
  'my-caps-2', 'my-caps-3', None)
```


<br>


## *dsl_gst_caps_delete_all*
```C++
DslReturnType dsl_gst_caps_delete_all();
```
This destructor deletes all GST Caps Objects in memory.


**Returns**
`DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure


**Python Example**
```Python
retval = dsl_gst_caps_delete_all()
```


### *dsl_gst_element_delete*
```C++
DslReturnType dsl_gst_element_delete(const wchar_t* name);
```
This destructor deletes a uniquely named GST Element. This service will fail if the Element is currently in-use .




**Parameters**
* `name` - [in] unique name for the GST Element to delete.


**Returns**
`DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure


**Python Example**
```Python
retval = dsl_gst_element_delete('my-element')
```


<br>


### *dsl_gst_element_delete_many*
```C++
DslReturnType dsl_gst_element_delete_many(const wchar_t** names);
```
This destructor deletes a NULL terminated list of GST Elements. This service will return with an error if any of the Elements are currently in-use or not found.


**Parameters**
* `names` - [in] NULL terminated list of GST Elements to delete.


**Returns**
`DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure




**Python Example**
```Python
retval = dsl_gst_element_delete_many('my-element-1',
  'my-element-2', 'my-element-3', None)
```


<br>


## *dsl_gst_element_delete_all*
```C++
DslReturnType dsl_gst_element_delete_all();
```
This destructor deletes all GST Elements in memory. This service will return with an error if any of the Elements are currently in-use.


**Returns**
`DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure


**Python Example**
```Python
retval = dsl_gst_element_delete_all()
```


---


## Caps Object Methods


### *dsl_gst_caps_string_get*
```C++
DslReturnType dsl_gst_caps_string_get(const wchar_t* name, const wchar_t** caps);
```
This service queries a uniquely named GST Caps Object for its current caps in string format.


**Parameters**
* `name` - [in] unique name for the Caps Object to query.
* `caps` - [out]  a string representation of the Caps Object's current caps.


**Returns**
`DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.


**Python Example**
```Python
retval, caps = dsl_gst_caps_string_get('my-caps-object')
```


<br>


## Element Methods


### *dsl_gst_element_property_boolean_get*
```C++
DslReturnType dsl_gst_element_property_boolean_get(const wchar_t* name,
  const wchar_t* property, boolean* value);
```
This service gets a named boolean property from a named Element.


**Parameters**
* `name` - [in] unique name for the Element to query.
* `property` - [in] unique name of the property to query.
* `value` - [out] current value for the named property.


**Returns**
`DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.


**Python Example**
```Python
retval, value = dsl_gst_element_property_boolean_get('my-element',
  'some-boolean-property')
```


<br>


### *dsl_gst_element_property_boolean_set*
```C++
DslReturnType dsl_gst_element_property_boolean_set(const wchar_t* name,
  const wchar_t* property, boolean value);
```
This service sets a named boolean property for a named Element.


**Parameters**
* `name` - [in] unique name for the Element to update.
* `property` - [in] unique name of the property to update.
* `value` - [in] new value for the named property.


**Returns**
`DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.


**Python Example**
```Python
retval = dsl_gst_element_property_boolean_set('my-element',
  'some-boolean-property', True)
```


<br>


### *dsl_gst_element_property_float_get*
```C++
DslReturnType dsl_gst_element_property_float_get(const wchar_t* name,
  const wchar_t* property, float* value);
```
This service gets a named float property from a named Element.


**Parameters**
* `name` - [in] unique name for the Element to query.
* `property` - [in] unique name of the property to query.
* `value` - [out] current value for the named property.


**Returns**
`DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.


**Python Example**
```Python
retval, value = dsl_gst_element_property_float_get('my-element',
  'some-float-property')
```


<br>


### *dsl_gst_element_property_float_set*
```C++
DslReturnType dsl_gst_element_property_float_set(const wchar_t* name,
  const wchar_t* property, float value);
```
This service sets a named float property for a named Element.


**Parameters**
* `name` - [in] unique name for the Element to update.
* `property` - [in] unique name of the property to update.
* `value` - [in] new value for the named property.


**Returns**
`DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.


**Python Example**
```Python
retval = dsl_gst_element_property_float_set('my-element',
  'some-float-property', 0.99)
```


<br>


### *dsl_gst_element_property_uint_get*
```C++
DslReturnType dsl_gst_element_property_uint_get(const wchar_t* name,
  const wchar_t* property, uint* value);
```
This service gets a named unsigned integer property from a named Element.


**Parameters**
* `name` - [in] unique name for the Element to query.
* `property` - [in] unique name of the property to query.
* `value` - [out] current value for the named property.


**Returns**
`DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.


**Python Example**
```Python
retval, value = dsl_gst_element_property_uint_get('my-element',
  'some-uint-property')
```


<br>


### *dsl_gst_element_property_uint_set*
```C++
DslReturnType dsl_gst_element_property_uint_set(const wchar_t* name,
  const wchar_t* property, uint value);
```
This service sets a named unsigned integer property for a named Element.


**Parameters**
* `name` - [in] unique name for the Element to update.
* `property` - [in] unique name of the property to update.
* `value` - [in] new value for the named property.


**Returns**
`DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.


**Python Example**
```Python
retval = dsl_gst_element_property_uint_set('my-element',
  'some-uint-property', 1234)
```


<br>


### *dsl_gst_element_property_int_get*
```C++
DslReturnType dsl_gst_element_property_int_get(const wchar_t* name,
  const wchar_t* property, int* value);
```
This service gets a named signed integer property from a named Element.


**Parameters**
* `name` - [in] unique name for the Element to query.
* `property` - [in] unique name of the property to query.
* `value` - [out] current value for the named property.


**Returns**
`DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.


**Python Example**
```Python
retval, value = dsl_gst_element_property_int_get('my-element',
  'some-int-property')
```


<br>


### *dsl_gst_element_property_int_set*
```C++
DslReturnType dsl_gst_element_property_int_set(const wchar_t* name,
  const wchar_t* property, int value);
```
This service sets a named signed integer property for a named Element.


**Parameters**
* `name` - [in] unique name for the Element to update.
* `property` - [in] unique name of the property to update.
* `value` - [in] new value for the named property.


**Returns**
`DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.


**Python Example**
```Python
retval = dsl_gst_element_property_int_set('my-element',
  'some-int-property', 1234)
```


<br>


### *dsl_gst_element_property_uint64_get*
```C++
DslReturnType dsl_gst_element_property_uint64_get(const wchar_t* name,
  const wchar_t* property, uint64_t* value);
```
This service gets a named 64 bit unsigned integer property from a named Element.


**Parameters**
* `name` - [in] unique name for the Element to query.
* `property` - [in] unique name of the property to query.
* `value` - [out] current value for the named property.


**Returns**
`DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.


**Python Example**
```Python
retval, value = dsl_gst_element_property_uint64_get('my-element',
  'some-uint64-property')
```


<br>


### *dsl_gst_element_property_uint64_set*
```C++
DslReturnType dsl_gst_element_property_uint64_set(const wchar_t* name,
  const wchar_t* property, uint64_t value);
```
This service sets a named 64 bit unsigned integer property for a named Element.


**Parameters**
* `name` - [in] unique name for the Element to update.
* `property` - [in] unique name of the property to update.
* `value` - [in] new value for the named property.


**Returns**
`DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.


**Python Example**
```Python
retval = dsl_gst_element_property_uint64_set('my-element',
  'some-uint64-property', 0x0123456789abcdef)
```


<br>


### *dsl_gst_element_property_int64_get*
```C++
DslReturnType dsl_gst_element_property_int64_get(const wchar_t* name,
  const wchar_t* property, int64_t* value);
```
This service gets a named 64 bit signed integer property from a named Element.


**Parameters**
* `name` - [in] unique name for the Element to query.
* `property` - [in] unique name of the property to query.
* `value` - [out] current value for the named property.


**Returns**
`DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.


**Python Example**
```Python
retval, value = dsl_gst_element_property_int64_get('my-element',
  'some-int64-property')
```


<br>


### *dsl_gst_element_property_int64_set*
```C++
DslReturnType dsl_gst_element_property_int64_set(const wchar_t* name,
  const wchar_t* property, int64_t value);
```
This service sets a named 64 bit signed integer property for a named Element.


**Parameters**
* `name` - [in] unique name for the Element to update.
* `property` - [in] unique name of the property to update.
* `value` - [in] new value for the named property.


**Returns**
`DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.


**Python Example**
```Python
retval = dsl_gst_element_property_int64_set('my-element',
  'some-int64-property', 0x0123456789abcdef)
```


<br>


### *dsl_gst_element_property_string_get*
```C++
DslReturnType dsl_gst_element_property_string_get(const wchar_t* name,
  const wchar_t* property, const wchar_t** value);
```
This service gets a named string property from a named Element.


**Parameters**
* `name` - [in] unique name for the Element to query.
* `property` - [in] unique name of the property to query.
* `value` - [out] current value for the named property.


**Returns**
`DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.


**Python Example**
```Python
retval, value = dsl_gst_element_property_string_get('my-element',
  'some-string-property')
```


<br>


### *dsl_gst_element_property_string_set*
```C++
DslReturnType dsl_gst_element_property_string_set(const wchar_t* name,
  const wchar_t* property, const wchar_t* value);
```
This service sets a named string property for a named Element.


**Parameters**
* `name` - [in] unique name for the Element to update.
* `property` - [in] unique name of the property to update.
* `value` - [in] new value for the named property.


**Returns**
`DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.


**Python Example**
```Python
retval = dsl_gst_element_property_string_set('my-element',
  'some-string-property', 'some-string-value')
```


### *dsl_gst_element_property_caps_get*
```C++
DslReturnType dsl_gst_element_property_caps_get(const wchar_t* name,
   const wchar_t* property, const wchar_t* caps);
```
This service creates a named GST Caps Object from a named Element's property of type caps. After the call, the new Caps Object can be queried for its string representation by calling [dsl_gst_caps_string_get](#dsl_gst_caps_string_get).


**Parameters**
* `name` - [in] unique name for the Element to query.
* `property` - [in] unique name of the property to query, typically "caps".
* `caps` - [out] unique name for the Caps Object to create.


**Returns**
`DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.


**Python Example**
```Python
# Create the new GST Caps object from the element's 'caps' property
retval = dsl_gst_element_property_caps_get('my-element', 'caps', 'new-caps-object')


# Read the caps string from the Caps Object.
retval, caps = dsl_gst_caps_string_get('new-caps-object')


# Delete the Caps Object when done.
retval = dsl_gst_caps_delele('new-caps-object')
```


<br>


### *dsl_gst_element_property_caps_set*
```C++
DslReturnType dsl_gst_element_property_caps_set(const wchar_t* name,
   const wchar_t* property, const wchar_t* caps);
```
This service sets a named string property for a named Element.


**Parameters**
* `name` - [in] unique name for the Element to update.
* `property` - [in] unique name of the property to update, typically "caps".
* `value` - [in] unique name of the Caps Object to use.


**Returns**
`DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.


**Python Example**
```Python
# Create the new Caps Object from a string representation of the desired caps.
retval = dsl_gst_caps_new('my-video-caps',
  'video/x-raw, framerate=15/1, width=1280, height=720')


# Set the Element's caps property used the Caps Object  
retval = dsl_gst_element_property_caps_set('my-element',
  'caps', 'my-caps-object')


# Delete the Caps Object when done.
retval = dsl_gst_caps_delele('my-caps-object')  
```


<br>


### *dsl_gst_element_pph_add*
```C++
DslReturnType dsl_gst_element_pph_add(const wchar_t* name,
  const wchar_t* handler, uint pad);
```
This service adds a [Pad Probe Handler](/docs/api-pph.md) to either the Sink or Source pad of the named Element.


**Parameters**
* `name` - [in] unique name of the Element to update.
* `handler` - [in] unique name of Pad Probe Handler to add.
* `pad` - [in]  which of the two pads to add the handler to: `DSL_PAD_SIK` or `DSL_PAD SRC`


**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.


**Python Example**
```Python
retval = dsl_gst_element_pph_add('my-element', 'my-pph-handler', DSL_PAD_SINK)
```


<br>


### *dsl_gst_element_pph_remove*
```C++
DslReturnType dsl_gst_element_pph_remove(const wchar_t* name,
  const wchar_t* handler, uint pad);
```
This service removes a [Pad Probe Handler](/docs/api-pph.md) from either the Sink or Source pad of the named GST Element. This service will fail if the named handler is not owned by the Inference Component


**Parameters**
* `name` - [in] unique name of the Element to update.
* `handler` - [in] unique name of Pad Probe Handler to remove
* `pad` - [in] to which of the two pads to remove the handler from: `DSL_PAD_SIK` or `DSL_PAD SRC`


**Returns**
* `DSL_RESULT_SUCCESS` on successful remove. One of the [Return Values](#return-values) defined above on failure.


**Python Example**
```Python
retval = dsl_gst_element_pph_remove('my-element', 'my-pph-handler', DSL_PAD_SINK)
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
* [Preprocessor](/docs/api-preproc.md)
* [Primary and Secondary Inference](/docs/api-infer.md)
* [Tracker](/docs/api-tracker.md)
* [Segmentation Visualizer](/docs/api-segvisual.md)
* [Tiler](/docs/api-tiler.md)
* [Demuxer and Splitter Tees](/docs/api-tee.md)
* [Remuxer](/docs/api-remuxer.md)
* [On-Screen Display](/docs/api-osd.md)
* [Sink](/docs/api-sink.md)
* **GST Element**
* [Pad Probe Handler](/docs/api-pph.md)
* [ODE Trigger](/docs/api-ode-trigger.md)
* [ODE Accumulator](/docs/api-ode-accumulator.md)
* [ODE Acton](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* [ODE Heat-Mapper](/docs/api-ode-heat-mapper.md)
* [Display Types](/docs/api-display-types.md)
* [branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
* [Mailer](/docs/api-mailer.md)
* [WebSocket Server](/docs/api-ws-server.md)
* [Message Broker](/docs/api-msg-broker.md)
* [Info API](/docs/api-info.md)
