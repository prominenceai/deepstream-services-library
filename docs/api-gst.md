# GStreamer (GST) Element and Bin API Reference
The GST API is used to create Custom DSL Pipeline Components. All DSL Pipeline Components are derived from the [GST Bin](https://gstreamer.freedesktop.org/documentation/application-development/basics/bins.html?gi-language=c) container class. Bins are used to  contain [GST Elements](https://gstreamer.freedesktop.org/documentation/application-development/basics/bins.html?gi-language=c) Bins allow you to combine a group of linked Elements into one logical Element. 

There are restrictions imposed on the type of Elements that can be created 
* Single input pad and output pad only. 
* Therefore, no tees, muxers, aggregators, or demuxers (this list may not be complete).

The first Element in each Bin is typically a [queue](https://gstreamer.freedesktop.org/documentation/coreelements/queue.html?gi-language=c#properties) Element. Adding a queue creates a new thread on the Element's source pad (output) to decouple the processing between input and output creating a new thread for the Custom Component. 

## Construction and Destruction
GST Elements are created by calling [`dsl_gst_element_new`](#dsl_gst_element_new) and deleted by calling [`dsl_gst_element_delete`](#dsl_gst_element_delete), [`dsl_gst_element_delete_many`](#dsl_gst_element_delete_many), or [`dsl_gst_element_delete_all`](#dsl_gst_element_delete_all).

GST Bins are created by calling [`dsl_gst_bin_new`](#dsl_gst_bin_new) or [`dsl_gst_bin_new_element_add_many`](#dsl_gst_bin_new_element_add_many). As with all Pipeline Components, GST Bins are deleted by calling [`dsl_component_delete`](/docs/api-component.md#dsl_component_delete), [`dsl_component_delete_many`](/docs/api-component.md#dsl_component_delete_many), or [`dsl_component_delete_all`](/docs/api-component.md#dsl_component_delete_all).

## Adding and Removing
The relation shipe between GST Bins and GST Elements is one to many. Once added to a Bin, an Element must be removed before it can be used with another. Elements are added to bins by calling [`dsl_gst_bin_new_element_add_many`](#dsl_gst_bin_new_element_add_many), [`dsl_gst_bin_element_add`](#dsl_gst_bin_element_add), and [`dsl_gst_bin_element_add_many`](#dsl_gst_bin_element_add_many). GST Elements can be removed from a GST Bin by calling [`dsl_gst_bin_element_remove`](#dsl_gst_bin_element_remove) or [`dsl_gst_bin_element_remove_many`](#dsl_gst_bin_element_remove_many).

The relationship between Pipelines/Branches  and GST Bins is one to many. Once added to a Pipeline or Branch, a Bin must be removed before it can be used with another. GST Bins are added to a Pipeline by calling [`dsl_pipeline_component_add`](/docs/api-pipeline.md#dsl_pipeline_component_add) or [`dsl_pipeline_component_add_many`](/docs/api-pipeline.md#dsl_pipeline_component_add_many) and removed with [`dsl_pipeline_component_remove`](/docs/api-pipeline.md#dsl_pipeline_component_remove), [`dsl_pipeline_component_remove_many`](/docs/api-pipeline.md#dsl_pipeline_component_remove_many), or [`dsl_pipeline_component_remove_all`](/docs/api-pipeline.md#dsl_pipeline_component_remove_all).

A similar set of Services are used when adding/removing a to/from a branch: [`dsl_branch_component_add`](api-branch.md#dsl_branch_component_add), [`dsl_branch_component_add_many`](/docs/api-branch.md#dsl_branch_component_add_many), [`dsl_branch_component_remove`](/docs/api-branch.md#dsl_branch_component_remove), [`dsl_branch_component_remove_many`](/docs/api-branch.md#dsl_branch_component_remove_many), and [`dsl_branch_component_remove_all`](/docs/api-branch.md#dsl_branch_component_remove_all).

```Python
# IMPORTANT! We create a queue element to be our first element of our bin.
# The queue will create a new thread on the source pad (output) to decouple 
# the processing on sink and source pad, effectively creating a new thread for 
# our custom component.
retval = dsl_gst_element_new('my-queue', 'queue')

# Create a new element from a proprietary plugin
retval = dsl_gst_element_new('my-element', 'my-plugin-name')
            
# Create a new bin and add the elements to it. The elements will be linked 
# in the order they're added.
ret_val = dsl_gst_bin_new_element_add_many('my-custom-bin', 
    elements = ['my-queue', 'my-element', None])

# The Custom Component can now be added to our Pipeline along with
# the our other Pipeline components. Add in the order to be linked.
retval = dsl_pipeline_new_component_add_many('pipeline', 
    ['my-source', 'my-primary-gie', 'my-iou-tracker', 'my-custom-bin',
    'my-on-screen-display', 'my-egl-sink', None])
            
# IMPORTANT! set the link method for the Pipeline to link by 
# add order (and not by fixed position - default)
retval = dsl_pipeline_link_method_set('pipeline',
    DSL_PIPELINE_LINK_METHOD_BY_ORDER)
```

## Adding/Removing Pad-Probe-handlers
Multiple sink (input) and/or source (output) [Pad-Probe Handlers](/docs/api-pph.md) can be added to any Primary or Secondary GIE or TIS by calling [`dsl_infer_pph_add`](#dsl_infer_pph_add) and removed with [`dsl_infer_pph_remove`](#dsl_infer_pph_remove).


---

## Primary and Secondary Inference API
**Constructors**
* [`dsl_gst_element_new`](#dsl_gst_element_new)
* [`dsl_gst_bin_new`](#dsl_gst_bin_new)
* [`dsl_gst_bin_new_element_add_many`](#dsl_gst_bin_new_element_add_many)

**Destructors**
* [`dsl_gst_element_delete`](#dsl_gst_element_delete)
* [`dsl_gst_element_delete_many`](#dsl_gst_element_delete_many)
* [`dsl_gst_element_delete_all`](#dsl_gst_element_delete_all)

**Methods**
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
* [`dsl_gst_element_list_size`](#dsl_gst_element_list_size)
* [`dsl_gst_element_pph_add`](#dsl_gst_element_pph_add)
* [`dsl_gst_element_pph_remove`](#dsl_gst_element_pph_remove)
* [`dsl_gst_bin_element_add`](#dsl_gst_bin_element_add)
* [`dsl_gst_bin_element_add_many`](#dsl_gst_bin_element_add_many)
* [`dsl_gst_bin_element_remove`](#dsl_gst_bin_element_remove)
* [`dsl_gst_bin_element_remove_many`](#dsl_gst_bin_element_remove_many)

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

The following return codes are used by the GStreamer Element API
```C
#define DSL_RESULT_GST_BIN_RESULT                                   0x00E00000
#define DSL_RESULT_GST_BIN_NAME_NOT_UNIQUE                          0x00E00001
#define DSL_RESULT_GST_BIN_NAME_NOT_FOUND                           0x00E00002
#define DSL_RESULT_GST_BIN_NAME_BAD_FORMAT                          0x00E00003
#define DSL_RESULT_GST_BIN_THREW_EXCEPTION                          0x00E00004
#define DSL_RESULT_GST_BIN_IS_IN_USE                                0x00E00005
#define DSL_RESULT_GST_BIN_SET_FAILED                               0x00E00006
#define DSL_RESULT_GST_BIN_ELEMENT_ADD_FAILED                       0x00E00007
#define DSL_RESULT_GST_BIN_ELEMENT_REMOVE_FAILED                    0x00E00008
#define DSL_RESULT_GST_BIN_ELEMENT_NOT_IN_USE                       0x00E00009
```

---

## Constructors

### *dsl_gst_element_new*
```C++
DslReturnType dsl_gst_element_new(const wchar_t* name, const wchar_t* factory_name);
```
This constructor creates a uniquely name GStreamer Element from a plugin factory name. Construction will fail if the name is currently in use.

**Parameters**
* `name` - [in] unique name for the GStreamer Element to create.
* `factory_name` - [in] factory (plugin) name for the Element to create.

**Returns**
`DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_gst_element_new('my-element', 'my-plugin)
```

<br>

### *dsl_gst_bin_new*
```C++
DslReturnType dsl_gst_bin_new(const wchar_t* name);
```
This constructor creates a uniquely name GStreamer Bin. Construction will fail if the name is currently in use.

**Parameters**
* `name` - [in] unique name for the GStreamer Been to create.

**Returns**
`DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_gst_bin_new('my-bin')
```

<br>

### *dsl_gst_bin_new_element_add_many*
```C++
DslReturnType dsl_gst_bin_new_element_add_many(const wchar_t* name, 
    const wchar_t** components);
```
This constructor creates a uniquely name GStreamer Bin and adds a list of Elements to it. Construction will fail if the name is currently in use.

**Parameters**
* `name` - [in] unique name for the GStreamer Been to create.
* `components` - [in] NULL terminated array of Element names to add.

**Returns**
`DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_gst_bin_new_element_add_many('my-bin', 
    [my-element-1, my-element-2, my-element-3, None])
```

---

## Destructors

### *dsl_gst_element_delete*
```C++
DslReturnType dsl_gst_element_delete(const wchar_t* name);
```
This destructor deletes a uniquely name GStreamer Element. This services will fail if the element is currently in-use with a GST Bin.

**Parameters**
* `name` - [in] unique name for the GStreamer Element to delete.

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
This destructor deletes a NULL terminated list of GStreamer Elements. This service will return with an error if any of the Elements are currently in-use or not found.

**Parameters**
* `names` - [in] NULL terminated list of GStreamer Elements to delete.

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
This destructor deletes a NULL terminated list of GStreamer Elements. This service will return with an error if any of the Elements are currently in-use.

**Parameters**

**Returns**
`DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_gst_element_delete_all()
```

--- 

## Methods

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

### *dsl_gst_bin_element_add*
```C++
DslReturnType dsl_gst_bin_element_add(const wchar_t* name, const wchar_t* element);
```
This service adds a single named Element to a named Bin. The add service will fail if the Element is currently `in-use` by any Bin. The Element's `in-use` state will be set to `true` on successful add.

**Parameters**
* `name` - [in] unique name for the Bin to update.
* `element` - [in] unique name of the Element to add.

**Returns**
* `DSL_RESULT_SUCCESS` on successful addition. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_gst_bin_element_add('my-bin', 'my-element')
```

<br>

### *dsl_gst_bin_element_add_many*
```C++
DslReturnType dsl_gst_bin_element_add_many(const wchar_t* name, const wchar_t** elements);
```
Adds a list of named Elements to a named Bin. The add service will fail if any of the Elements are currently `in-use` by any Bin. All of the Element's `in-use` state will be set to true on successful add.

* `name` - [in] unique name for the Bin to update.
* `elements` - [in] a NULL terminated array of uniquely named Elements to add.

**Returns**
* `DSL_RESULT_SUCCESS` on successful  addition. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_gst_bin_element_add_many('my-bin', 
    ['my-element-1', 'my-element-2', None])
```

<br>

---
### *dsl_gst_bin_element_remove*
```C++
DslReturnType dsl_gst_bin_element_remove(const wchar_t* name, const wchar_t* element);
```
This service removes a single named Element from a named Bin.

**Parameters**
* `name` - [in] unique name for the Bin to update.
* `element` - [in] unique name of the Element to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful removal. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_gst_bin_element_remove('my-bin', 'my-element')
```

<br>

### *dsl_gst_bin_element_remove_many*
```C++
DslReturnType dsl_gst_bin_element_remove_many(const wchar_t* name, const wchar_t** elements);
```
Removes a list of named Elements from a named Bin. 

* `name` - [in] unique name for the Bin to update.
* `elements` - [in] a NULL terminated array of uniquely named Elements to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful removal. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_gst_bin_element_remove_many('my-bin', 
    ['my-element-1', 'my-element-2', None])
```

<br>


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
* **Custom Component**
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