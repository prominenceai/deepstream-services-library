# Remuxer API

Built with a Demuxer, multiple Streammuxers, and a Metamux The Remuxer Tee splits the batched input stream into downstream branches, each with their own unique batched metatdata for parallel inference.  

Remuxing a batched stream is performed as follows:
1. The Demuxer plugin is used to demux the incoming batched stream into individual streams/source-pads.
2. GStreamer tee plugins are connected to the source-pads splitting each single stream into multiple single streams, as required for each downstream Branch.
3. Each added Branch is connected upstream to an NVIDIA [Gst-nvstreammux plugin](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvstreammux.html) 
4. Each Streammuxer is then connected upstream to some or all of the single stream Tees, as specified by the client.

DSL supports both the [**OLD** NVIDIA Streammux pluging](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvstreammux.html) and the [**NEW** NVIDIA Streammux plugin](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvstreammux2.html) 

## Remuxer API
**Constructors**
* [`dsl_remuxer_new`](#dsl_remuxer_new)
* [`dsl_remuxer_new_branch_add_many`](#dsl_remuxer_new_branch_add_many)

**Remuxer Methods (common)**
* [`dsl_remuxer_branch_add_to`](#dsl_remuxer_branch_add_to)
* [`dsl_remuxer_branch_add`](#dsl_remuxer_branch_add)
* [`dsl_remuxer_branch_add_many`](#dsl_remuxer_branch_add_many)
* [`dsl_remuxer_branch_remove`](#dsl_remuxer_branch_remove)
* [`dsl_remuxer_branch_remove_many`](#dsl_remuxer_branch_remove_many)
* [`dsl_remuxer_branch_remove_all`](#dsl_remuxer_branch_remove_all)
* [`dsl_remuxer_pph_add`](#dsl_remuxer_pph_add)
* [`dsl_remuxer_pph_remove`](#dsl_remuxer_pph_remove)

**Remuxer Methods (old Streammuxer)**
* [`dsl_remuxer_batch_properties_get`](#dsl__remuxer_batch_properties_get)
* [`dsl_remuxer_batch_properties_set`](#dsl_remuxer_batch_properties_set)
* [`dsl_remuxer_dimensions_get`](#dsl_remuxer_dimensions_get)
* [`dsl_remuxer_dimensions_set`](#dsl_remuxer_dimensions_set)

**Remuxer Methods (new Streammuxer)**
* [`dsl_remuxer_branch_config_file_get`](#dsl_remuxer_branch_config_file_get)
* [`dsl_remuxer_branch_config_file_set`](#dsl_remuxer_branch_config_file_set)
* [`dsl_remuxer_batch_size_get`](#dsl_remuxer_batch_size_get)
* [`dsl_remuxer_batch_size_set`](#dsl_remuxer_batch_size_set)

## Return Values
The following return codes are used by the Tee API
```C++

```

