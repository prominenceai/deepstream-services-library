# SMTP Mailer API
Mailer objects are used to send email using a client provided secure SMTPS Server URL and Credentials. 

Mailer objects are added to [ODE Actions](/docs/api-ode-action.md) and Recording [Sinks](/docs/api-sink.md) and [Taps](/docs/api-tap.md) enabling them to send email on specific events. Queuing of the event data occurs in the Action's/Component's real time context, while the tasks of assembling the message and uploading to the SMTP server are performed in a low priority background thread. Messages that fail to send will be purged from the queue, dropped and logged as an ERROR.

The relationship between Mailers and Actions/Components is many to many as multiple Mailers can be added to a single Action/Component and the same Mailer can be added to multiple Actions/Components. 

#### Mailer Construction and Destruction
Mailers are created by calling the constructor [dsl_mailer_new](#dsl_mailer_new). Once created, they must be set up with a Server URL, Credentials, etc., prior to use. Mailers are destructured by calling [dsl_mailer_delete](#dsl_mailer_delete) or [dsl_mailer_delete_all](#dsl_mailer_delete_all).

#### Adding Mailers to ODE Actions and Recording Components

* **Email Action** - added to the Action on construction with [dsl_ode_action_email_new](/docs/api-ode-action.md/#dsl_ode_action_email_new).
* **Image Capture Actions** - added to both Frame and Object Capture Actions with [dsl_ode_action_capture_mailer_add](/docs/api-ode-action.md#dsl_ode_action_capture_mailer_add) and removed with [dsl_ode_action_capture_mailer_remove](/docs/api-ode-action.md#dsl_ode_action_capture_mailer_remove).
* **Smart Recording Tap** - added with [dsl_tap_record_mailer_add](/docs/api-tap.md#dsl_tap_record_mailer_add) and removed with [dsl_tap_record_mailer_remove](/docs/api-tap.md#dsl_tap_record_mailer_remove).
* **Smart Recording Sink** - added with [dsl_sink_record_mailer_add](/docs/api-sink.md#dsl_sink_record_mailer_add) and removed with [dsl_sink_record_mailer_remove](/docs/api-sink.md#dsl_sink_record_mailer_remove)

## Using GMAIL's SMTP server
**IMPORTANT!** if using GMAIL, it is STRONGLY advised that you create a new, free [Gmail account](https://support.google.com/mail/answer/56256?hl=en) -- that is separate/unlinked from all your other email accounts -- strictly for the purpose of sending ODE Event data uploaded from DSL.  Then, add your Personal email address as a `To` address to receive the emails. 

Gmail considers regular email programs (i.e Outlook, etc.) and non-registered third-party apps to be "less secure". The email account used for sending email must have the "Allow less secure apps" option turned on. Once you've created this new account, you can go to the account settings and enable [Less secure app access](https://myaccount.google.com/lesssecureapps).

The Gmail secure SMTP server URL is `smtps://smtp.gmail.com:465`. Port `465` requires SSL to be enabled which is set by default. See [dsl_smtp_ssl_enabled_get](#dsl_smtp_ssl_enabled_get) and [dsl_smtp_ssl_enabled_set](#dsl_smtp_ssl_enabled_set)

### Example setup using Python: 
The following example assumes that all `retval` values are checked before proceeding to the next call.
```Python
# Create a new Mailer Object
retval = dsl_mailer_new('my-mailer')

# Setup the Server URL
retval = dsl_mailer_server_url_set('my-mailer', 'smtps://smtp.gmail.com:465')

# Using the credentials for your new Gmail SMTP account
retval = dsl_mailer_credentials_set('my-mailer', user_name, password)

# Set the From email address, again using the new SMTP account.
retval = dsl_mailer_address_from_set('my-mailer', '', 'my.smtp.server@gmail.com')

# Add your personal email account as a To address. 
# Optionally, add other To and Cc addresses.
retval = dsl_mailer_address_to_add('my-mailer', to_name, to_address)
        
# queue a test message to be sent out to ensure all settings are correct
retval = dsl_mailer_test_message_send('my-mailer')
```

## SMTP API
**Constructors:**
* [dsl_mailer_new](#dsl_mailer_new)

**Destructors:**
* [dsl_mailer_delete](#dsl_mailer_delete)
* [dsl_mailer_delete_all](#dsl_mailer_delete_all)
 
**Methods**
* [dsl_mailer_enabled_get](#dsl_mailer_enabled_get)
* [dsl_mailer_enabled_set](#dsl_mailer_enabled_set)
* [dsl_mailer_credentials_set](#dsl_mailer_credentials_set)
* [dsl_mailer_server_url_get](#dsl_mailer_server_url_get)
* [dsl_mailer_server_url_set](#dsl_mailer_server_url_set)
* [dsl_mailer_ssl_enabled_get](#dsl_mailer_ssl_enabled_get)
* [dsl_mailer_ssl_enabled_set](#dsl_mailer_ssl_enabled_set)
* [dsl_mailer_address_from_get](#dsl_mailer_address_from_get)
* [dsl_mailer_address_from_set](#dsl_mailer_address_from_set)
* [dsl_mailer_address_to_add](#dsl_mailer_address_to_add)
* [dsl_mailer_address_to_remove_all](#dsl_mailer_address_to_remove_all)
* [dsl_mailer_address_cc_add](#dsl_mailer_address_cc_add)
* [dsl_mailer_address_cc_remove_all](#dsl_mailer_address_cc_remove_all)
* [dsl_mailer_test_message_send](#dsl_mailer_test_message_send)
* [dsl_mailer_exists](#dsl_mailer_exists)
* [dsl_mailer_list_size](#dsl_mailer_list_size)

## Return Values
The following return codes are used by the SMTP API
```C++
#define DSL_RESULT_MAILER_RESULT                                    0x00500000
#define DSL_RESULT_MAILER_NAME_NOT_UNIQUE                           0x00500001
#define DSL_RESULT_MAILER_NAME_NOT_FOUND                            0x00500002
#define DSL_RESULT_MAILER_THREW_EXCEPTION                           0x00500003
#define DSL_RESULT_MAILER_IN_USE                                    0x00500004
#define DSL_RESULT_MAILER_SET_FAILED                                0x00500005
#define DSL_RESULT_MAILER_PARAMETER_INVALID                         0x00500006
```

## Constants
The following constant values are used by the SMTP API
```C
#define DSL_MAILER_MAX_PENDING_MESSAGES                             10
```

<br>

---

## Constructors
### *dsl_mailer_new*
```C++
DslReturnType dsl_mailer_new(const wchar_t* name);
```
The constructor creates a uniquely named, ***uninitialized*** Mailer Object. The Mailer must be set up with a Server URL, Credentials, and Email Addresses before first use.

**Parameters**
* `name` - [in] unique name for the Mailer to create.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_mailer_new('my-mailer')
```

---

## Destructors
### *dsl_mailer_delete*
```C++
DslReturnType dsl_mailer_delete(const wchar_t* name);
```
This destructor deletes a single, uniquely named Mailer. The destructor will fail if the Mailer is currently `in-use` by one or more Actions or Components.

**Parameters**
* `name` - [in] unique name of the Mailer to delete.

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_mailer_delete('my-mailer')
```

<br>

### *dsl_mailer_delete_all*
```C++
DslReturnType dsl_mailer_delete_all();
```
This destructor deletes all Mailers currently in memory. The destructor will fail if any one of the Mailers is currently `in-use` by one or more Actions or Components.. 

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_mailer_delete_all()
```

<br>

---

## Methods

### *dsl_mailer_enabled_get*
```C++
DslReturnType dsl_mailer_enabled_get(const wchar_t* name, boolean* enabled);
```
This service queries the Mailer for its current enabled state. Mailers are automatically disabled if and while their outgoing queue size exceeds `DSL_MAILER_MAX_PENDING_MESSAGES`. Services are enabled by default.

**Parameters**
* `name` - [in] unique name of the Mailer to query.
* `enabled` - [out] true if the Mailer is currently enabled, false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful call. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, enabled = dsl_mailer_enabled_get('my-mailer')
```

<br>

### *dsl_mailer_enabled_set*
```C++
DslReturnType dsl_mailer_enabled_set(const wchar_t* name, boolean enabled);
```
This service sets the enabled state for the named Mailer object. Setting the state to true while the outgoing queue size exceeds `DSL_SMTP_MAX_PENDING_MESSAGES` will return `DSL_RESULT_FAILURE`. Mailers are enabled by default.

**Parameters**
* `name` - [in] unique name of the Mailer to update.
* `enabled` - [in] set to true to enable the Mailer, false to disable.

**Returns**
* `DSL_RESULT_SUCCESS` on successful call. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_mailer_enabled_set('my-mailer', True)
```

<br>

### *dsl_mailer_credentials_set*
```C++
DslReturnType dsl_mailer_credentials_set(const wchar_t* name, 
    const wchar_t* username, const wchar_t* password);
```
This service is used to set the SMTP account credentials, username and password, for all subsequent emails sent by the Mailer.

**Parameters**
* `name` - [in] unique name of the Mailer to update.
* `username` - [in] username for the SMTP account.
* `password` - [in] password for the same account.

**Returns**
* `DSL_RESULT_SUCCESS` on successful call. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_mailer_credentials_set('my-mailer', 
  'my-dsl-smtp-account', 'my-dsl-smtp-password')
```

<br>

### *dsl_mailer_server_url_get*
```C++
DslReturnType dsl_mailer_server_url_get(const wchar_t* name,
    const wchar_t** server_url);
```
This service gets the current Server URL in use by the named Mailer. The service will return an empty string until the URL is set by calling [dsl_mailer_server_url_set](#dsl_mailer_server_url_set).

**Parameters**
* `name` - [in] unique name of the Mailer to query.
* `server_url` - [out] current Server URL in use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful call. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, server_url = dsl_mailer_server_url_get('my-mailer')
```

<br>

### *dsl_mailer_server_url_set*
```C++
DslReturnType dsl_mailer_server_url_set(const wchar_t* name,
    const wchar_t* server_url);
```
This service sets the Server URL to use for all subsequent emails sent out by the named Mailer.

**Parameters**
* `name` - [in] unique name of the Mailer to update.
* `server_url` - [in] new Server URL to use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful call. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_mailer_server_url_set('my-mailer',
    'smtps://smtp.gmail.com:465')
```

<br>

### *dsl_mailer_address_from_get*
```C++
DslReturnType dsl_mailer_address_from_get(const wchar_t* name,
    const wchar_t** display_name, const wchar_t** address);
```
This service gets the current `From` address in use by the named Mailer. Both the display_name and address parameters will return an empty string until set with a call to [dsl_mailer_address_from_set](#dsl_mailer_address_from_set).

**Parameters**
* `name` - [in] unique name of the Mailer to query.
* `display_name` - [out] returns the display name (optional) for the `From` address in use.
* `address` - [out] returns the email address in use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful call. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, display_name, address = dsl_mailer_address_from_get('my-mailer')
```

<br>

### *dsl_mailer_address_from_set*
```C++
DslReturnType dsl_mailer_address_from_set(const wchar_t* name,
    const wchar_t* display_name, const wchar_t* address);
```
This service sets the `From` address to use for all subsequent emails by the name Mailer.

**Parameters**
* `name` - [in] unique name of the Mailer to update.
* `display_name` - [in] the display name (optional) to use for the `From` address.
* `address` - [in] the email address to use. Will be set by the server if omitted.

**Returns**
* `DSL_RESULT_SUCCESS` on successful call. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval  = dsl_mailer_address_from_set('my-mailer', 
    'Joe Blow', 'joe.blow@example.com')
```

<br>

### *dsl_mailer_ssl_enabled_get*
```C++
DslReturnType dsl_mailer_ssl_enabled_get(const wchar_t* name, boolean* enabled);
```
This service gets the SSL enabled setting for the named Mailer. SSL is enabled by default

**Parameters**
* `name` - [in] unique name of the Mailer to query.
* `enabled` - [out] true if SSL is currently enabled, false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful call. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, enabled = dsl_mailer_ssl_enabled_get('my-mailer')
```

<br>

### *dsl_mailer_ssl_enabled_set*
```C++
DslReturnType dsl_mailer_ssl_enabled_set(const wchar_t* name, boolean enabled);
```
This service sets the SSL enabled state for the named Mailer. SSL is enabled by default.

**Parameters**
* `name` - [in] unique name of the Mailer to update.
* `enabled` - [in] set to true to enable SSL, false to disable.

**Returns**
* `DSL_RESULT_SUCCESS` on successful call. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_mailer_ssl_enabled_set('my-mailer', True)
```

<br>

### *dsl_mailer_address_to_add*
```C++
DslReturnType dsl_mailer_address_to_add(const wchar_t* name,
    const wchar_t* display_name, const wchar_t* address);
```
This service adds a `To` address to use for all subsequent emails.

**Parameters**
* `name` - [in] unique name of the Mailer to update.
* `display_name` - [in] the display name (optional) to use for the `To` address.
* `address` - [in] the email address to add. 

**Returns**
* `DSL_RESULT_SUCCESS` on successful call. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval  = dsl_mailer_address_to_add('my-mailer', 
    'Jane Doe', 'jane.doe@example.com')
```

<br>

### *dsl_mailer_address_to_remove_all*
```C++
DslReturnType dsl_mailer_address_to_remove_all(const wchar_t* name);
```
This service removes all `To` addresses from the named Mailer. At least one `To` address must exist for email to be sent out.

**Parameters**
* `name` - [in] unique name of the Mailer to update.

**Returns**
* `DSL_RESULT_SUCCESS` on successful call. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval  = dsl_mailer_address_to_remove_all('my-mailer')
```

<br>

### *dsl_mailer_address_cc_add*
```C++
DslReturnType dsl_mailer_address_cc_add(const wchar_t* name,
    const wchar_t* display_name, const wchar_t* address);
```
This service adds a `Cc` address to use for all subsequent emails by the named Mailer. `Cc` addresses are optional.

**Parameters**
* `name` - [in] unique name of the Mailer to update.
* `display_name` - [in] the display name (optional) to use for the `Cc` address.
* `address` - [in] the email address to add. 

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval  = dsl_mailer_address_cc_add('my-mailer',
    'Jane Doe', 'jane.doe@example.com')
```

<br>

### *dsl_mailer_address_cc_remove_all*
```C++
DslReturnType dsl_mailer_address_cc_remove_all(const wchar_t* name);
```
This service removes all `Cc` addresses from the named Mailer. `Cc` addresses are optional.

**Parameters**
* `name` - [in] unique name of the Mailer to update.

**Returns**
* `DSL_RESULT_SUCCESS` on successful call. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval  = dsl_mailer_address_cc_remove_all('my-mailer')
```

<br>

### *dsl_mailer_test_message_send*
```C++
DslReturnType dsl_mailer_test_message_send(const wchar_t* name);
```
This service sends a test message using the current SMTP settings and email addresses: `From`, `To`, and `Cc`.
**Note:** The message will not be sent until the `main_loop` has been started.

**Parameters**
* `name` - [in] unique name of the Mailer to test.

**Returns**
* `DSL_RESULT_SUCCESS` on successful queue. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval  = dsl_mailer_test_message_send('my-mailer')
```

<br>

### *dsl_mailer_exists*
```C++
boolean dsl_mailer_exists(const wchar_t* name);
```
This service is used to determine if a named Mailer currently exists in memory.

**Parameters**
* `name` - [in] unique name of the Mailer to test.

**Returns**
* `true` if the Mailer exists, false otherwise.

**Python Example**
```Python
exists = dsl_mailer_exists('my-mailer')
```

<br>

### *dsl_mailer_list_size*
```C++
uint dsl_mailer_list_size();
```
This service returns the current number of Mailers in memory.

**Returns**
* The current number of mailers in memory.

**Python Example**
```Python
num_mailers = dsl_mailer_list_size()
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
* [Inference Engine and Server](/docs/api-infer.md)
* [Tracker](/docs/api-tracker.md)
* [Segmentation Visualizer](/docs/api-segvisual.md)
* [Tiler](/docs/api-tiler.md)
* [Splitter and Demuxer](/docs/api-tee.md)
* [On-Screen Display](/docs/api-osd.md)
* [Sink](/docs/api-sink.md)
* [Pad Probe Handler](/docs/api-pad-probe-handler.md)
* [ODE Trigger](/docs/api-ode-trigger.md)
* [ODE Accumulator](/docs/api-ode-accumulator.md)
* [ODE Acton](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* [ODE Heat-Mapper](/docs/api-ode-heat-mapper.md)
* [Display Type](/docs/api-display-type.md)
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
* **Mailer**
* [WebSocket Server](/docs/api-ws-server.md)
* [Message Broker](/docs/api-msg-broker.md)
* [Info API](/docs/api-info.md)
