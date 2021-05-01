# SMTP Services API
This API is used to setup your SMTP Server URL, Credentials, and Email Addresses for sending email using an [ODE Email Action](/docs/api-action.md#dsl_ode_action_email_new) during and Object Detection Event (ODE) occurrence. Queuing of the event data occurs in the Action's real time context, while the tasks of assembling the message and uploading to the SMTP server are performed in a low priority background thread. Messages that fail to send will be purged from the queue, dropped and logged as an ERROR.

## Using GMAIL's SMTP server
**IMPORTANT!** it is STRONGLY advised that you create a new, free [Gmail account](https://support.google.com/mail/answer/56256?hl=en) -- that is seperate/unlinked from all your other email accounts -- strictly for the purpose of sending ODE Event data uploaded from DSL.  Then, add your Personal email address as a `To` address to
receive the emails. 

Gmail considers regular email programs (i.e Outlook, etc.) and non-registered third-party apps to be "less secure". The email account used for sending email must have the "Allow less secure apps" option turned on. Once you've created this new account, you can go to the account settings and enable [Less secure app access](https://myaccount.google.com/lesssecureapps).

The Gmail secure SMTP server URL is `smtps://smtp.gmail.com:465`. Port `465` requires SSL to be enabled which is set by default. See [dsl_smtp_ssl_enabled_get](#dsl_smtp_ssl_enabled_get) and [dsl_smtp_ssl_enabled_set](#dsl_smtp_ssl_enabled_set)

### Example setup using Python: 
```Python
retval = dsl_smtp_server_url_set('smtps://smtp.gmail.com:465')

# Using the credentials for your new Gmail SMTP account
retval += dsl_smtp_credentials_set(user_name, password)

# Set the From email address, again using the new SMTP account.
retval += dsl_smtp_address_from_set('', 'my.smtp.server@gmail.com')

# Add your personal email account as a To address. 
# Optionally, add other To and Cc addresses.
retval += dsl_smtp_address_to_add(to_name, to_address)
        
# queue a test message to be sent out to ensure all settings are correct
retval += dsl_smtp_test_message_send()

# ensure that all sercies were successful.
if reval != DSL_RESULT_SUCCESS:

```

## SMTP API
**Functions**
* [dsl_smtp_mail_enabled_get](#dsl_smtp_mail_enabled_get)
* [dsl_smtp_mail_enabled_set](#dsl_smtp_mail_enabled_set)
* [dsl_smtp_credentials_set](#dsl_smtp_credentials_set)
* [dsl_smtp_server_url_get](#dsl_smtp_server_url_get)
* [dsl_smtp_server_url_set](#dsl_smtp_server_url_set)
* [dsl_smtp_ssl_enabled_get](#dsl_smtp_ssl_enabled_get)
* [dsl_smtp_ssl_enabled_set](#dsl_smtp_ssl_enabled_set)
* [dsl_smtp_address_from_get](#dsl_smtp_address_from_get)
* [dsl_smtp_address_from_set](#dsl_smtp_address_from_set)
* [dsl_smtp_address_to_add](#dsl_smtp_address_to_add)
* [dsl_smtp_address_to_remove_all](#dsl_smtp_address_to_remove_all)
* [dsl_smtp_address_cc_add](#dsl_smtp_address_cc_add)
* [dsl_smtp_address_cc_remove_all](#dsl_smtp_address_cc_remove_all)
* [dsl_smtp_test_message_send](#dsl_smtp_test_message_send)

## Return Values
The following return codes are used by the SMTP API
```C++
#define DSL_RESULT_SUCCESS                                          0x00000000
#define DSL_RESULT_FAILURE                                          0x00000001
#define DSL_RESULT_INVALID_INPUT_PARAM                              0x00000003
#define DSL_RESULT_THREW_EXCEPTION                                  0x00000004
```

## Constants
The following constant values are used by the SMTP API
```C
#define DSL_SMTP_MAX_PENDING_MESSAGES                               10
```

<br>

---

## Functions

### *dsl_smtp_mail_enabled_get*
```C++
DslReturnType dsl_smtp_mail_enabled_get(boolean* enabled);
```
This services queries the SMTP services object for its current enabled state. Services are automatically disabled if and while the outgoing queue size exceeds `DSL_SMTP_MAX_PENDING_MESSAGES`. Services are enabled by default.

**Parameters**
* `enabled` [out] true if SMTP services are currently enabled, false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful call. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, enabled = dsl_smtp_mail_enabled_get()
```

<br>

### *dsl_smtp_mail_enabled_set*
```C++
DslReturnType dsl_smtp_mail_enabled_set(boolean enabled);
```
This services sets the enabled state for the SMTP services object. Setting the state to true while the outgoing queue size exceeds `DSL_SMTP_MAX_PENDING_MESSAGES` will return `DSL_RESULT_FAILURE`. Services are enabled by default.

**Parameters**
* `enabled` [in] set to true to enable SMTP services, false to disable.

**Returns**
* `DSL_RESULT_SUCCESS` on successful call. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_smtp_mail_enabled_set(True)
```

<br>

### *dsl_smtp_credentials_set*
```C++
DslReturnType dsl_smtp_credentials_set(const wchar_t* username, 
    const wchar_t* password);
```
This service is used to set the SMTP account credentials, username and password, for all subsequent emails sent.

**Parameters**
* `username` [in] username for the SMTP account
* `password` [in] password for the same account

**Returns**
* `DSL_RESULT_SUCCESS` on successful call. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_smtp_credentials_set('my-dsl-smtp-account', 'my-dsl-smtp-password')
```

<br>

### *dsl_smtp_server_url_get*
```C++
DslReturnType dsl_smtp_server_url_get(const wchar_t** server_url);
```
This service gets the current Server URL in use. The service will return an empty string unless set by a previous call made to [dsl_smtp_server_url_set](#dsl_smtp_server_url_set).

**Parameters**
* `server_url` [out] current Server URL in use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful call. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, server_url = dsl_smtp_server_url_get(')
```

<br>

### *dsl_smtp_server_url_set*
```C++
DslReturnType dsl_smtp_server_url_get(const wchar_t** server_url);
```
This service sets the Server URL to use for all subsequent emails sent.

**Parameters**
* `server_url` [in] new Server URL to use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful call. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_smtp_server_url_set('smtps://smtp.gmail.com:465')
```

<br>

### *dsl_smtp_address_from_get*
```C++
DslReturnType dsl_smtp_address_from_get(const wchar_t** name,
    const wchar_t** address);
```
This service gets the current `From` address in use by SMTP services, values that were previously set with a call to [dsl_smtp_address_from_set](#dsl_smtp_address_from_set).

**Parameters**
* `name` [out] returns the display name (optional) for the `From` address in use
* `address` [out] returns the email address in use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful call. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, display_name, address = dsl_smtp_address_from_get()
```

<br>

### *dsl_smtp_address_from_set*
```C++
DslReturnType dsl_smtp_address_from_get(const wchar_t* name,
    const wchar_t* address);
```
This service sets the `From` address to use for all subsequent emails.

**Parameters**
* `name` [in] the display name (optional) to use for the `From` address
* `address` [in] the email address to use. Will be set by the server if omitted.

**Returns**
* `DSL_RESULT_SUCCESS` on successful call. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval  = dsl_smtp_address_from_set('Joe Blow', 'joe.blow@example.com')
```

<br>

### *dsl_smtp_ssl_enabled_get*
```C++
DslReturnType dsl_smtp_ssl_enabled_get(boolean* enabled);
```
This service gets the SSL enabled setting for the SMTP services. SSL is enabled by default

**Parameters**
* `enabled` [out] true if SSL is currently enabled, false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful call. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, enabled = dsl_smtp_ssl_enabled_get()
```

<br>

### *dsl_smtp_ssl_enabled_set*
```C++
DslReturnType dsl_smtp_ssl_enabled_set(boolean enabled);
```
This service sets the SSL enabled state for SMTP Services. SSL is enabled by default

**Parameters**
* `enabled` [in] set to true to enable SSL, false to disable.

**Returns**
* `DSL_RESULT_SUCCESS` on successful call. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_smtp_ssl_enabled_set(True)
```

<br>

### *dsl_smtp_address_to_add*
```C++
DslReturnType dsl_smtp_address_to_add(const wchar_t* name,
    const wchar_t* address);
```
This service adds a `To` address to use for all subsequent emails.

**Parameters**
* `name` [in] the display name (optional) to use for the `To` address
* `address` [in] the email address to add. 

**Returns**
* `DSL_RESULT_SUCCESS` on successful call. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval  = dsl_smtp_address_to_add('Jane Doe', 'jane.doe@example.com')
```

<br>

### *dsl_smtp_address_to_remove_all*
```C++
DslReturnType dsl_smtp_address_to_remove_all();
```
This service removes all `To` addresses from memory. At least one `To` address must exist for email to be sent out.

**Returns**
* `DSL_RESULT_SUCCESS` on successful call. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval  = dsl_smtp_address_to_remove_all()
```

<br>

### *dsl_smtp_address_cc_add*
```C++
DslReturnType dsl_smtp_address_cc_add(const wchar_t* name,
    const wchar_t* address);
```
This service adds a `Cc` address to use for all subsequent emails. `Cc` addresses are optional.

**Parameters**
* `name` [in] the display name (optional) to use for the `Cc` address
git* `address` [in] the email address to add. 

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval  = dsl_smtp_address_cc_add('Jane Doe', 'jane.doe@example.com')
```

<br>

### *dsl_smtp_address_cc_remove_all*
```C++
DslReturnType dsl_smtp_address_to_remove_all();
```
This service removes all `Cc` addresses from memory. `Cc` addresses are optional.

**Returns**
* `DSL_RESULT_SUCCESS` on successful call. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval  = dsl_smtp_address_to_remove_all()
```

<br>

### *dsl_smtp_test_message_send*
```C++
DslReturnType dsl_smtp_test_message_send();
```
This service sends a test message using the current SMTP settings and email addresses: `From`, `To`, and `Cc`.

**Returns**
* `DSL_RESULT_SUCCESS` on successful queue. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval  = dsl_smtp_test_message_send()
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
* [Primary and Secondary GIE](/docs/api-gie.md)
* [Tracker](/docs/api-tracker.md)
* [On-Screen Display](/docs/api-osd.md)
* [Tiler](/docs/api-tiler.md)
* [Splitter and Demuxer](/docs/api-tee.md)
* [Sink](/docs/api-sink.md)
* [Pad Probe Handler](/docs/api-pad-probe-handler.md)
* [ODE Trigger](/docs/api-ode-trigger.md)
* [ODE Acton](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* [Display Type](/docs/api-display-type.md)
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
* **SMTP Services**
