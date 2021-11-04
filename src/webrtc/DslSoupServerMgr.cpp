/*
The MIT License

Copyright (c) 2021, Prominence AI, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in-
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "DslSoupServerMgr.h"

namespace DSL
{
    /**
     * @brief Ctor for the SignalingTransceiver class
     */
    SignalingTransceiver::SignalingTransceiver()
        : m_pConnection(NULL)
        , m_connectionState(DSL_SOCKET_CONNECTION_STATE_CLOSED)
        , m_pOffer(NULL)
        , m_pJsonParser(NULL)
        , m_closedSignalHandlerId(0)
        , m_messageSignalHandlerId(0)

    {
        LOG_FUNC();

        // New JSON Parser to use for the life of the Transceiver
        m_pJsonParser = json_parser_new();
        if (!m_pJsonParser)
        {
            LOG_ERROR("Failed to create new JSON Parser");
            throw;
        }

        g_mutex_init(&m_transceiverMutex);
    };

    /**
     * @brief Dtor for the SignalingTransceiver class
     */
    SignalingTransceiver::~SignalingTransceiver()
    {
        LOG_FUNC();
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_transceiverMutex);
            if (m_pJsonParser != NULL)
            {
                g_object_unref(G_OBJECT(m_pJsonParser));
            }
        }
        g_mutex_clear(&m_transceiverMutex);
    }

    const SoupWebsocketConnection* SignalingTransceiver::GetConnection()
    {
        LOG_FUNC();

        return m_pConnection;
    };

    void SignalingTransceiver::SetConnection(SoupWebsocketConnection* pConnection)
    {
        LOG_FUNC();

        m_pConnection = pConnection;

        if (pConnection)
        {
            // Need to add a reference so the object won't be freed on return.
            g_object_ref(G_OBJECT(pConnection));

            // Update the connection state
            m_connectionState = DSL_SOCKET_CONNECTION_STATE_INITIATED;

            // connect common callbacks to Websocked "closed" and "message" signals
            m_closedSignalHandlerId = g_signal_connect(G_OBJECT(m_pConnection), "closed", 
                G_CALLBACK(on_soup_websocket_closed_cb), (gpointer)this);
            m_messageSignalHandlerId = g_signal_connect(G_OBJECT(m_pConnection), "message", 
                G_CALLBACK(on_soup_websocket_message_cb), (gpointer)this);
        }
    };

    bool SignalingTransceiver::IsConnected()
    {
        LOG_FUNC();

        return (m_pConnection != NULL);
    };

    void SignalingTransceiver::ClearConnection()
    {
        LOG_FUNC();

        LOG_INFO("Clearing connections");

        if (m_closedSignalHandlerId)
        {
            g_signal_handler_disconnect(G_OBJECT(m_pConnection), m_closedSignalHandlerId);
            m_closedSignalHandlerId = 0;
        }
        if (m_messageSignalHandlerId)
        {
            g_signal_handler_disconnect(G_OBJECT(m_pConnection), m_messageSignalHandlerId);
            m_messageSignalHandlerId = 0;
        }
        m_pConnection = NULL;
        m_connectionState = DSL_SOCKET_CONNECTION_STATE_CLOSED;
    }

    void SignalingTransceiver::OnClosed(SoupWebsocketConnection* pConnection)
    {
        LOG_FUNC();

        ClearConnection();
    }

    void SignalingTransceiver::OnMessage(SoupWebsocketConnection* pConnection, 
        SoupWebsocketDataType dataType, GBytes* message)
    {        
        // NOTE: virtual base implementation for testing purposes only    
        LOG_ERROR("Derived class must implement On Message");
    }


    // ------------------------------------------------------------------------------
    // Client Reciever Websocket callback functions

    static void on_soup_websocket_closed_cb(SoupWebsocketConnection * pConnection, 
        gpointer pSignalingTransceiver)
    {
        static_cast<SignalingTransceiver*>(pSignalingTransceiver)->OnClosed(pConnection);
    }

    static void on_soup_websocket_message_cb(SoupWebsocketConnection* pConnection, 
        SoupWebsocketDataType dataType, GBytes* message, gpointer pSignalingTransceiver)
    {
        static_cast<SignalingTransceiver*>(pSignalingTransceiver)->OnMessage(pConnection,
            dataType, message);
    }

    // ------------------------------------------------------------------------------
    // Soup Server Implementation - Initialize the SoupServer single instance pointer
    SoupServerMgr* SoupServerMgr::m_pInstance = NULL;

    SoupServerMgr* SoupServerMgr::GetMgr()
    {
        // one time initialization of the single instance pointer
        if (!m_pInstance)
        {
            
            LOG_INFO("SoupServer Initialization");
            
            // Single instantiation for the lib's lifetime
            m_pInstance = new SoupServerMgr();
        }
        return m_pInstance;
    }

    SoupServerMgr::SoupServerMgr()
        : m_pSoupServer(NULL)
        , m_portNumber(0)
        , m_isListening(false)

    {
        LOG_FUNC();

        m_pSoupServer = soup_server_new(SOUP_SERVER_SERVER_HEADER, "webrtc", NULL);
        if (!m_pSoupServer)
        {
            LOG_ERROR("Websocket Server Manager failed to create new Server");
            throw;
        }

        soup_server_add_websocket_handler(m_pSoupServer, "/ws", NULL, NULL, 
            websocket_handler_cb, (gpointer)this, NULL);

        g_mutex_init(&m_serverMutex);
    }

    SoupServerMgr::~SoupServerMgr()
    {
        LOG_FUNC();

        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_serverMutex);
            if (m_pSoupServer)
            {
                g_object_unref(G_OBJECT(m_pSoupServer));        
            }
        }
        g_mutex_clear(&m_serverMutex);

    }

    bool SoupServerMgr::AddPath(const char* path)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_serverMutex);

        if (m_isListening)
        {
            LOG_ERROR("Unable to add Path '" << path 
                << "' to the Websocket Server when in a listening state");
            return false;
        }

        LOG_INFO("adding new websocket handler for Path: " << path );
        soup_server_add_websocket_handler(m_pSoupServer, path, NULL, NULL, 
            websocket_handler_cb, (gpointer)this, NULL);
            
        return true;
    }

    bool SoupServerMgr::StartListening(uint portNumber)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_serverMutex);

        if (m_isListening)
        {
            LOG_ERROR("Websocket Server Manager is already in a listening state");
            return false;
        }
        m_portNumber = portNumber;

        if (!soup_server_listen_all(m_pSoupServer, m_portNumber, 
            (SoupServerListenOptions) 0, NULL))
        {
            LOG_ERROR("Websocket Server Manager failed to listen on HTTP port: " 
                << m_portNumber);
            return false;
        }

        // Get and log the list of URIs the server is listening on
        GSList* uris = soup_server_get_uris(m_pSoupServer);

        for(GSList* uri = uris; uri; uri = uri->next) 
        {
            char* uriStr = soup_uri_to_string((SoupURI*)uri->data, 0);
            LOG_INFO("Websocket Server listening on " << uriStr);
            g_free(uriStr);
            soup_uri_free((SoupURI*)uri->data);
        }
        g_slist_free(uris);

        m_isListening = true;
        return true;
    }

    bool SoupServerMgr::StopListening()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_serverMutex);

        if (!m_isListening)
        {
            LOG_ERROR("Websocket Server Manager is not in a listening state");
            return false;
        }
        m_portNumber = 0;

        soup_server_disconnect(m_pSoupServer);
        m_isListening = false;
        return true;
    }

    bool SoupServerMgr::GetListeningState(uint* portNumber)
    {
        LOG_FUNC();

        *portNumber = m_portNumber;
        return m_isListening;
    }

    bool SoupServerMgr::AddSignalingTransceiver(SignalingTransceiver* pSignalingTransceiver)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_serverMutex);
        
        if (HasSignalingTransceiver(pSignalingTransceiver))
        {
            LOG_ERROR("Client Reciever is already a child of the Websocket Server Manager");
            return false;
        }

        // Add the client receiver while initializing it's connection to NULL
        m_signalingTransceivers.push_back(pSignalingTransceiver);
                        
        LOG_INFO("Client Reciever added to the Websocket Server Manager successfully");
        
        return true;
    }
    
    bool SoupServerMgr::RemoveSignalingTransceiver(SignalingTransceiver* pSignalingTransceiver)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_serverMutex);
        
        if (!HasSignalingTransceiver(pSignalingTransceiver))
        {
            LOG_ERROR("Client Reciever was never added to the Websocket Server Manager");
            return false;
        }

        // TODO - close connection if open.

        m_signalingTransceivers.erase(
            std::remove(m_signalingTransceivers.begin(), m_signalingTransceivers.end(), 
                pSignalingTransceiver), m_signalingTransceivers.end());

        LOG_INFO("Client Reciever removed from the Websocket Server Manager successfully");
        
        return true;
    }

    bool SoupServerMgr::HasSignalingTransceiver(SignalingTransceiver* pSignalingTransceiver)
    {
        LOG_FUNC();

        return (std::find(m_signalingTransceivers.begin(), 
            m_signalingTransceivers.end(), pSignalingTransceiver) != m_signalingTransceivers.end());
    }

    // const SignalingTransceiver* SoupServerMgr::GetSignalingTransceiver(SoupWebsocketConnection* pConnection)
    // {
    //     LOG_FUNC();
    //     LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_serverMutex);

    //     for (auto &imap: m_signalingTransceivers)
    //     {
    //         if (imap.second == pConnection)
    //         {
    //             return imap.first;
    //         }
    //     }
    //     // Connection was not found
    //     LOG_WARN("A WebRTC client-bin was not found for Connection "
    //         << std::to_string((uint64_t)pConnection));
    //     return NULL;
    // }

    bool SoupServerMgr::AddClientListener(
        dsl_websocket_server_client_listener_cb listener, void* clientData)
    {
        LOG_FUNC();

        if (m_clientListeners.find(listener) != m_clientListeners.end())
        {   
            LOG_ERROR("Client listener is not unique for the Websocket Server Manager");
            return false;
        }
        m_clientListeners[listener] = clientData;
        
        return true;
    }
    
    bool SoupServerMgr::RemoveClientListener(
        dsl_websocket_server_client_listener_cb listener)
    {
        LOG_FUNC();

        if (m_clientListeners.find(listener) == m_clientListeners.end())
        {   
            LOG_ERROR("Client listener was not found with the Websocket Server Manager");
            return false;
        }
        m_clientListeners.erase(listener);
        
        return true;
    }


    void SoupServerMgr::HandleOpen(SoupWebsocketConnection* pConnection, const char* path)
    {
        LOG_FUNC();

        // If we have registered client listeners, call them first to allow
        // the client to add a WebRTC Signaling Transciever based on the path
        if (m_clientListeners.size())
        {
            // Convert path to widestring 
            std::string pathString(path);
            std::wstring wPathString(pathString.begin(), pathString.end());

            // iterate through the map of client listeners calling each
            for(auto const& imap: m_clientListeners)
            {
                try
                {
                    imap.first(wPathString.c_str(), imap.second);
                }
                catch(...)
                {
                    LOG_ERROR("Exception calling Web RTC Sink Client Listener");
                }
            }
        }

        // Look for first unconnected transceiver
        for (auto &ivec: m_signalingTransceivers)
        {
            if (!ivec->IsConnected())
            {
                LOG_INFO("Unused Signaling Transceiver found");

                ivec->SetConnection(pConnection);

                return;
            }
        }
        LOG_ERROR("Websocket Server Manger found no available Signaling Transceivers.");
    }


    static void websocket_handler_cb(G_GNUC_UNUSED SoupServer* pServer, 
        SoupWebsocketConnection* pConnection, const char *path,
        G_GNUC_UNUSED SoupClientContext* clientContext, gpointer pSoupServerMgr)
    {
        LOG_INFO("Incomming connection.");
        static_cast<SoupServerMgr*>(pSoupServerMgr)->HandleOpen(
            pConnection, path);
    }

}