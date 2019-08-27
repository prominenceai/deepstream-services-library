/*
The MIT License

Copyright (c) 2019-Present, ROBERT HOWELL

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

#ifndef _DSS_LOG_H
#define _DSS_LOG_H


/**
Logs the Entry and Exit of a Function with the DEBUG level.
 * Add macro as the first statement to each function of interest.
 * Consider the intrussion/penalty of this call when adding.

@param message the message string to log.
*/
#define LOG_FUNC()

/**
Logs a message with the DEBUG level.

*/
#define LOG_DEBUG(message)

/**
Logs a message with the INFO level.

@param message the message string to log.
*/
#define LOG_INFO(message)

/**
Logs a message with the WARN level.

@param message the message string to log.
*/
#define LOG_WARN(message)

/**
Logs a message with the ERROR level.

@param message the message string to log.
*/
#define LOG_ERROR(message)

/**
Logs a message with the FATAL level.

@param message the message string to log.
*/
#define LOG_FATAL(message)

#endif // _DSS_LOG_H