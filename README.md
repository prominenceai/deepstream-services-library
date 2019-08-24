# on-target

## Installing Dependencies

### Apache Portable Runtime
Installing APR and APR util
```
sudo apt-get install apache2-dev libapr1-dev libaprutil1-dev
```
### Apache log4cxx
Source pulled from log4cxx-0.10.0 [trunk]( http://svn.apache.org/repos/asf/incubator/log4cxx/trunk)
```
$ cd apache-log4cxx-0.10.0
$ ./configure --build=x86_64-unknown-linux-gnu
$ make
$ make check
$ sudo make install
```
