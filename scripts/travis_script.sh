#! /bin/bash -ex

cd build
cmake --build .
ctest
if [ "$TRAVIS_OS_NAME" = "linux" ]; then
    cpack -G DEB
fi