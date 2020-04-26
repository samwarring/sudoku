#! /bin/bash

export CMAKE_ARGS=""
if [ -z $COVERAGE ]; then
    if [ -z $CONFIG ]; then
        echo "error: Missing required environment variable: CONFIG"
        exit 1
    fi
    echo "Configuring for $CONFIG"
    export CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_BUILD_TYPE=$CONFIG"
elif [ "$COVERAGE" = "ON" ]; then
    echo "Configuring for code coverage"
    export CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_BUILD_TYPE=Debug"
    export CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CXX_FLAGS=--coverage"
else
    echo "error: unrecognized environment variable combination" exit 1
fi

mkdir build
cd build
cmake .. $CMAKE_ARGS

