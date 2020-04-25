#! /bin/bash

if [ "$COVERAGE" = "ON" ]; then
    echo "Generating coverage.xml"
    gcovr -r .. . --xml-pretty > coverage.xml
    echo "Uploading coverage to codecov.io"
    bash <(curl -s https://codecov.io/bash)
fi

