#! /bin/bash

if [ "$COVERAGE" = "ON" ]; then
    echo "Uploading coverage to codecov.io"
    bash <(curl -s https://codecov.io/bash)
fi

