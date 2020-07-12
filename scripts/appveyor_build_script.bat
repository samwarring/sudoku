pushd build
cmake --build . --config %CONFIGURATION%
cpack -G WIX -C %CONFIGURATION%
popd
